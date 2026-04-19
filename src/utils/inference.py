"""Inference utilities for iXBRL tagging client.

Adapted from tagger/inference/inference.py — no tagger package dependency.
"""

import asyncio
import io
import base64
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Union, Optional

import aiohttp
from PIL import Image
from tqdm import tqdm


DEFAULT_API_BASE = "http://localhost:8000/v1"
DEFAULT_MODEL_NAME = "phase1_prod"
DEFAULT_MAX_TOKENS = 10000
DEFAULT_API_KEY = "EMPTY"
DEFAULT_MAX_IMAGE_DIM = 1568


def _resize_image(img: Image.Image, max_dim: int) -> Image.Image:
    """Downscale image so its longest side is at most max_dim pixels."""
    w, h = img.size
    if max(w, h) <= max_dim:
        return img
    scale = max_dim / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)


def _image_to_base64(image: Union[str, Path, Image.Image], max_dim: int = DEFAULT_MAX_IMAGE_DIM) -> str:
    """Convert an image path or PIL Image to a base64-encoded PNG string."""
    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        img = image.convert("RGB")
    else:
        raise TypeError(f"Expected str, Path, or PIL.Image, got {type(image)}")
    img = _resize_image(img, max_dim)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _make_image_content(image: Union[str, Path, Image.Image]) -> dict:
    """Create an OpenAI-style image_url content block from an image."""
    b64 = _image_to_base64(image)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{b64}"},
    }


def cap_repetitions(entities: list[dict], max_per_value: int = 10) -> list[dict]:
    """Remove excess repeated predictions where the model loops on a single value."""
    from collections import defaultdict
    count = defaultdict(int)
    result = []
    for e in entities:
        key = str(e.get("value", ""))
        if count[key] < max_per_value:
            result.append(e)
            count[key] += 1
    return result


def parse_json_response(text: str) -> list[dict]:
    """Parse model output into a list of entity dicts, handling markdown fences
    and truncated JSON (e.g. when max_tokens cuts off the output)."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    # Strip CoT prefix lines (Scale: ..., Year columns: ...) before JSON
    lines = cleaned.splitlines()
    json_start = next((i for i, l in enumerate(lines) if l.strip().startswith("[")), None)
    if json_start is not None and json_start > 0:
        cleaned = "\n".join(lines[json_start:])
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    objects = []
    for m in re.finditer(r"\{[^{}]*\}", cleaned):
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict):
                objects.append(obj)
        except json.JSONDecodeError:
            continue
    return objects


def _normalize_messages(messages: list[dict]) -> list[dict]:
    """Convert image shortcuts to proper base64 image_url blocks."""
    normalized = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            new_content = []
            for item in content:
                if item.get("type") == "image":
                    new_content.append(_make_image_content(item["image"]))
                elif item.get("type") == "image_url" and "image_url" in item:
                    url = item["image_url"].get("url", "")
                    if not url.startswith("data:"):
                        new_content.append(_make_image_content(url))
                    else:
                        new_content.append(item)
                else:
                    new_content.append(item)
            normalized.append({**msg, "content": new_content})
        else:
            normalized.append(msg)
    return normalized


def _build_url(api_base: str) -> str:
    """Build the chat completions URL, appending /v1 if needed."""
    url = f"{api_base.rstrip('/')}/chat/completions"
    if "/v1/" not in url and not url.endswith("/v1"):
        url = f"{api_base.rstrip('/')}/v1/chat/completions"
    return url


def _build_payload(messages, model_name, max_tokens, temperature, top_p):
    return {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }


def _build_headers(api_key):
    headers = {}
    if api_key and api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _build_shot_messages(prompt, image, shot_list):
    """Build OpenAI-style messages from prompt, image, and optional shots."""
    msgs = []
    if shot_list:
        for shot_image, shot_response in shot_list:
            user_content = [{"type": "text", "text": prompt}]
            if shot_image is not None:
                user_content.append({"type": "image", "image": shot_image})
            msgs.append({"role": "user", "content": user_content})
            msgs.append({
                "role": "assistant",
                "content": [{"type": "text", "text": shot_response}],
            })
    user_content = [{"type": "text", "text": prompt}]
    if image is not None:
        user_content.append({"type": "image", "image": image})
    msgs.append({"role": "user", "content": user_content})
    return msgs


async def infer_xt_batch_async(
    images: list[Union[str, Path, Image.Image]],
    input_texts: list[str],
    prompt: str,
    system_prompt: str,
    model_name: str = DEFAULT_MODEL_NAME,
    api_base: str = DEFAULT_API_BASE,
    api_key: str = DEFAULT_API_KEY,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
    top_p: float = 0.9,
    timeout: int = 600,
    max_concurrent: int = 32,
    max_workers: int = 32,
    max_input_chars: int = 24000,
    encode_progress_fn=None,
    infer_progress_fn=None,
) -> list[str]:
    """XT-style inference: one image per page + extracted-text user block.

    Matches the request shape of scripts/evaluate_prod_xt.call_unified.
    """
    assert len(images) == len(input_texts), "images and input_texts must align"
    url = _build_url(api_base)
    headers = _build_headers(api_key)

    def _prepare(idx):
        img = images[idx]
        txt = input_texts[idx][:max_input_chars]
        image_block = _make_image_content(img)
        text_block = {"type": "text", "text": f"{prompt}\n\n{txt}"}
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [image_block, text_block]},
        ]
        return idx, _build_payload(messages, model_name, max_tokens, temperature, top_p)

    n_workers = min(max_workers, len(images)) or 1
    payloads = [None] * len(images)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for done, (idx, payload) in enumerate(tqdm(
            pool.map(lambda i: _prepare(i), range(len(images))),
            total=len(images), desc="Encoding",
        ), start=1):
            payloads[idx] = payload
            if encode_progress_fn:
                encode_progress_fn(done, len(images))

    sem = asyncio.Semaphore(max_concurrent)
    results: list = [None] * len(images)
    pbar = tqdm(total=len(images), desc="Inference")

    async def _send(idx, payload, session):
        async with sem:
            try:
                async with session.post(url, json=payload, headers=headers,
                                        timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                    if resp.status >= 400:
                        body = await resp.text()
                        print(f"\n  [WARNING] idx={idx} HTTP {resp.status}: {body[:300]}")
                        pbar.update(1)
                        return idx, None
                    data = await resp.json()
                    choice = data["choices"][0]
                    finish_reason = choice.get("finish_reason", "")
                    if finish_reason == "length":
                        print(f"\n  [WARNING] idx={idx} finish_reason=length — output truncated")
                    pbar.update(1)
                    if infer_progress_fn:
                        infer_progress_fn(pbar.n, len(images))
                    return idx, choice["message"]["content"]
            except Exception as e:
                print(f"\n  [WARNING] idx={idx} exception: {type(e).__name__}: {e}")
                pbar.update(1)
                return idx, None

    async with aiohttp.ClientSession() as session:
        tasks = [_send(i, p, session) for i, p in enumerate(payloads)]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

    pbar.close()

    for item in completed:
        if isinstance(item, Exception):
            print(f"\n  [WARNING] async task raised {type(item).__name__}: {item!r}")
            continue
        idx, response = item
        results[idx] = response

    return results


async def infer_batch_async(
    images: list[Union[str, Path, Image.Image]],
    prompt: str,
    shots: Optional[list[tuple]] = None,
    per_image_shots: Optional[list[list[tuple]]] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    api_base: str = DEFAULT_API_BASE,
    api_key: str = DEFAULT_API_KEY,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
    top_p: float = 0.9,
    timeout: int = 600,
    max_concurrent: int = 500,
    max_workers: int = 64,
    encode_progress_fn=None,
    infer_progress_fn=None,
) -> list[str]:
    """Run inference on multiple images concurrently (hybrid threads + async).

    Phase 1: Pre-encode all images to base64 in parallel using threads (CPU-bound).
    Phase 2: Fire all HTTP requests via aiohttp with semaphore (IO-bound).
    """
    url = _build_url(api_base)
    headers = _build_headers(api_key)

    # Phase 1: Pre-encode images + build payloads in parallel threads
    def _prepare(idx):
        img = images[idx]
        s = per_image_shots[idx] if per_image_shots else shots
        messages = _build_shot_messages(prompt, img, s)
        normalized = _normalize_messages(messages)
        return idx, _build_payload(normalized, model_name, max_tokens, temperature, top_p)

    n_workers = min(max_workers, len(images))
    payloads = [None] * len(images)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for done, (idx, payload) in enumerate(tqdm(
            pool.map(lambda i: _prepare(i), range(len(images))),
            total=len(images), desc="Encoding",
        ), start=1):
            payloads[idx] = payload
            if encode_progress_fn:
                encode_progress_fn(done, len(images))

    # Phase 2: Fire all HTTP requests async with bounded concurrency
    sem = asyncio.Semaphore(max_concurrent)
    results = [None] * len(images)
    pbar = tqdm(total=len(images), desc="Inference")

    async def _send(idx, payload, session):
        async with sem:
            async with session.post(url, json=payload, headers=headers,
                                    timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    print(f"\n  [WARNING] idx={idx} HTTP {resp.status}: {body[:300]}")
                    pbar.update(1)
                    return idx, None
                data = await resp.json()
                choice = data["choices"][0]
                finish_reason = choice.get("finish_reason", "")
                if finish_reason == "length":
                    print(f"\n  [WARNING] idx={idx} finish_reason=length — output truncated")
                pbar.update(1)
                if infer_progress_fn:
                    infer_progress_fn(pbar.n, len(images))
                return idx, choice["message"]["content"]

    async with aiohttp.ClientSession() as session:
        tasks = [_send(i, p, session) for i, p in enumerate(payloads)]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

    pbar.close()

    for item in completed:
        if isinstance(item, Exception):
            print(f"\n  [WARNING] async task raised {type(item).__name__}: {item!r}")
            continue
        idx, response = item
        results[idx] = response

    return results
