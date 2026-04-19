"""End-to-end parity check: demo vs upstream evaluate_prod_xt.py.

Picks N small pages from data/eu_filings/processed-multi/mixed_val, runs
inference through the demo's code path, then scores with BOTH:
  1. The demo's xt_extract.match_entities / aggregate_typed (same file used in UI).
  2. The upstream evaluate_xt.match_entities / aggregate (imported from the
     training repo).

Prints side-by-side. They must match exactly.

Usage:
    /dev/shm/vllm/bin/python scripts/parity_check.py --n 5 --port 8000 --model phase2
"""

import argparse
import asyncio
import base64
import io
import json
import os
import random
import sys
from pathlib import Path

# Allow importing the demo's src/ helpers.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Allow importing the training repo's scripts/evaluate_xt helpers.
UPSTREAM_ROOT = Path("/home/compiling-ganesh/24m0797/workspace/ixbrl-tagging")
sys.path.insert(0, str(UPSTREAM_ROOT))

from PIL import Image
import requests

from utils.xt_extract import (
    PROMPT_PROD_XT_UNIFIED,
    SYSTEM_PROMPT,
    aggregate as demo_aggregate,
    aggregate_typed as demo_aggregate_typed,
    classify_entities,
    match_entities as demo_match,
    parse_xbrl_tags as demo_parse,
    typed_metrics_for_page,
)

# Upstream (authoritative)
from scripts.evaluate_xt import (
    match_entities as up_match,
    aggregate as up_aggregate,
    parse_xbrl_tags as up_parse,
)
from scripts.evaluate_prod_xt import (
    load_gt_spaced as up_load_gt,
    aggregate_typed as up_aggregate_typed,
)

DATA_ROOT = UPSTREAM_ROOT / "data/eu_filings/processed-multi/mixed_val"
MAX_IMAGE_DIM = 1568


def img_b64(path: Path) -> str:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_IMAGE_DIM:
        s = MAX_IMAGE_DIM / max(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def call(model, port, b64, input_text, year, max_new=12000, timeout=300):
    full = f"Filing year: {year}\n\n{input_text}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": PROMPT_PROD_XT_UNIFIED + "\n\n" + full[:24000]},
            ]},
        ],
        "max_tokens": max_new,
        "temperature": 0,
    }
    r = requests.post(f"http://localhost:{port}/v1/chat/completions",
                      json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def fmt(d, keys):
    return "  ".join(f"{k}={d.get(k, 0):.3f}" if isinstance(d.get(k), float)
                     else f"{k}={d.get(k, 0)}" for k in keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--model", default="phase2")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    manifest = json.loads((DATA_ROOT / "manifest.json").read_text())
    pages = manifest["pages"]
    rng = random.Random(args.seed)
    rng.shuffle(pages)
    chosen = []
    for p in pages:
        stem = p["stem"]
        if all((DATA_ROOT / f"{stem}{ext}").exists()
               for ext in (".png", "-input.txt", "-output.txt")):
            chosen.append(p)
        if len(chosen) >= args.n:
            break

    print(f"Parity-checking {len(chosen)} pages against model {args.model}:{args.port}")
    print("=" * 90)

    per_page_typed_demo = []
    per_page_typed_up = []

    for p in chosen:
        stem = p["stem"]
        year = p["filing_year"]
        input_text = (DATA_ROOT / f"{stem}-input.txt").read_text(errors="ignore")
        output_text = (DATA_ROOT / f"{stem}-output.txt").read_text(errors="ignore")
        b64 = img_b64(DATA_ROOT / f"{stem}.png")

        resp = call(args.model, args.port, b64, input_text, year)

        # DEMO side
        gt_demo = __import__("utils.xt_extract", fromlist=["load_gt_from_output"]).load_gt_from_output(output_text)
        pred_demo = demo_parse(resp)
        typed_demo = typed_metrics_for_page(gt_demo, pred_demo)
        per_page_typed_demo.append(typed_demo)

        # UPSTREAM side
        gt_up = up_load_gt(DATA_ROOT / f"{stem}-output.txt")
        pred_up = up_parse(resp)
        overall_up = up_match(gt_up, pred_up, num_soft=False, text_soft=True)
        gn, gt_ = [e for e in gt_up if e.get("unit")], [e for e in gt_up if not e.get("unit")]
        pn, pt = [e for e in pred_up if e.get("unit")], [e for e in pred_up if not e.get("unit")]
        numeric_up = up_match(gn, pn, num_soft=False, text_soft=False)
        text_up = up_match(gt_, pt, num_soft=False, text_soft=True)
        per_page_typed_up.append({"overall": overall_up, "numeric": numeric_up, "text": text_up})

        print(f"\n-- {stem} (year {year}) --")
        for k in ("overall", "numeric", "text"):
            d = typed_demo[k]; u = per_page_typed_up[-1][k]
            print(f"  {k:7s} DEMO    : {fmt(d, ['tp','fp','fn','precision','recall','f1','concept_hits','year_hits','unit_hits','scale_hits'])}")
            print(f"  {k:7s} UPSTREAM: {fmt(u, ['tp','fp','fn','precision','recall','f1','concept_hits','year_hits','unit_hits','scale_hits'])}")

    print("\n" + "=" * 90)
    print("HOLISTIC\n")

    # Demo aggregation
    demo_hol = demo_aggregate_typed(per_page_typed_demo)
    # Upstream aggregation — mirror evaluate_prod_xt: feed per-page overall metrics + gt/pred lists
    up_overall = up_aggregate([pg["overall"] for pg in per_page_typed_up])
    up_num = up_aggregate([pg["numeric"] for pg in per_page_typed_up
                           if pg["numeric"]["n_gt"] + pg["numeric"]["n_pred"] > 0])
    up_text = up_aggregate([pg["text"] for pg in per_page_typed_up
                            if pg["text"]["n_gt"] + pg["text"]["n_pred"] > 0])
    up_hol = {"overall": up_overall, "numeric": up_num, "text": up_text}

    keys = ["tp", "fp", "fn", "precision", "recall", "f1",
            "concept_acc", "year_acc", "unit_acc", "scale_acc"]
    for k in ("overall", "numeric", "text"):
        print(f"{k:7s} DEMO    : {fmt(demo_hol[k], keys)}")
        print(f"{k:7s} UPSTREAM: {fmt(up_hol[k], keys)}")
        print()


if __name__ == "__main__":
    main()
