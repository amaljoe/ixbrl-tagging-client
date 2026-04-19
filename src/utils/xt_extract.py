"""XT extraction + evaluation utilities for prod-unified-xt model.

Ported from ../ixbrl-tagging/scripts/build_xt_dataset.py and evaluate_xt.py
so the client demo produces identical inputs, GT, and metrics as the upstream
evaluate_prod_xt.py pipeline.

Three stages handled here:
  1. Build input/output plain-text for one rendered page from its HTML.
  2. Parse <xbrl>-tagged model response → entity list.
  3. Match GT vs pred entities (typed soft/hard, concept/year/unit/scale).
"""

from __future__ import annotations

import difflib
import re
from collections import Counter
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup, NavigableString, Tag

from utils.concept_map import concept_to_spaced


# ── Prompt (verbatim from tagger.pipelines.prompt_configs.PROMPT_PROD_XT_UNIFIED) ──

PROMPT_PROD_XT_UNIFIED = (
    "You are an IFRS financial data extraction system.\n\n"
    "You are given:\n"
    "1. Rendered screenshots of 1 or 2 consecutive pages from an ESEF iXBRL financial filing.\n"
    "2. The extracted text content of those pages, starting with 'Filing year: YYYY'"
    " and separated by '--- PAGE 2 ---' if 2 pages.\n\n"
    "Task: Return the same text with every tagged entity on the FIRST page wrapped in <xbrl> tags."
    " If an entity from the first page continues onto the second page, tag the continuation too."
    " Do NOT tag entities that start on the second page.\n\n"
    "For numeric values:\n"
    '  <xbrl id="N" concept="ifrs-full: revenue" year="YYYY" unit="CUR" scale="S">value</xbrl>\n\n'
    "For text disclosures:\n"
    '  <xbrl id="N" concept="ifrs-full: name of reporting entity" year="YYYY">text span</xbrl>\n\n'
    "Rules:\n"
    "- Use the provided filing year for the year attribute unless the entity clearly refers to a prior year.\n"
    "- Tag ALL entities on the first page: numeric values AND text disclosures.\n"
    "- Preserve all text, spacing, and line breaks exactly. No other output."
)

SYSTEM_PROMPT = "You are an IFRS financial data extraction system."

PAGE_SEP = "\n--- PAGE 2 ---\n"


# ── Regex patterns ──────────────────────────────────────────────────────────

_MERGE_XBRL_RE = re.compile(
    r'<xbrl(\s[^>]*)id="(\d+)"([^>]*)>(.*?)</xbrl>[ \t]*\n[ \t]*<xbrl\s+id="\2">',
    re.DOTALL,
)
_RE_XBRL_FULL = re.compile(r'<xbrl\s+([^>]*)>(.*?)</xbrl>', re.DOTALL)
_RE_XBRL_OPEN = re.compile(r'<xbrl\s+([^>]*)>')
_RE_CONCEPT_ATTR = re.compile(r'concept="([^"]*)"')
_RE_ID_ATTR = re.compile(r'id="(\d+)"')


# ── Continuation maps for a single page ──────────────────────────────────────

def build_continuation_maps(
    page_soup: BeautifulSoup,
    all_cont_ids: set,
) -> tuple[dict, dict, dict]:
    page_cont_ids = {c.get("id", "") for c in page_soup.find_all("ix:continuation")}
    page_conts = {c.get("id", ""): c for c in page_soup.find_all("ix:continuation")}

    entities = page_soup.find_all(["ix:nonfraction", "ix:nonnumeric"])
    entity_id_map: dict = {}
    cont_to_entity: dict = {}

    for seq_i, ent in enumerate(entities):
        cur = ent.get("continuedat") or ent.get("continuedAt") or ""
        while cur and cur in page_cont_ids:
            cont_to_entity[cur] = seq_i
            cont_tag = page_conts[cur]
            cur = cont_tag.get("continuedat") or cont_tag.get("continuedAt") or ""
        continues_next = bool(cur and cur in all_cont_ids and cur not in page_cont_ids)
        entity_id_map[id(ent)] = (seq_i, continues_next)

    next_seq = len(entities)
    orphan_cont_ids: dict = {}
    for cont in page_soup.find_all("ix:continuation"):
        cid = cont.get("id", "")
        if cid not in cont_to_entity:
            orphan_cont_ids[cid] = next_seq
            next_seq += 1

    return entity_id_map, cont_to_entity, orphan_cont_ids


# ── DOM walker ──────────────────────────────────────────────────────────────

def _walk(
    node,
    current_line: list,
    lines: list,
    mode: str,
    entity_id_map: dict,
    cont_to_entity: dict,
    orphan_cont_ids: dict,
    context_map: dict,
    unit_map: dict,
) -> None:
    if isinstance(node, NavigableString):
        text = str(node)
        if text.strip():
            current_line.append(text)
        return

    if not isinstance(node, Tag):
        return

    tag_name = (node.name or "").lower()

    if tag_name == "ix:header":
        return

    if tag_name in ("ix:nonfraction", "ix:nonnumeric"):
        raw_text = node.get_text()
        if mode == "output":
            seq_i, continues_next = entity_id_map.get(id(node), (0, False))
            attrs = f'id="{seq_i}"'
            if tag_name == "ix:nonfraction":
                ctx = node.get("contextref") or node.get("contextRef") or ""
                unit_ref = node.get("unitref") or node.get("unitRef") or ""
                year = context_map.get(ctx, "")
                unit = unit_map.get(unit_ref, "")
                scale = node.get("scale") or "0"
                concept = node.get("name") or ""
                attrs += f' concept="{concept}" year="{year}" unit="{unit}" scale="{scale}"'
            else:
                ctx = node.get("contextref") or node.get("contextRef") or ""
                year = context_map.get(ctx, "")
                concept = node.get("name") or ""
                attrs += f' concept="{concept}" year="{year}"'
            if continues_next:
                attrs += ' continued="next"'
            current_line.append(f"<xbrl {attrs}>{raw_text}</xbrl>")
        else:
            current_line.append(raw_text)
        return

    if tag_name == "ix:continuation":
        cid = node.get("id") or ""
        raw_text = node.get_text()
        if mode == "output":
            if cid in cont_to_entity:
                seq_i = cont_to_entity[cid]
                current_line.append(f'<xbrl id="{seq_i}">{raw_text}</xbrl>')
            else:
                seq_i = orphan_cont_ids.get(cid, 0)
                current_line.append(
                    f'<xbrl id="{seq_i}" continued="prev">{raw_text}</xbrl>'
                )
        else:
            current_line.append(raw_text)
        return

    if tag_name.startswith("ix:"):
        for child in node.children:
            _walk(child, current_line, lines, mode,
                  entity_id_map, cont_to_entity, orphan_cont_ids,
                  context_map, unit_map)
        return

    # Regular HTML tag — flush current_line, recurse into children, flush again
    if current_line:
        line = "".join(current_line).strip()
        if line:
            lines.append(line)
        current_line.clear()

    for child in node.children:
        _walk(child, current_line, lines, mode,
              entity_id_map, cont_to_entity, orphan_cont_ids,
              context_map, unit_map)

    if current_line:
        line = "".join(current_line).strip()
        if line:
            lines.append(line)
        current_line.clear()


def merge_consecutive_xbrl(text: str) -> str:
    def _replace(m: re.Match) -> str:
        pre = re.sub(r'\s*continued="(?:next|prev)"', '', m.group(1))
        id_ = m.group(2)
        post = re.sub(r'\s*continued="(?:next|prev)"', '', m.group(3))
        content = m.group(4)
        return f'<xbrl{pre}id="{id_}"{post}>{content}\n'

    prev = None
    while prev != text:
        prev = text
        text = _MERGE_XBRL_RE.sub(_replace, text)
    return text


def extract_page_text(
    page_html: str,
    full_soup: BeautifulSoup,
    context_map: dict,
    unit_map: dict,
    mode: str,
) -> str:
    """mode='input' → plain text (ix: transparent); mode='output' → <xbrl>-wrapped."""
    page_soup = BeautifulSoup(page_html, "html.parser")

    all_cont_ids = {c.get("id", "") for c in full_soup.find_all("ix:continuation")}
    entity_id_map, cont_to_entity, orphan_cont_ids = build_continuation_maps(
        page_soup, all_cont_ids
    )

    lines: list = []
    current_line: list = []
    for child in page_soup.children:
        _walk(child, current_line, lines, mode,
              entity_id_map, cont_to_entity, orphan_cont_ids,
              context_map, unit_map)
    if current_line:
        line = "".join(current_line).strip()
        if line:
            lines.append(line)

    result = "\n".join(lines)
    if mode == "output":
        result = merge_consecutive_xbrl(result)
    return result


# ── Filing year detection ───────────────────────────────────────────────────

def detect_filing_year(context_map: dict) -> str:
    """Return the most common 4-digit year across all contexts (the filing year).

    Matches the upstream build logic (ESEF_train.json) where filing_year is the
    document year. Falls back to the highest year if counts tie.
    """
    counts = Counter(y for y in context_map.values() if y.isdigit() and len(y) == 4)
    if not counts:
        return ""
    # Tie-break on highest year so 2023 > 2022 when equal frequency.
    top = counts.most_common()
    top.sort(key=lambda x: (-x[1], -int(x[0])))
    return top[0][0]


# ── Output parsing (from evaluate_xt.parse_xbrl_tags) ───────────────────────

def parse_xbrl_tags(text: str) -> list[dict]:
    """Parse <xbrl> tagged output, grouping continuations by id."""
    soup = BeautifulSoup(text, "html.parser")
    by_id: dict = {}
    order: list = []
    for tag in soup.find_all("xbrl"):
        eid = str(tag.get("id", ""))
        concept = tag.get("concept", "")
        year = tag.get("year", "")
        unit = tag.get("unit", "")
        scale = tag.get("scale", "0")
        val = tag.get_text()
        if eid not in by_id:
            by_id[eid] = {"concept": concept, "year": year, "unit": unit,
                          "scale": scale, "value": ""}
            order.append(eid)
        else:
            if concept and not by_id[eid]["concept"]:
                by_id[eid]["concept"] = concept
            if year and not by_id[eid]["year"]:
                by_id[eid]["year"] = year
        by_id[eid]["value"] += val
    # Skip orphan continuations
    return [by_id[k] for k in order if by_id[k]["concept"]]


# ── GT loading: parse output text, skip non-ifrs, convert to spaced concepts ─

def filter_non_ifrs(output_txt: str) -> str:
    """Unwrap <xbrl> tags whose concept doesn't start with ifrs-full:."""
    def _replace(m):
        attrs = m.group(1)
        value = m.group(2)
        concept_m = _RE_CONCEPT_ATTR.search(attrs)
        if concept_m:
            concept = concept_m.group(1)
            if concept and not concept.startswith("ifrs-full:"):
                return value
        return m.group(0)
    return _RE_XBRL_FULL.sub(_replace, output_txt)


def gt_concepts_to_spaced(entities: list[dict]) -> list[dict]:
    out = []
    for e in entities:
        e2 = dict(e)
        if e2.get("concept"):
            e2["concept"] = concept_to_spaced(e2["concept"])
        out.append(e2)
    return out


def load_gt_from_output(output_txt: str) -> list[dict]:
    """Match evaluate_prod_xt.load_gt_spaced exactly: parse then spaced (no filter).

    Upstream intentionally does NOT filter non-ifrs tags out of GT — those
    entities become FNs since the model is trained only on ifrs-full concepts.
    Filtering them here would make demo recall look better than upstream eval.
    """
    entities = parse_xbrl_tags(output_txt)
    return gt_concepts_to_spaced(entities)


def classify_entities(entities: list[dict]) -> tuple[list[dict], list[dict]]:
    numeric = [e for e in entities if e.get("unit")]
    text = [e for e in entities if not e.get("unit")]
    return numeric, text


# ── Matching (verbatim from scripts/evaluate_xt.py) ─────────────────────────

def _norm(v: str) -> str:
    return re.sub(r"\s+", " ", str(v)).strip().lower()


def _norm_ws(v: str) -> str:
    return re.sub(r"[\s\u00a0\u2002\u2003\u2009]+", "", str(v)).lower()


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _word_contains(needle: str, haystack: str) -> bool:
    n_words = _norm(needle).split()
    h_words = _norm(haystack).split()
    if len(n_words) < 2:
        return False
    n = len(n_words)
    return any(h_words[i:i+n] == n_words for i in range(len(h_words) - n + 1))


def _values_match(gt_val: str, pred_val: str, soft: bool) -> bool:
    if _norm(gt_val) == _norm(pred_val):
        return True
    if not soft:
        return False
    gt_s, pred_s = _norm_ws(gt_val), _norm_ws(pred_val)
    if gt_s and pred_s and gt_s == pred_s:
        return True
    if gt_s and pred_s:
        shorter, longer = (gt_s, pred_s) if len(gt_s) <= len(pred_s) else (pred_s, gt_s)
        if len(shorter) >= 10 and longer.startswith(shorter):
            return True
    if _word_contains(gt_val, pred_val) or _word_contains(pred_val, gt_val):
        return True
    return _jaccard(_norm(gt_val), _norm(pred_val)) > 0.5


def match_entities(
    gt: list[dict],
    pred: list[dict],
    num_soft: bool = False,
    text_soft: bool = True,
) -> dict:
    used_pred = set()
    matched_pairs: list = []
    unmatched_gt: list = []

    for e_gt in gt:
        gt_val = e_gt.get("value", "")
        is_num = bool(e_gt.get("unit", ""))
        use_soft = num_soft if is_num else text_soft
        found = False
        for j, e_pred in enumerate(pred):
            if j in used_pred:
                continue
            if _values_match(gt_val, e_pred.get("value", ""), use_soft):
                used_pred.add(j)
                matched_pairs.append((e_gt, e_pred))
                found = True
                break
        if not found:
            unmatched_gt.append(e_gt)

    tp = len(matched_pairs)
    fn = len(unmatched_gt)
    fp = len(pred) - len(used_pred)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    concept_hits = sum(1 for g, p in matched_pairs if p.get("concept") == g.get("concept"))
    year_hits = sum(1 for g, p in matched_pairs if p.get("year") == g.get("year"))
    unit_hits = sum(1 for g, p in matched_pairs if p.get("unit") == g.get("unit"))
    scale_hits = sum(1 for g, p in matched_pairs if p.get("scale") == g.get("scale"))

    # Expose matched pairs + unmatched lists so the demo can colour-code.
    fps = [p for j, p in enumerate(pred) if j not in used_pred]

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": prec, "recall": rec, "f1": f1,
        "concept_hits": concept_hits,
        "year_hits": year_hits,
        "unit_hits": unit_hits,
        "scale_hits": scale_hits,
        "n_gt": len(gt), "n_pred": len(pred),
        "matched_pairs": matched_pairs,
        "unmatched_gt": unmatched_gt,
        "unmatched_pred": fps,
    }


def aggregate(metrics: list[dict]) -> dict:
    tp = sum(m["tp"] for m in metrics)
    fp = sum(m["fp"] for m in metrics)
    fn = sum(m["fn"] for m in metrics)
    ng = sum(m["n_gt"] for m in metrics)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    direct_tp = tp
    return {
        "precision": prec, "recall": rec, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
        "concept_acc": sum(m["concept_hits"] for m in metrics) / direct_tp if direct_tp > 0 else 0.0,
        "year_acc":    sum(m["year_hits"]    for m in metrics) / direct_tp if direct_tp > 0 else 0.0,
        "unit_acc":    sum(m["unit_hits"]    for m in metrics) / direct_tp if direct_tp > 0 else 0.0,
        "scale_acc":   sum(m["scale_hits"]   for m in metrics) / direct_tp if direct_tp > 0 else 0.0,
        "n_pages": len(metrics),
        "n_gt_total": ng,
    }


def typed_metrics_for_page(gt: list[dict], pred: list[dict]) -> dict:
    """Match once per type + once overall; return all three metric dicts."""
    gt_num, gt_text = classify_entities(gt)
    pred_num, pred_text = classify_entities(pred)
    overall = match_entities(gt, pred, num_soft=False, text_soft=True)
    numeric = match_entities(gt_num, pred_num, num_soft=False, text_soft=False)
    textual = match_entities(gt_text, pred_text, num_soft=False, text_soft=True)
    return {"overall": overall, "numeric": numeric, "text": textual}


def aggregate_typed(per_page_typed: list[dict]) -> dict:
    """Aggregate per-page typed metrics into overall/numeric/text holistic dicts."""
    def _agg_key(key: str) -> dict:
        return aggregate([pg[key] for pg in per_page_typed
                          if pg[key]["n_gt"] + pg[key]["n_pred"] > 0])
    return {
        "overall": _agg_key("overall"),
        "numeric": _agg_key("numeric"),
        "text":    _agg_key("text"),
    }


# ── Text preservation (from evaluate_xt.text_preservation) ──────────────────

def strip_xbrl_tags(text: str) -> str:
    return re.sub(r"</?xbrl[^>]*>", "", text)


def text_preservation(input_text: str, output_text: str) -> dict:
    stripped = strip_xbrl_tags(output_text)
    exact = (stripped == input_text)
    ratio = difflib.SequenceMatcher(None, input_text, stripped).ratio()
    return {"exact": exact, "similarity": ratio}
