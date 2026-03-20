"""Shared evaluation metrics for iXBRL tag extraction.

Computes entity-level Precision/Recall/F1 on values and per-field accuracy
(concept, year, unit, scale) on matched entities.
"""

from collections import Counter


def _is_numeric(entity: dict) -> bool:
    return str(entity.get("unit", "")).strip() != ""


def _normalise(v) -> str:
    v = str(v)
    return v.replace(",", "").replace(" ", "").replace("\u00a0", "").strip()


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def _prf_and_attrs(gt_entities: list[dict], pred_entities: list[dict],
                   normalise_values: bool = False,
                   edit_distance: int = 0) -> dict:
    """Compute P/R/F1 + attribute accuracy for a set of entities.

    edit_distance=0  — exact match (default)
    edit_distance=N  — a prediction matches the nearest GT value within
                       Levenshtein distance N on the normalised value string.
    """
    fuzzy = edit_distance > 0

    def _key(v):
        return _normalise(str(v)) if normalise_values else str(v)

    if not fuzzy:
        gt_values = Counter(_key(e["value"]) for e in gt_entities)
        pred_values = Counter(_key(e.get("value", "")) for e in pred_entities)
        tp = sum(min(gt_values[k], pred_values[k]) for k in gt_values)
        fp = sum(max(pred_values[k] - gt_values.get(k, 0), 0) for k in pred_values)
        fn = sum(max(gt_values[k] - pred_values.get(k, 0), 0) for k in gt_values)
    else:
        # Fuzzy: consume GT pool greedily (exact first, then neighbour-aware fuzzy)
        gt_pool: dict[str, list] = {}
        for e in gt_entities:
            k = _key(e["value"])
            gt_pool.setdefault(k, []).append(e)

        tp = fp = 0
        for pe in pred_entities:
            pv = _key(pe.get("value", ""))
            matched = False
            # exact first
            if gt_pool.get(pv):
                gt_pool[pv].pop(0)
                if not gt_pool[pv]:
                    del gt_pool[pv]
                matched = True
            elif len(pv) >= 3:
                # Collect all GT candidates within edit_distance
                # Skip short values (len<3) to avoid false matches on '-','0','1' etc.
                candidates = []
                for gk, bucket in gt_pool.items():
                    if bucket and len(gk) >= 3:
                        ed = _levenshtein(pv, gk)
                        if ed <= edit_distance:
                            candidates.append((ed, gk, bucket[0]))
                if candidates:
                    # Neighbour-aware: prefer candidate with most matching attributes,
                    # breaking ties by edit distance (lowest first).
                    attr_fields = ["concept", "year", "unit", "scale"]
                    def _score(cand):
                        ed, gk, ge = cand
                        attr_matches = sum(
                            str(pe.get(f, "")).strip() == str(ge.get(f, "")).strip()
                            for f in attr_fields
                        )
                        return (-attr_matches, ed)  # more attr matches = better
                    candidates.sort(key=_score)
                    _, best_gk, _ = candidates[0]
                    gt_pool[best_gk].pop(0)
                    if not gt_pool[best_gk]:
                        del gt_pool[best_gk]
                    matched = True
            if matched:
                tp += 1
            else:
                fp += 1
        fn = sum(len(v) for v in gt_pool.values())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Attribute accuracy — always use exact value matching for the GT lookup
    # so we pair each pred with its closest GT for field comparison.
    attr_fields = ["concept", "year", "unit", "scale"]
    attr_correct = {f: 0 for f in attr_fields}
    attr_total = 0

    gt_by_value: dict[str, list] = {}
    for e in gt_entities:
        k = _key(e["value"])
        gt_by_value.setdefault(k, []).append(e)

    for pe in pred_entities:
        pv = _key(pe.get("value", ""))
        ge = None
        if gt_by_value.get(pv):
            ge = gt_by_value[pv].pop(0)
        elif fuzzy:
            for gk in list(gt_by_value.keys()):
                if gt_by_value[gk] and _levenshtein(pv, gk) <= edit_distance:
                    ge = gt_by_value[gk].pop(0)
                    break
        if ge is not None:
            attr_total += 1
            for f in attr_fields:
                if str(pe.get(f, "")).strip() == str(ge.get(f, "")).strip():
                    attr_correct[f] += 1

    attr_accuracy = {f: attr_correct[f] / attr_total if attr_total > 0 else 0.0
                     for f in attr_fields}

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "attr_accuracy": attr_accuracy,
        "attr_total": attr_total,
        "attr_correct": attr_correct,
    }


def compute_page_metrics(gt_entities: list[dict], pred_entities: list[dict],
                         normalise_numeric: bool = True,
                         edit_distance: int = 0) -> dict:
    """Compute entity-level P/R/F1 and attribute accuracy for one page.

    Matching is done by value string. For entities with duplicate values,
    matches are consumed in order (first-come first-served).
    Includes per-type breakdown: numeric (unit!="") and text (unit=="").
    When normalise_numeric=True (default), comma/space formatting differences
    are ignored for numeric entity matching.
    """
    overall = _prf_and_attrs(gt_entities, pred_entities,
                             normalise_values=normalise_numeric,
                             edit_distance=edit_distance)

    gt_num  = [e for e in gt_entities   if _is_numeric(e)]
    gt_text = [e for e in gt_entities   if not _is_numeric(e)]
    pred_num  = [e for e in pred_entities if _is_numeric(e)]
    pred_text = [e for e in pred_entities if not _is_numeric(e)]

    overall["by_type"] = {
        "numeric": _prf_and_attrs(gt_num,  pred_num,
                                  normalise_values=normalise_numeric,
                                  edit_distance=edit_distance),
        "text":    _prf_and_attrs(gt_text, pred_text),
    }
    return overall


def _aggregate_metrics(metrics_list: list[dict]) -> dict:
    """Micro-average a list of _prf_and_attrs dicts."""
    total_tp = sum(m["tp"] for m in metrics_list)
    total_fp = sum(m["fp"] for m in metrics_list)
    total_fn = sum(m["fn"] for m in metrics_list)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    attr_fields = ["concept", "year", "unit", "scale"]
    total_attr_total = sum(m["attr_total"] for m in metrics_list)
    overall_attr = {
        f: sum(m["attr_correct"][f] for m in metrics_list) / total_attr_total
        if total_attr_total > 0 else 0.0
        for f in attr_fields
    }
    return {
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "precision": precision, "recall": recall, "f1": f1,
        "attr_accuracy": overall_attr,
        "attr_total": total_attr_total,
    }


def compute_holistic_metrics(page_metrics_list: list[dict]) -> dict:
    """Micro-averaged metrics across all pages, with per-type breakdown."""
    result = _aggregate_metrics(page_metrics_list)
    result["by_type"] = {
        "numeric": _aggregate_metrics([m["by_type"]["numeric"] for m in page_metrics_list
                                       if "by_type" in m]),
        "text":    _aggregate_metrics([m["by_type"]["text"]    for m in page_metrics_list
                                       if "by_type" in m]),
    }
    return result
