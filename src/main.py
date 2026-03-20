"""iXBRL Tagging Demo — client-facing Streamlit app.

Step 1: Upload an XHTML filing or PDF and tag financial facts.
Step 2: Evaluate predictions against ground truth (XHTML only).
"""

import asyncio
import sys
import os
import tempfile

_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import streamlit as st

from utils.inference import infer_batch_async, parse_json_response, cap_repetitions
from utils.concept_map import map_back_to_camel
from utils.metrics import compute_page_metrics, compute_holistic_metrics
from utils.xhtml_utils import parse_xhtml, extract_page_tags
from utils.xhtml_renderer import render_xhtml_pages
from utils.pdf_utils import render_pdf_pages


# ── Prompt ────────────────────────────────────────────────────────────────────

PROMPT_PROD30K = (
    "You are an IFRS financial data extraction system.\n\n"
    "Task: Identify every tagged financial fact on this page.\n\n"
    "For each fact, extract:\n"
    "- value: the numeric value as shown (e.g. '38,249')\n"
    "- concept: the IFRS tag in spaced format (e.g. 'ifrs-full: rental income')\n"
    "- year: the reporting period year (e.g. '2022')\n"
    "- unit: the currency code (e.g. 'GBP', 'EUR')\n"
    "- scale: the power-of-10 multiplier ('0'=units, '3'=thousands, '6'=millions)\n\n"
    "Return ONLY a JSON array. No other text."
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalise(v) -> str:
    return str(v).replace(",", "").replace(" ", "").replace("\u00a0", "").strip()


def _normalise_entity_values(entities: list[dict]) -> list[dict]:
    out = []
    for e in entities:
        ec = dict(e)
        ec["value"] = _normalise(ec.get("value", ""))
        out.append(ec)
    return out


def _post_process_preds(preds_raw: list[dict], rep_cap: int = 50) -> list[dict]:
    """Apply rep_cap and convert spaced concept → CamelCase."""
    preds = cap_repetitions(preds_raw, max_per_value=rep_cap) if rep_cap > 0 else preds_raw
    result = []
    for p in preds:
        spaced = str(p.get("concept", "")).strip()
        result.append({
            "value":   str(p.get("value",  "")).strip(),
            "concept": map_back_to_camel(spaced),
            "year":    str(p.get("year",   "")).strip(),
            "unit":    str(p.get("unit",   "")).strip(),
            "scale":   str(p.get("scale",  "0")).strip(),
        })
    return result


def match_entities_for_display(
    gt_entities: list[dict],
    pred_entities: list[dict],
) -> tuple[list[tuple], list[dict], list[dict]]:
    """Match GT vs preds by normalised value; return (matched_pairs, false_negatives, false_positives)."""
    gt_by_value: dict[str, list[dict]] = {}
    for e in gt_entities:
        key = _normalise(e["value"])
        gt_by_value.setdefault(key, []).append(e)

    matched_pairs: list[tuple] = []
    false_positives: list[dict] = []

    for pe in pred_entities:
        pv = _normalise(pe.get("value", ""))
        if pv in gt_by_value and gt_by_value[pv]:
            ge = gt_by_value[pv].pop(0)
            matched_pairs.append((ge, pe))
        else:
            false_positives.append(pe)

    false_negatives = [e for lst in gt_by_value.values() for e in lst]
    return matched_pairs, false_negatives, false_positives


def _attr_badge(gt_val: str, pred_val: str) -> str:
    ok = str(gt_val).strip() == str(pred_val).strip()
    colour = "#2d862d" if ok else "#cc2222"
    icon = "✓" if ok else "✗"
    return (
        f'<span style="color:{colour};font-weight:600">{icon} </span>'
        f'<span style="color:{colour}">{pred_val or "—"}</span>'
    )


def render_entity_table(
    matched_pairs: list[tuple],
    fns: list[dict],
    fps: list[dict],
) -> str:
    FIELDS = ["value", "concept", "year", "unit", "scale"]
    th_style = "padding:4px 8px;text-align:left;background:#444;color:#fff;font-size:12px"
    td_style = "padding:3px 8px;font-size:12px;white-space:nowrap"

    def _esc(s) -> str:
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    rows = []

    if matched_pairs:
        rows.append(
            f'<tr><td colspan="7" style="{td_style};background:#1a3a1a;color:#8fbc8f;'
            f'font-weight:700;padding:6px 8px">✓ MATCHED — {len(matched_pairs)} entities</td></tr>'
        )
        for gt, pred in matched_pairs:
            gt_cells = "".join(
                f'<td style="{td_style}">{_esc(gt.get(f,""))}</td>' for f in FIELDS
            )
            rows.append(
                f'<tr style="background:#1e3a1e">'
                f'<td style="{td_style};color:#6dba6d;font-weight:700">GT</td>'
                f'{gt_cells}</tr>'
            )
            pred_cells = "".join(
                f'<td style="{td_style}">{_attr_badge(gt.get(f,""), pred.get(f,""))}</td>'
                for f in FIELDS
            )
            rows.append(
                f'<tr style="background:#162d16">'
                f'<td style="{td_style};color:#aaa;font-size:11px">pred</td>'
                f'{pred_cells}</tr>'
            )

    if fns:
        rows.append(
            f'<tr><td colspan="7" style="{td_style};background:#3a1a1a;color:#e08080;'
            f'font-weight:700;padding:6px 8px">✗ MISSED — {len(fns)} entities</td></tr>'
        )
        for gt in fns:
            gt_cells = "".join(
                f'<td style="{td_style}">{_esc(gt.get(f,""))}</td>' for f in FIELDS
            )
            rows.append(
                f'<tr style="background:#3a1a1a">'
                f'<td style="{td_style};color:#e08080;font-weight:700">✗</td>'
                f'{gt_cells}</tr>'
            )

    if fps:
        rows.append(
            f'<tr><td colspan="7" style="{td_style};background:#3a2a00;color:#d4a020;'
            f'font-weight:700;padding:6px 8px">⚠ HALLUCINATED — {len(fps)} entities</td></tr>'
        )
        for pred in fps:
            pred_cells = "".join(
                f'<td style="{td_style}">{_esc(pred.get(f,""))}</td>' for f in FIELDS
            )
            rows.append(
                f'<tr style="background:#3a2a00">'
                f'<td style="{td_style};color:#d4a020;font-weight:700">⚠</td>'
                f'{pred_cells}</tr>'
            )

    header = "".join(f'<th style="{th_style}">{h}</th>' for h in ["", *FIELDS])
    body = "".join(rows) or f'<tr><td colspan="7" style="{td_style}">No entities</td></tr>'
    return (
        '<div style="overflow-x:auto">'
        f'<table style="border-collapse:collapse;width:100%;font-family:monospace">'
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table></div>"
    )


def render_pred_table(preds: list[dict]) -> str:
    """Simple entity table for PDF (no GT, no colour coding)."""
    FIELDS = ["value", "concept", "year", "unit", "scale"]
    th_style = "padding:4px 8px;text-align:left;background:#444;color:#fff;font-size:12px"
    td_style = "padding:3px 8px;font-size:12px;white-space:nowrap"

    def _esc(s) -> str:
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    rows = []
    for pred in preds:
        cells = "".join(f'<td style="{td_style}">{_esc(pred.get(f,""))}</td>' for f in FIELDS)
        rows.append(f'<tr style="background:#222">{cells}</tr>')

    header = "".join(f'<th style="{th_style}">{h}</th>' for h in FIELDS)
    body = "".join(rows) or f'<tr><td colspan="5" style="{td_style}">No entities</td></tr>'
    return (
        '<div style="overflow-x:auto">'
        f'<table style="border-collapse:collapse;width:100%;font-family:monospace">'
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table></div>"
    )


def show_holistic(h: dict, n_pages: int):
    a = h["attr_accuracy"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Precision", f"{h['precision']:.4f}")
    c2.metric("Recall",    f"{h['recall']:.4f}")
    c3.metric("F1",        f"{h['f1']:.4f}")
    st.caption(f"TP={h['tp']}  FP={h['fp']}  FN={h['fn']}  |  {n_pages} pages evaluated")
    a4 = st.columns(4)
    for col, field in zip(a4, ["concept", "year", "unit", "scale"]):
        col.metric(field.capitalize(), f"{a[field]:.2%}")


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="iXBRL Tagger", layout="wide")
st.title("iXBRL Financial Report Tagger")
st.caption("Upload a financial report to extract and evaluate IFRS-tagged financial facts.")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Server Configuration")
    api_base = st.text_input(
        "vLLM Server URL",
        value="http://localhost:8000",
        help="Address where the AI model server is running",
    )
    model_name = st.text_input(
        "Model name",
        value="phase1_prod",
        help="Name of the loaded model",
    )
    st.divider()
    st.header("Inference Settings")
    max_tokens = st.slider("Max tokens", 4000, 16000, 10000, step=500)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, step=0.05)
    timeout_s = st.slider(
        "Timeout (s)", 60, 900, 600, step=30,
        help="Time to wait per page. A10 GPU may be slower.",
    )
    rep_cap = st.slider(
        "Rep-cap (0=off)", 0, 200, 50, step=5,
        help="Limits repeated predictions for the same value",
    )

# Build full API base with /v1
api_base_v1 = api_base.rstrip("/") + "/v1"

# ── Step 1: Upload & Tag ──────────────────────────────────────────────────────

st.header("Step 1 — Upload & Tag")

uploaded = st.file_uploader(
    "Upload financial report",
    type=["xhtml", "html", "pdf"],
    help="Upload an XBRL-tagged XHTML filing for full evaluation, or a PDF for tag-only mode (no auto-evaluation).",
)

if uploaded is None:
    st.info("Upload a file above to get started.")
    st.stop()

file_ext = uploaded.name.rsplit(".", 1)[-1].lower()
is_xhtml = file_ext in ("xhtml", "html")
is_pdf = file_ext == "pdf"

if is_xhtml:
    st.info("XHTML detected — automatic evaluation is available after tagging.")
    class_name = st.text_input("Page element class name", value="page",
                               help="CSS class used to identify individual pages in the XHTML (e.g. 'page', 'pf')")
    tag_btn = st.button("Tag Document")
elif is_pdf:
    st.info("PDF detected — tags will be extracted but evaluation is not available (no ground truth).")
    pages_input = st.text_input(
        "Pages to tag (comma-separated, e.g. 1,3,5)",
        help="Enter the page numbers you want to tag",
    )
    tag_btn = st.button("Tag Selected Pages")
else:
    st.error("Unsupported file type.")
    st.stop()

if tag_btn:
    file_bytes = uploaded.read()

    # ── Render pages ──────────────────────────────────────────────────────────
    with st.status("Rendering pages...") as render_status:
        if is_xhtml:
            with tempfile.NamedTemporaryFile(suffix=f".{file_ext}", delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            try:
                pages = render_xhtml_pages(tmp_path, class_name=class_name, numeric_only=True)
            finally:
                os.unlink(tmp_path)
            render_status.update(label=f"Rendered {len(pages)} pages.", state="complete")

            # Parse GT context/unit maps from XHTML
            with tempfile.NamedTemporaryFile(suffix=f".{file_ext}", delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            try:
                context_map, unit_map = parse_xhtml(tmp_path)
            finally:
                os.unlink(tmp_path)

            st.session_state["xhtml_context_map"] = context_map
            st.session_state["xhtml_unit_map"] = unit_map

        else:  # PDF
            if not pages_input.strip():
                st.error("Please enter page numbers to tag.")
                st.stop()
            try:
                page_numbers = [int(x.strip()) for x in pages_input.split(",") if x.strip()]
            except ValueError:
                st.error("Invalid page numbers — enter comma-separated integers.")
                st.stop()
            images = render_pdf_pages(file_bytes, page_numbers)
            pages = [(img, "") for img in images]
            render_status.update(label=f"Rendered {len(pages)} PDF pages.", state="complete")
            st.session_state["pdf_page_numbers"] = page_numbers

    if not pages:
        st.warning("No pages found to tag. Check the class name or page numbers.")
        st.stop()

    # ── Batch inference with live progress ────────────────────────────────────
    n = len(pages)
    progress_bar = st.progress(0, text=f"Tagging pages... (0/{n})")
    BATCH = max(1, min(5, n))
    all_raw: list[str] = []

    for i in range(0, n, BATCH):
        batch_images = [img for img, _ in pages[i:i + BATCH]]
        batch_results = asyncio.run(infer_batch_async(
            batch_images,
            prompt=PROMPT_PROD30K,
            api_base=api_base_v1,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout_s,
        ))
        all_raw.extend(batch_results)
        done = len(all_raw)
        progress_bar.progress(min(done / n, 1.0), text=f"Tagging pages... ({done}/{n})")

    progress_bar.progress(1.0, text=f"Done — tagged {n} pages.")

    # Post-process predictions
    all_preds = []
    for raw in all_raw:
        preds_raw = parse_json_response(raw) if raw else []
        preds = _post_process_preds(preds_raw, rep_cap=rep_cap)
        all_preds.append(preds)

    st.session_state["tagged_pages"] = pages
    st.session_state["tagged_preds"] = all_preds
    st.session_state["tagged_raw"] = all_raw
    st.session_state["file_type"] = "xhtml" if is_xhtml else "pdf"
    st.session_state["eval_done"] = False

# ── Show tagging results ──────────────────────────────────────────────────────

if "tagged_pages" in st.session_state:
    pages = st.session_state["tagged_pages"]
    all_preds = st.session_state["tagged_preds"]
    all_raw = st.session_state["tagged_raw"]
    file_type = st.session_state["file_type"]

    st.divider()
    st.subheader(f"Tagged {len(pages)} pages")

    for i, ((img, inner_html), preds) in enumerate(zip(pages, all_preds)):
        page_label = f"Page {i + 1} — {len(preds)} tags extracted"
        with st.expander(page_label, expanded=(i == 0)):
            img_col, tbl_col = st.columns([1, 2])
            with img_col:
                st.image(img, use_container_width=True)
            with tbl_col:
                st.markdown(render_pred_table(preds), unsafe_allow_html=True)

    # ── Step 2: Evaluate ──────────────────────────────────────────────────────

    st.divider()
    st.header("Step 2 — Evaluate")

    eval_disabled = file_type != "xhtml"
    eval_tooltip = ("Evaluate requires XHTML ground truth. Not available for PDF uploads."
                    if eval_disabled else "Compare predictions against XHTML ground truth tags.")

    eval_btn = st.button("Evaluate", disabled=eval_disabled, help=eval_tooltip)

    if eval_btn and not eval_disabled:
        context_map = st.session_state["xhtml_context_map"]
        unit_map = st.session_state["xhtml_unit_map"]

        # Extract GT per page from inner_html
        all_gt = []
        for img, inner_html in pages:
            gt = extract_page_tags(inner_html, context_map, unit_map)
            all_gt.append(gt)

        # Compute per-page metrics
        page_metrics = []
        for gt, preds in zip(all_gt, all_preds):
            gt_norm = _normalise_entity_values(gt)
            preds_norm = _normalise_entity_values(preds)
            m = compute_page_metrics(gt_norm, preds_norm)
            page_metrics.append(m)

        holistic = compute_holistic_metrics(page_metrics)

        st.session_state["eval_done"] = True
        st.session_state["all_gt"] = all_gt
        st.session_state["page_metrics"] = page_metrics
        st.session_state["holistic"] = holistic

    if st.session_state.get("eval_done") and file_type == "xhtml":
        holistic = st.session_state["holistic"]
        page_metrics = st.session_state["page_metrics"]
        all_gt = st.session_state["all_gt"]

        st.subheader("Holistic Metrics")
        show_holistic(holistic, n_pages=len(pages))

        st.divider()
        st.subheader("Per-Page Results")

        for i, ((img, inner_html), preds, gt, m) in enumerate(
            zip(pages, all_preds, all_gt, page_metrics)
        ):
            a = m["attr_accuracy"]
            page_label = (
                f"Page {i + 1} — "
                f"F1={m['f1']:.2f} | Prec={m['precision']:.2f} | Rec={m['recall']:.2f} | "
                f"Concept={a['concept']:.0%} | Year={a['year']:.0%} | "
                f"Unit={a['unit']:.0%} | Scale={a['scale']:.0%}"
            )
            with st.expander(page_label, expanded=(i == 0)):
                img_col, tbl_col = st.columns([1, 2])
                with img_col:
                    st.image(img, use_container_width=True)
                with tbl_col:
                    gt_norm = _normalise_entity_values(gt)
                    preds_norm = _normalise_entity_values(preds)
                    matched, fns, fps = match_entities_for_display(gt_norm, preds_norm)
                    st.markdown(render_entity_table(matched, fns, fps), unsafe_allow_html=True)
