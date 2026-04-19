"""iXBRL Tagging Demo — client-facing Streamlit app (phase2 / prod-unified-xt).

Step 1: Upload an XHTML filing or PDF and tag financial + text entities.
Step 2: Evaluate predictions against embedded iXBRL ground truth (XHTML only).

Matches scripts/evaluate_prod_xt.py from the training repo: identical parse,
matching logic, and metrics (numeric / text / overall).
"""

import asyncio
import os
import sys
import tempfile

_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import streamlit as st
from bs4 import BeautifulSoup

from utils.inference import infer_xt_batch_async
from utils.xhtml_renderer import render_xhtml_pages
from utils.pdf_utils import render_pdf_pages
from utils.xhtml_utils import parse_xhtml
from utils.xt_extract import (
    PROMPT_PROD_XT_UNIFIED,
    SYSTEM_PROMPT,
    aggregate,
    aggregate_typed,
    classify_entities,
    detect_filing_year,
    extract_page_text,
    load_gt_from_output,
    match_entities,
    parse_xbrl_tags,
    typed_metrics_for_page,
)


# ── Display helpers ──────────────────────────────────────────────────────────

_FIELDS = ["value", "concept", "year", "unit", "scale"]
_TD = "padding:3px 8px;font-size:12px;white-space:nowrap;vertical-align:top"
_TH = "padding:4px 8px;text-align:left;background:#444;color:#fff;font-size:12px"


def _esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _attr_cell(gt_val, pred_val) -> str:
    ok = str(gt_val).strip() == str(pred_val).strip()
    colour = "#6dba6d" if ok else "#d07070"
    icon = "✓" if ok else "✗"
    return (f'<td style="{_TD};color:{colour}">'
            f'<span style="font-weight:600">{icon}</span> {_esc(pred_val) or "—"}</td>')


def render_entity_table(
    matched_pairs: list,
    fns: list,
    fps: list,
    title: str = "",
) -> str:
    rows = []
    if matched_pairs:
        rows.append(
            f'<tr><td colspan="7" style="{_TD};background:#1a3a1a;color:#8fbc8f;'
            f'font-weight:700;padding:6px 8px">✓ MATCHED — {len(matched_pairs)}</td></tr>'
        )
        for gt, pred in matched_pairs:
            gt_cells = "".join(
                f'<td style="{_TD}">{_esc(gt.get(f,""))}</td>' for f in _FIELDS
            )
            rows.append(
                f'<tr style="background:#1e3a1e">'
                f'<td style="{_TD};color:#6dba6d;font-weight:700">GT</td>{gt_cells}</tr>'
            )
            pred_cells = "".join(_attr_cell(gt.get(f, ""), pred.get(f, "")) for f in _FIELDS)
            rows.append(
                f'<tr style="background:#162d16">'
                f'<td style="{_TD};color:#aaa;font-size:11px">pred</td>{pred_cells}</tr>'
            )
    if fns:
        rows.append(
            f'<tr><td colspan="7" style="{_TD};background:#3a1a1a;color:#e08080;'
            f'font-weight:700;padding:6px 8px">✗ MISSED — {len(fns)}</td></tr>'
        )
        for gt in fns:
            cells = "".join(f'<td style="{_TD}">{_esc(gt.get(f,""))}</td>' for f in _FIELDS)
            rows.append(
                f'<tr style="background:#3a1a1a">'
                f'<td style="{_TD};color:#e08080;font-weight:700">✗</td>{cells}</tr>'
            )
    if fps:
        rows.append(
            f'<tr><td colspan="7" style="{_TD};background:#3a2a00;color:#d4a020;'
            f'font-weight:700;padding:6px 8px">⚠ HALLUCINATED — {len(fps)}</td></tr>'
        )
        for pred in fps:
            cells = "".join(f'<td style="{_TD}">{_esc(pred.get(f,""))}</td>' for f in _FIELDS)
            rows.append(
                f'<tr style="background:#3a2a00">'
                f'<td style="{_TD};color:#d4a020;font-weight:700">⚠</td>{cells}</tr>'
            )

    header = "".join(f'<th style="{_TH}">{h}</th>' for h in ["", *_FIELDS])
    body = "".join(rows) or f'<tr><td colspan="7" style="{_TD}">(no entities)</td></tr>'
    title_row = (f'<tr><td colspan="7" style="{_TD};background:#2a2a40;color:#c0c0ff;'
                 f'font-weight:700;padding:6px 8px">{title}</td></tr>') if title else ""
    return (
        '<div style="overflow-x:auto">'
        '<table style="border-collapse:collapse;width:100%;font-family:monospace">'
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{title_row}{body}</tbody>"
        "</table></div>"
    )


def render_pred_only_table(preds: list) -> str:
    rows = []
    for pred in preds:
        cells = "".join(f'<td style="{_TD}">{_esc(pred.get(f,""))}</td>' for f in _FIELDS)
        rows.append(f'<tr style="background:#222">{cells}</tr>')
    header = "".join(f'<th style="{_TH}">{h}</th>' for h in _FIELDS)
    body = "".join(rows) or f'<tr><td colspan="5" style="{_TD}">(no entities)</td></tr>'
    return (
        '<div style="overflow-x:auto">'
        '<table style="border-collapse:collapse;width:100%;font-family:monospace">'
        f"<thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def metrics_block(label: str, agg: dict):
    """One row: F1 | P | R | Concept | Year | Unit | Scale + TP/FP/FN caption."""
    st.markdown(f"**{label}**")
    if not agg or agg.get("n_gt_total", 0) == 0 and agg.get("n_pages", 0) == 0:
        st.caption("(no entities)")
        return
    c = st.columns(7)
    c[0].metric("F1", f"{agg.get('f1', 0):.3f}")
    c[1].metric("Precision", f"{agg.get('precision', 0):.3f}")
    c[2].metric("Recall", f"{agg.get('recall', 0):.3f}")
    c[3].metric("Concept", f"{agg.get('concept_acc', 0):.2%}")
    c[4].metric("Year", f"{agg.get('year_acc', 0):.2%}")
    c[5].metric("Unit", f"{agg.get('unit_acc', 0):.2%}")
    c[6].metric("Scale", f"{agg.get('scale_acc', 0):.2%}")
    st.caption(
        f"TP={agg.get('tp',0)}  FP={agg.get('fp',0)}  FN={agg.get('fn',0)}  |  "
        f"GT={agg.get('n_gt_total',0)}  pages={agg.get('n_pages',0)}"
    )


def compact_metric_line(m: dict) -> str:
    """One-line per-type summary for the page expander label."""
    if not m or m.get("n_gt", 0) + m.get("n_pred", 0) == 0:
        return "—"
    tp = m.get("tp", 0)
    return (f"F1={m.get('f1',0):.2f} P={m.get('precision',0):.2f} R={m.get('recall',0):.2f} "
            f"[{tp}/{m.get('fp',0)}/{m.get('fn',0)}]")


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="iXBRL Tagger — phase2", layout="wide")
st.title("iXBRL Financial Report Tagger — phase2 (numeric + text)")
st.caption(
    "Unified Qwen3-VL-2B tagger (prod-unified-xt). Tags both numeric facts and "
    "text disclosures inline with <xbrl> tags."
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Server")
    api_base = st.text_input("vLLM server URL", value="http://localhost:8000")
    model_name = st.text_input("Model name", value="phase2")
    st.divider()
    st.header("Inference")
    max_tokens = st.slider("Max tokens", 4000, 16000, 12000, step=500)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, step=0.05)
    timeout_s = st.slider("Timeout (s)", 60, 900, 600, step=30)
    max_concurrent = st.slider("Concurrency", 1, 32, 8, step=1)
    max_input_chars = st.slider("Max input chars (per page)", 4000, 32000, 24000, step=1000)
    st.divider()
    st.caption(
        "Metrics match scripts/evaluate_prod_xt.py: value matching uses hard for "
        "numeric and soft (ws-strip + prefix + word-containment + Jaccard>0.5) for text."
    )

api_base_v1 = api_base.rstrip("/") + "/v1"

# ── Step 1: Upload & Tag ─────────────────────────────────────────────────────

st.header("Step 1 — Upload & Tag")

uploaded = st.file_uploader(
    "Upload financial report",
    type=["xhtml", "html", "pdf"],
    help="Upload an XBRL-tagged XHTML filing for full evaluation, or a PDF for tag-only mode.",
)

if uploaded is None:
    st.info("Upload a file above to get started.")
    st.stop()

file_ext = uploaded.name.rsplit(".", 1)[-1].lower()
is_xhtml = file_ext in ("xhtml", "html")
is_pdf = file_ext == "pdf"

if is_xhtml:
    st.info("XHTML detected — ground-truth evaluation is available after tagging.")
    col_a, col_b, col_c = st.columns([2, 1, 1])
    class_name = col_a.text_input("Page element class name", value="page",
                                  help="CSS class used to identify individual pages (e.g. 'page', 'pf')")
    max_pages = col_b.number_input("Max pages (0 = all)", min_value=0, max_value=5000,
                                   value=0, step=1)
    filing_year_input = col_c.text_input("Filing year", value="",
                                         placeholder="auto-detect",
                                         help="Override the auto-detected filing year.")
    tag_btn = st.button("Tag Document")
elif is_pdf:
    st.info("PDF detected — tags will be extracted but evaluation is not available.")
    col_a, col_b = st.columns([3, 1])
    pages_input = col_a.text_input("Pages to tag (comma-separated, ranges supported — e.g. 1,3,5-10,15)")
    filing_year_input = col_b.text_input("Filing year", value="", placeholder="e.g. 2024")
    tag_btn = st.button("Tag Selected Pages")
else:
    st.error("Unsupported file type."); st.stop()

if tag_btn:
    file_bytes = uploaded.read()

    with st.status("Rendering pages...") as render_status:
        if is_xhtml:
            with tempfile.NamedTemporaryFile(suffix=f".{file_ext}", delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            try:
                pages = render_xhtml_pages(tmp_path, class_name=class_name, numeric_only=False)
                context_map, unit_map = parse_xhtml(tmp_path)
            finally:
                os.unlink(tmp_path)

            full_soup = BeautifulSoup(file_bytes.decode(errors="ignore"), "html.parser")
            filing_year = filing_year_input.strip() or detect_filing_year(context_map)

            if max_pages and len(pages) > max_pages:
                pages = pages[:max_pages]
            render_status.update(label=f"Rendered {len(pages)} pages. Filing year: {filing_year}",
                                 state="complete")

            st.session_state["xhtml_context_map"] = context_map
            st.session_state["xhtml_unit_map"] = unit_map
            st.session_state["filing_year"] = filing_year

            # Build input + output text for each page (same logic as build_xt_dataset)
            with st.spinner("Extracting page text..."):
                input_texts = []
                output_texts = []
                for img, inner_html in pages:
                    inp = extract_page_text(inner_html, full_soup, context_map, unit_map, "input")
                    out = extract_page_text(inner_html, full_soup, context_map, unit_map, "output")
                    input_texts.append(inp)
                    output_texts.append(out)
            st.session_state["input_texts"] = input_texts
            st.session_state["output_texts"] = output_texts
        else:
            if not pages_input.strip():
                st.error("Please enter page numbers to tag."); st.stop()
            try:
                page_numbers = []
                for part in pages_input.split(","):
                    part = part.strip()
                    if not part:
                        continue
                    if "-" in part:
                        a, b = part.split("-", 1)
                        page_numbers.extend(range(int(a.strip()), int(b.strip()) + 1))
                    else:
                        page_numbers.append(int(part))
                page_numbers = sorted(set(page_numbers))
            except ValueError:
                st.error("Invalid page numbers — use integers, commas, and ranges (e.g. 1,3,5-10)."); st.stop()
            images = render_pdf_pages(file_bytes, page_numbers)
            pages = [(img, "") for img in images]
            input_texts = [""] * len(pages)  # no input text available for PDFs
            st.session_state["input_texts"] = input_texts
            st.session_state["output_texts"] = [""] * len(pages)
            st.session_state["filing_year"] = filing_year_input.strip()
            render_status.update(label=f"Rendered {len(pages)} PDF pages.", state="complete")

    if not pages:
        st.warning("No pages found to tag."); st.stop()

    n = len(pages)
    filing_year = st.session_state["filing_year"]

    # Build per-page user text: Filing year line + extracted text.
    per_page_user_text = [
        f"Filing year: {filing_year}\n\n{txt}" if filing_year else txt
        for txt in st.session_state["input_texts"]
    ]

    progress_bar = st.progress(0, text=f"Tagging pages... (0/{n})")

    all_raw: list[str] = [""] * n

    def _progress_cb(done, total):
        progress_bar.progress(min(done / max(total, 1), 1.0),
                              text=f"Tagging pages... ({done}/{total})")

    results = asyncio.run(infer_xt_batch_async(
        [img for img, _ in pages],
        per_page_user_text,
        prompt=PROMPT_PROD_XT_UNIFIED,
        system_prompt=SYSTEM_PROMPT,
        api_base=api_base_v1,
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout_s,
        max_concurrent=max_concurrent,
        max_input_chars=max_input_chars,
        infer_progress_fn=_progress_cb,
    ))
    all_raw = [r or "" for r in results]
    progress_bar.progress(1.0, text=f"Done — tagged {n} pages.")

    all_preds = [parse_xbrl_tags(r) for r in all_raw]

    st.session_state["tagged_pages"] = pages
    st.session_state["tagged_preds"] = all_preds
    st.session_state["tagged_raw"] = all_raw
    st.session_state["file_type"] = "xhtml" if is_xhtml else "pdf"
    st.session_state["eval_done"] = False

# ── Show tagging results ─────────────────────────────────────────────────────

if "tagged_pages" in st.session_state:
    pages = st.session_state["tagged_pages"]
    all_preds = st.session_state["tagged_preds"]
    all_raw = st.session_state["tagged_raw"]
    file_type = st.session_state["file_type"]

    st.divider()
    st.subheader(f"Tagged {len(pages)} pages")
    total_num = sum(len([e for e in p if e.get("unit")]) for p in all_preds)
    total_txt = sum(len([e for e in p if not e.get("unit")]) for p in all_preds)
    st.caption(f"Predictions: {total_num} numeric + {total_txt} text entities "
               f"(filing year: {st.session_state.get('filing_year') or '—'})")

    for i, ((img, _), preds) in enumerate(zip(pages, all_preds)):
        n_num = len([e for e in preds if e.get("unit")])
        n_txt = len([e for e in preds if not e.get("unit")])
        label = f"Page {i + 1} — {n_num} numeric + {n_txt} text tags"
        with st.expander(label, expanded=(i == 0 and file_type != "xhtml")):
            img_col, tbl_col = st.columns([1, 2])
            with img_col:
                st.image(img, use_container_width=True)
            with tbl_col:
                st.markdown(render_pred_only_table(preds), unsafe_allow_html=True)

    # ── Step 2: Evaluate ─────────────────────────────────────────────────────
    st.divider()
    st.header("Step 2 — Evaluate")

    eval_disabled = file_type != "xhtml"
    eval_btn = st.button("Evaluate",
                         disabled=eval_disabled,
                         help="XHTML ground truth required." if eval_disabled
                              else "Compare predictions against embedded iXBRL ground truth.")

    if eval_btn and not eval_disabled:
        output_texts = st.session_state["output_texts"]

        all_gt = [load_gt_from_output(out) for out in output_texts]
        per_page_typed = [typed_metrics_for_page(gt, pred)
                          for gt, pred in zip(all_gt, all_preds)]
        holistic = aggregate_typed(per_page_typed)

        st.session_state["all_gt"] = all_gt
        st.session_state["per_page_typed"] = per_page_typed
        st.session_state["holistic"] = holistic
        st.session_state["eval_done"] = True

    if st.session_state.get("eval_done") and file_type == "xhtml":
        holistic = st.session_state["holistic"]
        per_page_typed = st.session_state["per_page_typed"]
        all_gt = st.session_state["all_gt"]

        st.subheader("Holistic Metrics (entire filing)")
        metrics_block("Overall", holistic["overall"])
        metrics_block("Numeric", holistic["numeric"])
        metrics_block("Text", holistic["text"])

        st.divider()
        st.subheader("Per-Page Results")

        for i, ((img, _), preds, gt, typed) in enumerate(
            zip(pages, all_preds, all_gt, per_page_typed)
        ):
            ov = typed["overall"]
            label = (
                f"Page {i+1} — Overall {compact_metric_line(ov)}  |  "
                f"Num {compact_metric_line(typed['numeric'])}  |  "
                f"Txt {compact_metric_line(typed['text'])}"
            )
            with st.expander(label, expanded=(i == 0)):
                img_col, tbl_col = st.columns([1, 2])
                with img_col:
                    st.image(img, use_container_width=True)
                with tbl_col:
                    gt_num, gt_text = classify_entities(gt)
                    pred_num, pred_text = classify_entities(preds)

                    overall_m = typed["overall"]
                    numeric_m = typed["numeric"]
                    text_m = typed["text"]

                    a = st.tabs(["Overall", "Numeric", "Text"])
                    with a[0]:
                        st.markdown(
                            render_entity_table(
                                overall_m["matched_pairs"],
                                overall_m["unmatched_gt"],
                                overall_m["unmatched_pred"],
                            ),
                            unsafe_allow_html=True,
                        )
                    with a[1]:
                        st.markdown(
                            render_entity_table(
                                numeric_m["matched_pairs"],
                                numeric_m["unmatched_gt"],
                                numeric_m["unmatched_pred"],
                            ),
                            unsafe_allow_html=True,
                        )
                    with a[2]:
                        st.markdown(
                            render_entity_table(
                                text_m["matched_pairs"],
                                text_m["unmatched_gt"],
                                text_m["unmatched_pred"],
                            ),
                            unsafe_allow_html=True,
                        )
