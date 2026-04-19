"""Parity on numeric-heavy pages. Same as parity_check.py but filters to pages
whose output.txt contains at least one unit= attribute."""

import sys, json, random, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
UPSTREAM_ROOT = Path("/home/compiling-ganesh/24m0797/workspace/ixbrl-tagging")
sys.path.insert(0, str(UPSTREAM_ROOT))

from PIL import Image
import base64, io, requests
from utils.xt_extract import (
    PROMPT_PROD_XT_UNIFIED, SYSTEM_PROMPT,
    aggregate_typed as demo_aggregate_typed,
    parse_xbrl_tags as demo_parse,
    typed_metrics_for_page,
    load_gt_from_output,
)
from scripts.evaluate_xt import match_entities as up_match, aggregate as up_aggregate, parse_xbrl_tags as up_parse
from scripts.evaluate_prod_xt import load_gt_spaced as up_load_gt

DATA = UPSTREAM_ROOT / "data/eu_filings/processed-multi/mixed_val"

def b64(path):
    img = Image.open(path).convert("RGB")
    w,h=img.size
    M=1568
    if max(w,h)>M:
        s=M/max(w,h); img=img.resize((int(w*s),int(h*s)), Image.Resampling.LANCZOS)
    buf=io.BytesIO(); img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def call(model, port, im, txt, year):
    full = f"Filing year: {year}\n\n{txt}"
    r = requests.post(f"http://localhost:{port}/v1/chat/completions", timeout=300, json={
        "model": model,
        "messages": [
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":[
                {"type":"image_url","image_url":{"url":f"data:image/png;base64,{im}"}},
                {"type":"text","text":PROMPT_PROD_XT_UNIFIED+"\n\n"+full[:24000]},
            ]},
        ],
        "max_tokens":12000,"temperature":0,
    })
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def fmt(d, keys):
    return "  ".join(f"{k}={d.get(k,0):.3f}" if isinstance(d.get(k), float) else f"{k}={d.get(k,0)}" for k in keys)

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=8); ap.add_argument("--port", type=int, default=8000); ap.add_argument("--model", default="phase2")
args = ap.parse_args()
manifest = json.loads((DATA/"manifest.json").read_text())
random.seed(1)
random.shuffle(manifest["pages"])
chosen=[]
for pg in manifest["pages"]:
    op = DATA/f"{pg['stem']}-output.txt"
    ip = DATA/f"{pg['stem']}-input.txt"
    pn = DATA/f"{pg['stem']}.png"
    if op.exists() and ip.exists() and pn.exists():
        if 'unit="' in op.read_text(errors="ignore"):
            chosen.append(pg)
    if len(chosen)>=args.n: break
print(f"Numeric-parity on {len(chosen)} pages:")
demo_list=[]; up_list=[]
for pg in chosen:
    stem=pg["stem"]; year=pg["filing_year"]
    inp=(DATA/f"{stem}-input.txt").read_text(errors="ignore")
    im=b64(DATA/f"{stem}.png")
    resp=call(args.model, args.port, im, inp, year)
    gt_demo = load_gt_from_output((DATA/f"{stem}-output.txt").read_text(errors="ignore"))
    pred_demo = demo_parse(resp)
    td = typed_metrics_for_page(gt_demo, pred_demo)
    demo_list.append(td)
    gt_up = up_load_gt(DATA/f"{stem}-output.txt")
    pred_up = up_parse(resp)
    ov = up_match(gt_up, pred_up, num_soft=False, text_soft=True)
    gn=[e for e in gt_up if e.get("unit")]; gt_=[e for e in gt_up if not e.get("unit")]
    pn=[e for e in pred_up if e.get("unit")]; pt=[e for e in pred_up if not e.get("unit")]
    nm = up_match(gn, pn, num_soft=False, text_soft=False)
    tm = up_match(gt_, pt, num_soft=False, text_soft=True)
    up_list.append({"overall":ov,"numeric":nm,"text":tm})
    print(f"\n{stem} year={year}")
    for k in ("overall","numeric","text"):
        d=td[k]; u=up_list[-1][k]
        print(f"  {k:7s} DEMO    : {fmt(d,['tp','fp','fn','precision','recall','f1','concept_hits','year_hits','unit_hits','scale_hits'])}")
        print(f"  {k:7s} UPSTREAM: {fmt(u,['tp','fp','fn','precision','recall','f1','concept_hits','year_hits','unit_hits','scale_hits'])}")
print("\n"+"="*80+"\nHOLISTIC")
demo_hol = demo_aggregate_typed(demo_list)
up_hol = {
    "overall": up_aggregate([x["overall"] for x in up_list]),
    "numeric": up_aggregate([x["numeric"] for x in up_list if x["numeric"]["n_gt"]+x["numeric"]["n_pred"]>0]),
    "text": up_aggregate([x["text"] for x in up_list if x["text"]["n_gt"]+x["text"]["n_pred"]>0]),
}
keys=["tp","fp","fn","precision","recall","f1","concept_acc","year_acc","unit_acc","scale_acc"]
for k in ("overall","numeric","text"):
    print(f"{k:7s} DEMO    : {fmt(demo_hol[k], keys)}")
    print(f"{k:7s} UPSTREAM: {fmt(up_hol[k], keys)}")
