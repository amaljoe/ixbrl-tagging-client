# iXBRL Tagging Demo

## Prerequisites

- Python 3.10+
- NVIDIA GPU with vLLM serving the phase1_prod checkpoint
- Chromium for Playwright (install once — see below)

## Setup

```bash
cd ixbrl-tagging-client
pip install -r requirements.txt
playwright install chromium
```

### Download model weights

```bash
bash download.sh
```

This will prompt for your HuggingFace token.

## Running

**Terminal 1 — start vLLM server:**

```basdh
bash scripts/vllm_ft_qwen.sh models/phase1_prod phase1_prod
```

**Terminal 2 — start the demo:**

```bash
cd ixbrl-tagging-client
streamlit run src/main.py
```

Open `http://localhost:8501` in your browser.

## Usage

### XHTML filing (full evaluation)

1. Upload an ESEF XHTML filing (`.xhtml` or `.html`)
2. Enter the page class name used in the filing (default: `page`; common alternatives: `pf`)
3. Click **Tag Document** — pages render and the model tags each one in batches
4. Click **Evaluate** to compare predictions against embedded ground truth
5. View holistic metrics (Precision, Recall, F1, Concept/Year/Unit/Scale accuracy) and per-page breakdowns

### PDF (tag-only mode)

1. Upload a PDF
2. Enter page numbers to tag (comma-separated, e.g. `1,3,5`)
3. Click **Tag Selected Pages**
4. View extracted entities per page (no evaluation available)

## Sidebar configuration

| Setting | Default | Description |
|---|---|---|
| vLLM Server URL | `http://localhost:8000` | Address of the model server |
| Model name | `phase1_prod` | Name registered with vLLM |
| Max tokens | 10000 | Maximum output tokens per page |
| Temperature | 0.0 | Sampling temperature (0 = deterministic) |
| Timeout (s) | 600 | Per-page HTTP timeout |
| Rep-cap | 50 | Max repetitions of the same value |
