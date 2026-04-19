"""Playwright-based XHTML page renderer.

Renders individual pages from an XHTML filing to PIL Images.
"""

import io
from bs4 import BeautifulSoup
from PIL import Image
from playwright.sync_api import sync_playwright


def render_xhtml_pages(
    xhtml_path: str,
    class_name: str = "page",
    tagged_only: bool = True,
    only_indices: set[int] | None = None,
    progress_fn=None,
) -> list[tuple[Image.Image, str]]:
    """Render pages from an XHTML filing using Playwright.

    Args:
        tagged_only: Skip pages with no ix:nonfraction or ix:nonnumeric tags.
        only_indices: If given, only render 1-based page positions in this set.
        progress_fn: Optional callback(scanned, total, rendered) called after each page.

    Returns:
        List of (PIL Image, inner_html) tuples.
    """
    results = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 3840, "height": 2160},
            device_scale_factor=2,
        )
        page = context.new_page()
        page.goto(f"file://{xhtml_path}")
        page.wait_for_load_state("domcontentloaded")

        elements = page.query_selector_all(f".{class_name}")
        candidates = [
            (i, el) for i, el in enumerate(elements, start=1)
            if only_indices is None or i in only_indices
        ]
        total = len(candidates)

        for scanned, (i, el) in enumerate(candidates, start=1):
            inner_html = el.inner_html()
            if tagged_only:
                soup = BeautifulSoup(inner_html, "html.parser")
                if not soup.find("ix:nonfraction") and not soup.find("ix:nonnumeric"):
                    if progress_fn:
                        progress_fn(scanned, total, len(results))
                    continue
            el.scroll_into_view_if_needed()
            png_bytes = el.screenshot()
            img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
            results.append((img, inner_html))
            if progress_fn:
                progress_fn(scanned, total, len(results))

        browser.close()

    return results
