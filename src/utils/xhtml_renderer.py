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
    numeric_only: bool = True,
) -> list[tuple[Image.Image, str]]:
    """Render pages from an XHTML filing using Playwright.

    Args:
        xhtml_path:   Absolute path to the XHTML file.
        class_name:   CSS class name used to identify page elements (e.g. 'page', 'pf').
        numeric_only: If True, only return pages containing ix:nonfraction tags.

    Returns:
        List of (PIL Image, inner_html) tuples for each matched page element.
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

        for el in elements:
            inner_html = el.inner_html()
            if numeric_only:
                soup = BeautifulSoup(inner_html, "html.parser")
                if not soup.find("ix:nonfraction"):
                    continue
            el.scroll_into_view_if_needed()
            png_bytes = el.screenshot()
            img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
            results.append((img, inner_html))

        browser.close()

    return results
