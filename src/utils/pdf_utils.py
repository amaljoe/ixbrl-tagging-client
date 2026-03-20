"""pymupdf-based PDF page renderer."""

from PIL import Image
import fitz  # pymupdf


def render_pdf_pages(pdf_bytes: bytes, page_numbers: list[int], dpi: int = 150) -> list[Image.Image]:
    """Render selected 1-indexed page numbers from PDF to PIL Images.

    Args:
        pdf_bytes:    Raw PDF bytes.
        page_numbers: 1-indexed page numbers to render.
        dpi:          Resolution for rendering (default 150).

    Returns:
        List of PIL Images, one per requested page.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    n_pages = len(doc)
    images = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    for page_num in page_numbers:
        idx = page_num - 1
        if idx < 0 or idx >= n_pages:
            continue
        page = doc[idx]
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    doc.close()
    return images
