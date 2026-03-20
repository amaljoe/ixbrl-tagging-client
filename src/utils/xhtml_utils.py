"""Utilities for parsing ESEF iXBRL / XHTML filings.

Extracted from tagger/experiments/build_dataset.py so the same logic can be
reused by demos, evaluation scripts, and the dataset builder.
"""

from bs4 import BeautifulSoup


def parse_xhtml(html_path: str) -> tuple[dict, dict]:
    """Parse a full XHTML filing and build context / unit lookup maps.

    Args:
        html_path: Absolute path to the XHTML file.

    Returns:
        context_map: {contextref_id -> 4-digit year string}
        unit_map:    {unitref_id   -> currency code (e.g. 'GBP', 'EUR')}
    """
    with open(html_path, "r", errors="ignore") as f:
        content = f.read()
    soup = BeautifulSoup(content, "html.parser")

    context_map: dict[str, str] = {}
    for ctx in soup.find_all("xbrli:context"):
        cid = ctx.get("id")
        period = ctx.find("xbrli:period")
        if not period:
            continue
        instant = period.find("xbrli:instant")
        end = period.find("xbrli:enddate")
        date_node = instant or end
        if date_node:
            context_map[cid] = date_node.text.strip()[:4]

    unit_map: dict[str, str] = {}
    for unit in soup.find_all("xbrli:unit"):
        uid = unit.get("id")
        measure = unit.find("xbrli:measure")
        if measure:
            val = measure.text.strip()
            unit_map[uid] = val.split(":")[-1] if ":" in val else val

    return context_map, unit_map


def extract_page_tags(
    page_html: str,
    context_map: dict,
    unit_map: dict,
    prefix_filter: str = "ifrs-full:",
    include_nonnumeric: bool = False,
) -> list[dict]:
    """Extract iXBRL facts from a single page's HTML with resolved year and unit.

    Args:
        page_html:          Inner/outer HTML string of one rendered page element.
        context_map:        Output of parse_xhtml — maps contextref → year.
        unit_map:           Output of parse_xhtml — maps unitref → currency code.
        prefix_filter:      Only include concepts with this namespace prefix.
                            Pass '' or None to include all concepts.
        include_nonnumeric: Also extract ix:nonNumeric text entities (unit/scale = "").

    Returns:
        List of dicts with keys: value, concept, year, unit, scale.
        Concepts are in CamelCase (e.g. 'ifrs-full:RentalIncome').
        Text entities have unit="" and scale="".
    """
    soup = BeautifulSoup(page_html, "html.parser")
    facts = []
    for tag in soup.find_all("ix:nonfraction"):
        name = tag.get("name", "")
        if prefix_filter and not name.startswith(prefix_filter):
            continue
        ctx_id = tag.get("contextref", "")
        unit_id = tag.get("unitref", "")
        facts.append({
            "value":   tag.get_text(strip=True),
            "concept": name,                              # CamelCase, e.g. ifrs-full:RentalIncome
            "year":    context_map.get(ctx_id, ""),       # resolved from xbrli:context
            "unit":    unit_map.get(unit_id, ""),         # resolved from xbrli:unit
            "scale":   tag.get("scale", "0"),
        })
    if include_nonnumeric:
        for tag in soup.find_all("ix:nonnumeric"):
            name = tag.get("name", "")
            if prefix_filter and not name.startswith(prefix_filter):
                continue
            ctx_id = tag.get("contextref", "")
            facts.append({
                "value":   tag.get_text(strip=True),
                "concept": name,
                "year":    context_map.get(ctx_id, ""),
                "unit":    "",
                "scale":  "",
            })
    return facts
