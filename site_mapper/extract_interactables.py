#!/usr/bin/env python3
"""
extract_interactables.py
------------------------
Extract all interactable elements from crawled HTML pages (buttons, inputs, links, event handlers, etc.)

Usage:
    python extract_interactables.py <site_results_path> [--output-dir <output_dir>]

Outputs per-page JSON files listing tag, attributes, and text of interactive elements, plus an aggregated JSON.
"""
import argparse
import json
from pathlib import Path
from bs4 import BeautifulSoup

# Tags that inherently represent interactive controls
INTERACTIVE_TAGS = [
    "button", "input", "select", "textarea", "a", "label", "details", "summary"
]
# HTML event handler attributes indicating interactivity
EVENT_ATTRIBUTES = [
    "onclick", "onchange", "onsubmit", "onmouseover", "onmouseout",
    "onmouseenter", "onmouseleave", "onfocus", "onblur", "oninput",
    "onkeydown", "onkeyup", "onkeypress"
]
# ARIA roles that imply interactivity
INTERACTIVE_ROLES = [
    "button", "link", "menuitem", "tab", "checkbox", "switch",
    "slider", "radio", "textbox", "combobox"
]

def extract_from_html(html_path: Path):
    """Parse HTML and extract interactive elements"""
    soup = BeautifulSoup(html_path.read_text(encoding='utf-8'), 'html.parser')
    elements = []
    
    for tag in soup.find_all(True):
        info = {
            "tag": tag.name,
            "attributes": {},
            "text": tag.get_text(strip=True)
        }
        interactive = False
        # Tag-based
        if tag.name in INTERACTIVE_TAGS:
            interactive = True
        # Event handlers
        for evt in EVENT_ATTRIBUTES:
            if tag.has_attr(evt):
                interactive = True
                info["attributes"][evt] = tag.get(evt)
        # href for links
        if tag.name == 'a' and tag.has_attr('href'):
            interactive = True
            info["attributes"]["href"] = tag.get('href')
        # tabindex
        if tag.has_attr('tabindex'):
            val = tag.get('tabindex')
            interactive = True
            info["attributes"]["tabindex"] = val
        # ARIA role
        if tag.has_attr('role') and tag['role'] in INTERACTIVE_ROLES:
            interactive = True
            info["attributes"]["role"] = tag['role']
        # id and class for context
        if tag.has_attr('id'):
            info["attributes"]["id"] = tag['id']
        if tag.has_attr('class'):
            info["attributes"]["class"] = tag['class']
        
        if interactive:
            elements.append(info)
    return elements


def main():
    parser = argparse.ArgumentParser(description='Extract interactive elements from crawled HTML')
    parser.add_argument('site_results_path', help='Path to site results directory')
    parser.add_argument('--output-dir', help='Directory to write JSON outputs')
    args = parser.parse_args()
    
    base_path = Path(args.site_results_path)
    if not base_path.exists() or not base_path.is_dir():
        print(f"Error: {base_path} does not exist or is not a directory")
        return
    
    output_dir = Path(args.output_dir) if args.output_dir else base_path / 'interactables'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    aggregated = {}
    
    # Iterate over page directories
    for page_dir in base_path.iterdir():
        if not page_dir.is_dir():
            continue
        # Find primary HTML file
        html_file = None
        for name in ('page.html', 'crawled.html', 'final_page.html'):
            candidate = page_dir / name
            if candidate.exists():
                html_file = candidate
                break
        if not html_file:
            continue
        # Extract
        elements = extract_from_html(html_file)
        aggregated[page_dir.name] = elements
        # Write per-page
        out_file = output_dir / f"{page_dir.name}_interactables.json"
        out_file.write_text(json.dumps(elements, indent=2), encoding='utf-8')
        print(f"Extracted {len(elements)} interactable elements from {page_dir.name}")
    
    # Write aggregated
    agg_file = output_dir / 'all_interactables.json'
    agg_file.write_text(json.dumps(aggregated, indent=2), encoding='utf-8')
    print(f"Aggregated interactables written to {agg_file}")

if __name__ == '__main__':
    main() 