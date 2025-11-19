#!/usr/bin/env python3
"""
Convert CANNs markdown documentation to Jupyter notebooks.

This script parses markdown files and creates properly formatted Jupyter notebooks
with alternating markdown and code cells.

Supports tier-based folder structure:
- tier1_why_canns/
- tier2_quick_starts/
- tier3_core_concepts/
- tier4_full_details/
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any


def create_notebook_metadata() -> Dict[str, Any]:
    """Create standard notebook metadata."""
    return {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.11.0"
        }
    }


def parse_markdown_to_cells(md_content: str) -> List[Dict[str, Any]]:
    """
    Parse markdown content into notebook cells.

    Splits content into code blocks (```python...```) and markdown text.
    """
    cells = []
    cell_id_counter = 0

    # Split by code blocks while preserving them
    # Pattern: ```python\n...\n```
    pattern = r'```python\n(.*?)\n```'

    last_end = 0
    for match in re.finditer(pattern, md_content, re.DOTALL):
        # Add markdown cell for text before code block
        text_before = md_content[last_end:match.start()].strip()
        if text_before:
            # Split lines and add newline to each line except the last
            lines = text_before.split('\n')
            source = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []
            cells.append({
                "cell_type": "markdown",
                "id": f"cell-{cell_id_counter}",
                "metadata": {},
                "source": source
            })
            cell_id_counter += 1

        # Add code cell
        code = match.group(1)
        # Split lines and add newline to each line except the last
        code_lines = code.split('\n')
        code_source = [line + '\n' for line in code_lines[:-1]] + [code_lines[-1]] if code_lines else []
        cells.append({
            "cell_type": "code",
            "id": f"cell-{cell_id_counter}",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": code_source
        })
        cell_id_counter += 1

        last_end = match.end()

    # Add remaining markdown
    text_after = md_content[last_end:].strip()
    if text_after:
        # Split lines and add newline to each line except the last
        lines = text_after.split('\n')
        source = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []
        cells.append({
            "cell_type": "markdown",
            "id": f"cell-{cell_id_counter}",
            "metadata": {},
            "source": source
        })

    return cells


def convert_md_to_notebook(md_file: Path, output_file: Path) -> None:
    """Convert a markdown file to a Jupyter notebook."""
    print(f"Converting {md_file.name} → {output_file.name}")

    # Read markdown content
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Parse into cells
    cells = parse_markdown_to_cells(md_content)

    # Create notebook structure
    notebook = {
        "cells": cells,
        "metadata": create_notebook_metadata(),
        "nbformat": 4,
        "nbformat_minor": 5
    }

    # Write notebook
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print(f"  ✓ Created {output_file} with {len(cells)} cells")


def convert_tier_folder(tier_folder: Path) -> int:
    """
    Convert all markdown files in a tier folder to notebooks.

    Returns the number of files converted.
    """
    count = 0
    md_files = sorted(tier_folder.glob("*.md"))

    for md_file in md_files:
        # Output notebook has same name but .ipynb extension
        nb_file = md_file.with_suffix(".ipynb")

        try:
            convert_md_to_notebook(md_file, nb_file)
            count += 1
        except Exception as e:
            print(f"  ✗ Error converting {md_file.name}: {e}")

    return count


def main():
    """Convert all markdown docs to notebooks across all tiers."""
    # Script is in docs_draft/scripts/, drafts are in docs_draft/drafts/
    script_dir = Path(__file__).parent
    drafts_dir = script_dir.parent / "drafts"

    # Define tier folders
    tier_folders = [
        "tier1_why_canns",
        "tier2_quick_starts",
        "tier3_core_concepts",
        "tier4_full_details",
    ]

    print("=" * 60)
    print("Converting Markdown to Jupyter Notebooks")
    print(f"Base directory: {drafts_dir}")
    print("=" * 60)

    total_converted = 0

    for tier_name in tier_folders:
        tier_path = drafts_dir / tier_name

        if not tier_path.exists():
            print(f"\n⚠ Skipping {tier_name}/ (folder not found)")
            continue

        md_files = list(tier_path.glob("*.md"))
        if not md_files:
            print(f"\n⚠ Skipping {tier_name}/ (no .md files)")
            continue

        print(f"\n📁 {tier_name}/")
        print("-" * 40)

        count = convert_tier_folder(tier_path)
        total_converted += count

        print(f"  → Converted {count} file(s)")

    print("\n" + "=" * 60)
    print(f"Conversion complete! Total: {total_converted} file(s)")
    print("=" * 60)


if __name__ == "__main__":
    main()
