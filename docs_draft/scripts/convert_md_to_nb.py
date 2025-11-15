#!/usr/bin/env python3
"""
Convert CANNs markdown documentation to Jupyter notebooks.

This script parses markdown files and creates properly formatted Jupyter notebooks
with alternating markdown and code cells.
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


def main():
    """Convert all markdown docs to notebooks."""
    # Define conversions
    conversions = [
        ("01_why_canns.md", "00_why_canns.ipynb"),
        ("02_how_to_build_cann_model.md", "01_build_model.ipynb"),
        ("03_how_to_generate_task_data.md", "02_generate_tasks.ipynb"),
        ("04_how_to_analyze_cann_model.md", "03_analyze_model.ipynb"),
        ("05_how_to_analyze_experimental_data.md", "04_analyze_data.ipynb"),
        ("06_how_to_train_brain_inspired_model.md", "05_train_brain_inspired.ipynb"),
    ]

    # Script is now in docs_draft/scripts/, drafts are in docs_draft/drafts/
    script_dir = Path(__file__).parent
    drafts_dir = script_dir.parent / "drafts"
    output_dir = drafts_dir  # Save in same directory

    print("=" * 60)
    print("Converting Markdown to Jupyter Notebooks")
    print("=" * 60)

    for md_name, nb_name in conversions:
        md_file = drafts_dir / md_name
        nb_file = output_dir / nb_name

        if not md_file.exists():
            print(f"  ⚠ Skipping {md_name} (not found)")
            continue

        try:
            convert_md_to_notebook(md_file, nb_file)
        except Exception as e:
            print(f"  ✗ Error converting {md_name}: {e}")

    print("=" * 60)
    print("Conversion complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
