#!/usr/bin/env python3
"""
Convert markdown tutorials to Jupyter notebooks.
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Any


def parse_markdown_to_cells(md_content: str) -> List[Dict[str, Any]]:
    """Parse markdown content into notebook cells."""
    cells = []

    # Add warning cell at the beginning
    warning_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "> **⚠️ Important**: Run cells in order, as each cell depends on the previous state.\n"
        ]
    }
    cells.append(warning_cell)

    # Split content by code blocks
    # Pattern to match ```python ... ``` blocks
    code_pattern = r'```python\n(.*?)\n```'

    parts = []
    last_end = 0

    for match in re.finditer(code_pattern, md_content, re.DOTALL):
        # Add text before code block
        if match.start() > last_end:
            text = md_content[last_end:match.start()].strip()
            if text:
                parts.append(('markdown', text))

        # Add code block
        code = match.group(1)
        parts.append(('code', code))
        last_end = match.end()

    # Add remaining text
    if last_end < len(md_content):
        text = md_content[last_end:].strip()
        if text:
            parts.append(('markdown', text))

    # Convert parts to cells
    for cell_type, content in parts:
        if cell_type == 'markdown':
            # Update cross-references from .md to .ipynb
            content = re.sub(r'\.md(\)|#)', r'.ipynb\1', content)

            # Split long markdown sections at H2 headers
            sections = re.split(r'\n(## )', content)

            current_text = sections[0].strip() if sections else ""

            for i in range(1, len(sections), 2):
                if current_text:
                    cells.append({
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": current_text.split('\n')
                    })
                if i + 1 < len(sections):
                    current_text = (sections[i] + sections[i+1]).strip()

            if current_text:
                cells.append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": current_text.split('\n')
                })

        elif cell_type == 'code':
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": content.split('\n')
            })

    return cells


def create_notebook(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a Jupyter notebook structure."""
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
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
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    return notebook


def convert_md_to_notebook(md_file: Path, output_file: Path):
    """Convert a markdown file to a Jupyter notebook."""
    print(f"Converting {md_file.name}...")

    # Read markdown
    md_content = md_file.read_text(encoding='utf-8')

    # Parse to cells
    cells = parse_markdown_to_cells(md_content)

    # Create notebook
    notebook = create_notebook(cells)

    # Write notebook
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

    print(f"  → Created {output_file}")


def main():
    """Convert all tutorial markdown files to notebooks."""
    base_dir = Path(__file__).parent
    drafts_dir = base_dir / "drafts" / "tier4_full_details"
    output_base = base_dir.parent / "docs" / "en" / "3_full_detail_tutorials"

    # Define conversions: (source_md, target_nb)
    conversions = [
        # Scenario 1: CANN Modeling
        ("scenario_1_cann_modeling/01_build_cann_model.md",
         "01_cann_modeling/01_build_cann_model.ipynb"),
        ("scenario_1_cann_modeling/02_task_and_simulation.md",
         "01_cann_modeling/02_task_and_simulation.ipynb"),
        ("scenario_1_cann_modeling/03_analysis_visualization.md",
         "01_cann_modeling/03_analysis_visualization.ipynb"),
        ("scenario_1_cann_modeling/04_parameter_effects.md",
         "01_cann_modeling/04_parameter_effects.ipynb"),
        ("scenario_1_cann_modeling/05_hierarchical_network.md",
         "01_cann_modeling/05_hierarchical_network.ipynb"),
        ("scenario_1_cann_modeling/06_theta_sweep_hd_grid.md",
         "01_cann_modeling/06_theta_sweep_hd_grid.ipynb"),
        ("scenario_1_cann_modeling/07_theta_sweep_place_cell.md",
         "01_cann_modeling/07_theta_sweep_place_cell.ipynb"),

        # Scenario 3: Brain-Inspired Learning
        ("scenario_3_brain_inspired_learning/01_pattern_storage_recall.md",
         "03_brain_inspired/01_pattern_storage_recall.ipynb"),
        ("scenario_3_brain_inspired_learning/02_feature_learning_temporal.md",
         "03_brain_inspired/02_feature_learning_temporal.ipynb"),

        # Scenario 4: End-to-End Pipeline
        ("scenario_4_end_to_end_pipeline/01_theta_sweep_pipeline.md",
         "04_pipeline/01_theta_sweep_pipeline.ipynb"),
    ]

    print(f"Converting {len(conversions)} markdown files to Jupyter notebooks...")
    print()

    for source_rel, target_rel in conversions:
        source_file = drafts_dir / source_rel
        target_file = output_base / target_rel

        if source_file.exists():
            convert_md_to_notebook(source_file, target_file)
        else:
            print(f"WARNING: Source file not found: {source_file}")

    print()
    print(f"✓ Conversion complete! Created {len(conversions)} notebooks.")
    print(f"  Output directory: {output_base}")


if __name__ == "__main__":
    main()
