#!/usr/bin/env python3
"""
Translate Tier 4 Jupyter notebooks from English to Chinese.

IMPORTANT: Preserves PlotConfig plot parameters (title, xlabel, ylabel, etc.) in English
to maintain cell output consistency.

Usage:
    export OPENROUTER_API_KEY='your-key-here'
    python translate_tier4_notebooks.py --source docs/en/3_full_detail_tutorials \\
                                         --target docs/zh/3_full_detail_tutorials
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List
import requests
import argparse


# Patterns to protect (keep in English)
PLOT_PARAM_PATTERNS = [
    r'(title\s*=\s*["\'])([^"\']+)(["\'])',
    r'(xlabel\s*=\s*["\'])([^"\']+)(["\'])',
    r'(ylabel\s*=\s*["\'])([^"\']+)(["\'])',
    r'(legend\s*=\s*["\'])([^"\']+)(["\'])',
    r'(label\s*=\s*["\'])([^"\']+)(["\'])',
    r'(zlabel\s*=\s*["\'])([^"\']+)(["\'])',
]


def protect_plot_params(code: str) -> tuple[str, Dict[str, str]]:
    """
    Replace plot parameters with placeholders before translation.
    Returns modified code and mapping of placeholders to original values.
    """
    placeholders = {}
    counter = 0

    protected_code = code
    for pattern in PLOT_PARAM_PATTERNS:
        matches = list(re.finditer(pattern, protected_code))
        for match in reversed(matches):  # Reverse to maintain indices
            param_name = match.group(1)
            param_value = match.group(2)
            closing = match.group(3)

            placeholder = f"__PLOT_PARAM_{counter}__"
            placeholders[placeholder] = match.group(0)

            # Replace with placeholder
            protected_code = (
                protected_code[:match.start()] +
                param_name + placeholder + closing +
                protected_code[match.end():]
            )
            counter += 1

    return protected_code, placeholders


def restore_plot_params(code: str, placeholders: Dict[str, str]) -> str:
    """Restore protected plot parameters after translation."""
    restored_code = code
    for placeholder, original in placeholders.items():
        # The placeholder might have been split, so we need to be careful
        # Extract just the value part for restoration
        pattern = re.search(r'(["\'])([^"\']+)(["\'])', original)
        if pattern:
            original_value = pattern.group(2)
            # Find the placeholder and restore the value
            restored_code = restored_code.replace(placeholder, original_value)

    return restored_code


def translate_text(content: str, context: str) -> str:
    """Translate text content using OpenRouter API."""

    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set. Please export your API key.")

    prompt = f"""Please translate the following content from English to Chinese.

IMPORTANT RULES:
1. Translate ALL text content to natural Chinese
2. For code blocks:
   - Translate comments (lines starting with #)
   - Translate strings in print() statements
   - Keep all code syntax unchanged (import statements, function names, variable names, class names)
3. Keep markdown formatting intact (headers, lists, code blocks, links, tables)
4. Keep mathematical formulas and LaTeX unchanged
5. Keep URLs and file paths unchanged
6. Use proper Chinese technical terminology
7. Keep any __PLOT_PARAM_X__ placeholders exactly as they are (do not translate)

Context: {context}

Content to translate:
```
{content}
```

Provide ONLY the translated content without any explanation or additional formatting."""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/Routhleck/canns",
        "X-Title": "CANNS Tier 4 Tutorial Translator",
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json={
            "model": "anthropic/claude-haiku-4.5",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 16000,
        },
        timeout=120
    )

    if response.status_code != 200:
        raise Exception(f"OpenRouter API error: {response.status_code} - {response.text}")

    result = response.json()["choices"][0]["message"]["content"]

    # Strip markdown code fences if present
    if result.startswith('```'):
        lines = result.split('\n')
        if lines[0].strip().startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        result = '\n'.join(lines)

    return result


def translate_notebook(notebook_path: Path, output_path: Path):
    """Translate a single Jupyter notebook."""

    print(f"\n  Processing: {notebook_path.name}")

    # Read notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    # Track translation stats
    stats = {'markdown_cells': 0, 'code_cells': 0, 'protected_params': 0}

    # Translate cells
    for i, cell in enumerate(notebook['cells']):
        cell_type = cell['cell_type']

        if cell_type == 'markdown':
            # Translate markdown cell
            source = ''.join(cell['source'])
            if source.strip():
                try:
                    translated = translate_text(source, f"Markdown cell {i+1}")
                    cell['source'] = translated.split('\n')
                    stats['markdown_cells'] += 1
                    print(f"    ✓ Translated markdown cell {i+1}")
                except Exception as e:
                    print(f"    ✗ Error translating markdown cell {i+1}: {e}")

        elif cell_type == 'code':
            # Translate code comments only
            source = ''.join(cell['source'])
            if '#' in source or 'print(' in source:
                try:
                    # Protect plot parameters
                    protected_source, placeholders = protect_plot_params(source)
                    if placeholders:
                        stats['protected_params'] += len(placeholders)

                    # Translate (comments will be translated, code unchanged)
                    translated = translate_text(protected_source, f"Code cell {i+1}")

                    # Restore plot parameters
                    if placeholders:
                        translated = restore_plot_params(translated, placeholders)

                    cell['source'] = translated.split('\n')
                    stats['code_cells'] += 1
                    print(f"    ✓ Translated code cell {i+1} ({len(placeholders)} params protected)")
                except Exception as e:
                    print(f"    ✗ Error translating code cell {i+1}: {e}")

        # Keep outputs unchanged (preserves rendered plots)

    # Write translated notebook
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

    print(f"  ✓ Saved: {output_path}")
    print(f"    Stats: {stats['markdown_cells']} MD cells, {stats['code_cells']} code cells, "
          f"{stats['protected_params']} plot params protected")


def main():
    parser = argparse.ArgumentParser(description='Translate Tier 4 notebooks to Chinese')
    parser.add_argument('--source', type=str, required=True,
                        help='Source directory (e.g., docs/en/3_full_detail_tutorials)')
    parser.add_argument('--target', type=str, required=True,
                        help='Target directory (e.g., docs/zh/3_full_detail_tutorials)')
    parser.add_argument('--dry-run', action='store_true',
                        help='List files without translating')

    args = parser.parse_args()

    source_dir = Path(args.source)
    target_dir = Path(args.target)

    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}")
        return

    # Find all notebooks
    notebooks = list(source_dir.rglob('*.ipynb'))

    print("=" * 70)
    print("Tier 4 Tutorial Notebook Translation")
    print("=" * 70)
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    print(f"Found {len(notebooks)} notebooks")

    if args.dry_run:
        print("\nDRY RUN - Files to translate:")
        for nb in notebooks:
            rel_path = nb.relative_to(source_dir)
            print(f"  - {rel_path}")
        return

    # Translate each notebook
    print("\nStarting translation...")
    success_count = 0

    for notebook_path in notebooks:
        try:
            # Compute output path
            rel_path = notebook_path.relative_to(source_dir)
            output_path = target_dir / rel_path

            # Translate
            translate_notebook(notebook_path, output_path)
            success_count += 1

        except Exception as e:
            print(f"  ✗ Failed to translate {notebook_path.name}: {e}")

    print("\n" + "=" * 70)
    print(f"Translation complete: {success_count}/{len(notebooks)} notebooks translated")
    print("=" * 70)


if __name__ == '__main__':
    main()
