#!/usr/bin/env python3
"""
Translate Tier 4 RST files from English to Chinese.

Usage:
    export OPENROUTER_API_KEY='your-key-here'
    python translate_tier4_rst.py --source docs/en/3_full_detail_tutorials \\
                                   --target docs/zh/3_full_detail_tutorials
"""

import argparse
import os
from pathlib import Path
import requests


def translate_rst(content: str, file_path: str) -> str:
    """Translate RST content using OpenRouter API."""

    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set. Please export your API key.")

    prompt = f"""Please translate the following ReStructuredText documentation from English to Chinese.

IMPORTANT RULES:
1. Preserve ALL RST formatting (titles, sections, code blocks, links, etc.)
2. Keep all directives exactly as they are (.. toctree::, .. code-block::, etc.)
3. Keep all cross-references and links unchanged
4. Keep filenames and paths unchanged
5. Keep mathematical formulas unchanged
6. Keep API references and class names unchanged
7. Translate ONLY the human-readable text content to natural Chinese
8. Keep the same structure and hierarchy

File: {file_path}

Content to translate:
```
{content}
```

Provide ONLY the translated content without any explanation or additional formatting."""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/Routhleck/canns",
        "X-Title": "CANNS Tier 4 RST Translator",
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


def main():
    parser = argparse.ArgumentParser(description='Translate Tier 4 RST files to Chinese')
    parser.add_argument('--source', type=str, required=True,
                        help='Source directory')
    parser.add_argument('--target', type=str, required=True,
                        help='Target directory')

    args = parser.parse_args()

    source_dir = Path(args.source)
    target_dir = Path(args.target)

    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}")
        return

    # Find all RST files
    rst_files = list(source_dir.rglob('*.rst'))

    print("=" * 70)
    print("Tier 4 RST File Translation")
    print("=" * 70)
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    print(f"Found {len(rst_files)} RST files")

    success_count = 0

    for rst_path in rst_files:
        try:
            rel_path = rst_path.relative_to(source_dir)
            output_path = target_dir / rel_path

            print(f"\n  Translating: {rel_path}")

            # Read content
            with open(rst_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Translate
            translated = translate_rst(content, str(rel_path))

            # Write
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(translated)

            print(f"  ✓ Saved: {output_path}")
            success_count += 1

        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("\n" + "=" * 70)
    print(f"Translation complete: {success_count}/{len(rst_files)} files translated")
    print("=" * 70)


if __name__ == '__main__':
    main()
