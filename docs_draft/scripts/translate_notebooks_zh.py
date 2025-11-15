#!/usr/bin/env python3
"""
Translate English markdown (from notebooks) to Chinese.

Usage:
    export OPENROUTER_API_KEY='your-key-here'
    cd docs/zh
    uv run python ../../docs_draft/scripts/translate_notebooks_zh.py
"""

import os
from pathlib import Path
import requests

def translate_markdown_to_chinese(content, file_path):
    """Translate markdown content from English to Chinese using OpenRouter API."""

    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    prompt = f"""Please translate the following markdown documentation from English to Chinese.

IMPORTANT RULES:
1. Translate ALL text content to Chinese, including:
   - Headers and titles
   - Body text and explanations
   - Comments in code blocks (lines starting with #)
   - Strings in print() statements
   - Docstrings
2. Keep ALL code syntax unchanged (import statements, function names, variable names)
3. Keep all markdown formatting (titles, code blocks, links, etc.)
4. Keep mathematical formulas and LaTeX unchanged
5. Keep class names and API references unchanged
6. Translate naturally - use proper Chinese technical terminology
7. For code comments: translate to Chinese but keep the # symbol
8. For print statements: translate the content inside quotes

File: {file_path}

Content to translate:
```
{content}
```

Provide ONLY the translated content without any explanation."""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/Routhleck/canns",
        "X-Title": "CANNS Documentation Translator",
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
        if lines[0].strip() == '```' or lines[0].strip().startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        result = '\n'.join(lines)

    return result


def main():
    # Files to translate
    files_to_translate = [
        "0_why_canns.md",
        "1_quick_starts/01_build_model.md",
        "1_quick_starts/02_generate_tasks.md",
        "1_quick_starts/03_analyze_model.md",
        "1_quick_starts/04_analyze_data.md",
        "1_quick_starts/05_train_brain_inspired.md",
    ]

    docs_zh_dir = Path("/Users/sichaohe/Documents/GitHub/canns/docs/zh")

    print("=" * 60)
    print("Translating Markdown Files to Chinese")
    print("=" * 60)

    for file_path in files_to_translate:
        full_path = docs_zh_dir / file_path

        if not full_path.exists():
            print(f"  ⚠ Skipping {file_path} (not found)")
            continue

        print(f"\nTranslating: {file_path}")

        # Read content
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Translate
        try:
            translated = translate_markdown_to_chinese(content, file_path)

            # Write translated content
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(translated)

            print(f"  ✓ Translated successfully")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("\n" + "=" * 60)
    print("Translation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
