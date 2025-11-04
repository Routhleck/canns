#!/usr/bin/env python3
"""
Translate Chinese documentation to English using Claude API.

Usage:
    export ANTHROPIC_API_KEY='your-key-here'
    python scripts/translate_docs.py --source docs/zh --target docs/en [--batch-size 5]

    # Or for OpenRouter
    export OPENROUTER_API_KEY='your-key-here'
    python scripts/translate_docs.py --source docs/zh --target docs/en --use-openrouter
"""

import argparse
import os
import re
from pathlib import Path
import requests
from anthropic import Anthropic

def should_translate_file(file_path):
    """Check if file should be translated."""
    # Only translate .rst files
    if not file_path.endswith('.rst'):
        return False
    # Skip auto-generated files
    if 'autoapi' in str(file_path):
        return False
    return True

def extract_metadata(content):
    """Extract RST metadata that shouldn't be translated."""
    # Find directives and code blocks that shouldn't be modified
    metadata = {
        'directives': re.findall(r'\.\. [a-z\-]+::', content),
        'code_blocks': re.findall(r'.. code-block:: \w+', content),
    }
    return metadata

def translate_file_with_claude(content, file_path, use_openrouter=False):
    """Translate RST file content using Claude API.

    API keys must be set via environment variables:
    - ANTHROPIC_API_KEY for Anthropic API
    - OPENROUTER_API_KEY for OpenRouter API
    """

    prompt = f"""Please translate the following ReStructuredText documentation from Chinese to English.

IMPORTANT RULES:
1. Preserve ALL RST formatting (titles, sections, code blocks, links, etc.)
2. Keep all directives exactly as they are (.. code-block::, .. literalinclude::, etc.)
3. Keep all cross-references and links unchanged
4. Keep filenames and paths unchanged
5. Keep mathematical formulas and LaTeX unchanged
6. Keep API references and class names unchanged
7. Translate ONLY the human-readable text content
8. Keep the same structure and hierarchy

File: {file_path}

Content to translate:
```
{content}
```

Provide ONLY the translated content without any explanation."""

    if use_openrouter:
        result = translate_with_openrouter(prompt)
    else:
        result = translate_with_anthropic(prompt)

    # Strip markdown code fences if present
    if result.startswith('```'):
        lines = result.split('\n')
        # Remove first line if it's just ```
        if lines[0].strip() == '```':
            lines = lines[1:]
        # Remove last line if it's just ```
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        result = '\n'.join(lines)

    return result

def translate_with_anthropic(prompt):
    """Translate using direct Anthropic API.

    API key must be set via ANTHROPIC_API_KEY environment variable.
    """
    client = Anthropic()  # Automatically uses ANTHROPIC_API_KEY env var

    message = client.messages.create(
        model="claude-opus",  # Using claude-opus as it's the latest available
        max_tokens=8000,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return message.content[0].text

def translate_with_openrouter(prompt):
    """Translate using OpenRouter API.

    API key must be set via OPENROUTER_API_KEY environment variable.
    """
    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not set. Please set it via:\n"
            "  export OPENROUTER_API_KEY='your-key-here'"
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/Routhleck/canns",
        "X-Title": "CANNS Documentation Translator",
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json={
            "model": "anthropic/claude-haiku-4.5",  # Most cost-effective option
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 8000,
        },
        timeout=30  # Prevent hanging on network issues
    )

    if response.status_code != 200:
        raise Exception(f"OpenRouter API error: {response.status_code} - {response.text}")

    return response.json()["choices"][0]["message"]["content"]

def translate_directory(source_dir, target_dir, batch_size=5, use_openrouter=False):
    """Translate all RST files in source directory to target directory.

    API keys must be set via environment variables:
    - ANTHROPIC_API_KEY for Anthropic API (default)
    - OPENROUTER_API_KEY for OpenRouter API (if --use-openrouter is specified)
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # Find all RST files
    rst_files = []
    for root, dirs, files in os.walk(source_path):
        # Skip certain directories
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git']]

        for file in files:
            file_path = os.path.join(root, file)
            if should_translate_file(file_path):
                rst_files.append(file_path)

    print(f"Found {len(rst_files)} RST files to translate")

    # Create target directory structure
    target_path.mkdir(parents=True, exist_ok=True)

    translated_count = 0
    failed_count = 0

    for i, source_file in enumerate(rst_files, 1):
        try:
            # Read source file
            with open(source_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Create target directory structure
            relative_path = os.path.relpath(source_file, source_path)
            target_file = os.path.join(target_path, relative_path)

            # Skip if target file already exists
            if os.path.exists(target_file):
                print(f"[{i}/{len(rst_files)}] Skipping (already translated): {target_file}")
                continue

            print(f"[{i}/{len(rst_files)}] Translating: {source_file}")

            # Translate content
            translated_content = translate_file_with_claude(
                content, source_file, use_openrouter=use_openrouter
            )

            # Create target directory structure
            target_file_dir = os.path.dirname(target_file)
            os.makedirs(target_file_dir, exist_ok=True)

            # Write translated content
            with open(target_file, 'w', encoding='utf-8') as f:
                f.write(translated_content)

            print(f"  ✓ Saved to: {target_file}")
            translated_count += 1

            # Print progress
            if i % batch_size == 0:
                print(f"\n--- Progress: {translated_count}/{len(rst_files)} translated ---\n")

        except Exception as e:
            print(f"  ✗ Error translating {source_file}: {e}")
            failed_count += 1

    print(f"\n{'='*50}")
    print(f"Translation complete!")
    print(f"  Successfully translated: {translated_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Total files: {len(rst_files)}")
    print(f"{'='*50}")

    return translated_count, failed_count

def main():
    parser = argparse.ArgumentParser(
        description="Translate Chinese documentation to English using Claude API"
    )
    parser.add_argument(
        '--source',
        default='docs/zh',
        help='Source directory containing Chinese docs (default: docs/zh)'
    )
    parser.add_argument(
        '--target',
        default='docs/en',
        help='Target directory for English docs (default: docs/en)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=5,
        help='Print progress every N files (default: 5)'
    )
    parser.add_argument(
        '--use-openrouter',
        action='store_true',
        help='Use OpenRouter API instead of direct Anthropic API'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be translated without translating'
    )

    args = parser.parse_args()

    if args.dry_run:
        source_path = Path(args.source)
        rst_files = []
        for root, dirs, files in os.walk(source_path):
            dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git']]
            for file in files:
                file_path = os.path.join(root, file)
                if should_translate_file(file_path):
                    rst_files.append(file_path)

        print(f"Would translate {len(rst_files)} files:")
        for f in rst_files:
            print(f"  - {f}")
        return

    # Check if source directory exists
    if not os.path.exists(args.source):
        print(f"Error: Source directory '{args.source}' not found")
        return

    print(f"Starting translation from {args.source} to {args.target}")
    if args.use_openrouter:
        print("Using OpenRouter API")
    else:
        print("Using Anthropic API")
    print("This may take a while depending on the number of files...")
    print()

    translate_directory(
        args.source, args.target, args.batch_size,
        use_openrouter=args.use_openrouter
    )

if __name__ == '__main__':
    main()
