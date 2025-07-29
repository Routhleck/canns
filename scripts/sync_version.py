#!/usr/bin/env python3
"""
Script to synchronize version across all project files.
This ensures consistency between git tags, package version, and documentation.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

def get_git_version():
    """Get the latest git tag version."""
    try:
        result = subprocess.run(
            ['git', 'describe', '--tags', '--abbrev=0'],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT
        )
        if result.returncode == 0:
            return result.stdout.strip().lstrip('v')
    except Exception as e:
        print(f"Error getting git version: {e}")
    return None

def update_version_in_file(file_path, version, pattern, replacement):
    """Update version in a specific file using regex pattern."""
    if not file_path.exists():
        print(f"Warning: {file_path} does not exist")
        return False
    
    try:
        content = file_path.read_text()
        new_content = re.sub(pattern, replacement, content)
        
        if content != new_content:
            file_path.write_text(new_content)
            print(f"Updated version in {file_path}")
            return True
        else:
            print(f"Version already up to date in {file_path}")
            return False
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

def main():
    """Main function to sync versions."""
    # Get target version
    if len(sys.argv) > 1:
        version = sys.argv[1].lstrip('v')
        print(f"Using provided version: {version}")
    else:
        version = get_git_version()
        if not version:
            print("Could not determine version from git tags")
            print("Usage: python sync_version.py [version]")
            sys.exit(1)
        print(f"Using git tag version: {version}")
    
    # Validate version format
    if not re.match(r'^\d+\.\d+\.\d+', version):
        print(f"Invalid version format: {version}")
        sys.exit(1)
    
    updated_files = []
    
    # Update _version.py fallback version
    version_file = PROJECT_ROOT / "src" / "canns" / "_version.py"
    if update_version_in_file(
        version_file,
        version,
        r'__version__ = "[^"]*\+dev"',
        f'__version__ = "{version}+dev"'
    ):
        updated_files.append(version_file)
    
    # Update docs/conf.py fallback version
    docs_conf = PROJECT_ROOT / "docs" / "conf.py"
    if update_version_in_file(
        docs_conf,
        version,
        r"version = '[^']*'",
        f"version = '{version}'"
    ):
        updated_files.append(docs_conf)
    
    if update_version_in_file(
        docs_conf,
        version,
        r"release = '[^']*'",
        f"release = '{version}'"
    ):
        updated_files.append(docs_conf)
    
    # Update README badges (if any version-specific badges exist)
    readme_files = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "README_zh.md"
    ]
    
    for readme_file in readme_files:
        if update_version_in_file(
            readme_file,
            version,
            r'pip install canns==\d+\.\d+\.\d+',
            f'pip install canns=={version}'
        ):
            updated_files.append(readme_file)
    
    # Update environment.yml if it has version pinning
    env_file = PROJECT_ROOT / "environment.yml"
    if update_version_in_file(
        env_file,
        version,  
        r'- canns==\d+\.\d+\.\d+',
        f'- canns=={version}'
    ):
        updated_files.append(env_file)
    
    # Summary
    if updated_files:
        print(f"\nSuccessfully updated version to {version} in:")
        for file_path in updated_files:
            print(f"  - {file_path.relative_to(PROJECT_ROOT)}")
        
        print(f"\nNext steps:")
        print(f"1. Review changes: git diff")
        print(f"2. Commit changes: git add -A && git commit -m 'Bump version to {version}'")
        print(f"3. Create tag: git tag v{version}")
        print(f"4. Push: git push origin --tags")
    else:
        print(f"\nAll files already have version {version}")

if __name__ == "__main__":
    main()