#!/usr/bin/env python3
"""
Script to test documentation build locally before pushing to ReadTheDocs.
This helps catch issues early in the development process.
"""

import os
import subprocess
import sys
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"

def run_command(cmd, cwd=None, capture_output=False):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, 
            cwd=cwd, 
            capture_output=capture_output, 
            text=True,
            check=True
        )
        if capture_output:
            return result.stdout.strip()
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if capture_output and e.stdout:
            print(f"stdout: {e.stdout}")
        if capture_output and e.stderr:
            print(f"stderr: {e.stderr}")
        return False

def test_python_imports():
    """Test that we can import the package and its dependencies."""
    print("\n=== Testing Python Imports ===")
    
    # Test core package import
    try:
        import canns
        print(f"✅ Successfully imported canns version: {canns.__version__}")
    except ImportError as e:
        print(f"❌ Failed to import canns: {e}")
        print("   Make sure to install the package: pip install -e .")
        return False
    
    # Test documentation dependencies
    required_packages = [
        'sphinx',
        'nbsphinx', 
        'sphinx_rtd_theme',
        'myst_parser'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} available")
        except ImportError:
            print(f"❌ {package} not available")
            print(f"   Install with: pip install {package}")
            return False
    
    return True

def test_sphinx_config():
    """Test that Sphinx configuration is valid."""
    print("\n=== Testing Sphinx Configuration ===")
    
    config_file = DOCS_DIR / "conf.py"
    if not config_file.exists():
        print(f"❌ Sphinx config file not found: {config_file}")
        return False
    
    # Test config syntax
    try:
        spec = {}
        with open(config_file) as f:
            exec(f.read(), spec)
        print("✅ Sphinx configuration syntax is valid")
        
        # Check required settings
        required_settings = ['project', 'author', 'extensions']
        for setting in required_settings:
            if setting in spec:
                print(f"✅ Found required setting: {setting}")
            else:
                print(f"⚠️  Missing setting: {setting}")
                
    except Exception as e:
        print(f"❌ Error in Sphinx configuration: {e}")
        return False
    
    return True

def test_notebooks():
    """Test that notebooks are valid and can be processed."""
    print("\n=== Testing Notebooks ===")
    
    notebook_dir = DOCS_DIR / "en" / "notebooks"
    if not notebook_dir.exists():
        print(f"❌ Notebook directory not found: {notebook_dir}")
        return False
    
    notebooks = list(notebook_dir.glob("*.ipynb"))
    if not notebooks:
        print("⚠️  No notebooks found")
        return True
    
    print(f"Found {len(notebooks)} notebooks")
    
    for notebook in notebooks:
        try:
            # Basic JSON validation
            import json
            with open(notebook) as f:
                nb_data = json.load(f)
            
            # Check basic notebook structure
            if 'cells' in nb_data and 'metadata' in nb_data:
                print(f"✅ {notebook.name} has valid structure")
            else:
                print(f"❌ {notebook.name} has invalid structure")
                return False
                
        except json.JSONDecodeError as e:
            print(f"❌ {notebook.name} has invalid JSON: {e}")
            return False
        except Exception as e:
            print(f"❌ Error checking {notebook.name}: {e}")
            return False
    
    return True

def test_docs_build():
    """Test building the documentation."""
    print("\n=== Testing Documentation Build ===")
    
    # Clean previous build
    build_dir = DOCS_DIR / "_build"
    if build_dir.exists():
        import shutil
        shutil.rmtree(build_dir)
        print("🧹 Cleaned previous build")
    
    # Try to build docs
    cmd = ["sphinx-build", "-b", "html", ".", "_build/html", "-W"]
    success = run_command(cmd, cwd=DOCS_DIR)
    
    if success:
        print("✅ Documentation built successfully")
        
        # Check if main files exist
        index_file = build_dir / "html" / "index.html" 
        if index_file.exists():
            print("✅ Index page generated")
        else:
            print("⚠️  Index page not found")
            
        return True
    else:
        print("❌ Documentation build failed")
        return False

def main():
    """Run all tests."""
    print("🔍 Testing ReadTheDocs Configuration")
    print(f"📁 Project root: {PROJECT_ROOT}")
    print(f"📚 Docs directory: {DOCS_DIR}")
    
    tests = [
        ("Python Imports", test_python_imports),
        ("Sphinx Config", test_sphinx_config),
        ("Notebooks", test_notebooks),
        ("Docs Build", test_docs_build),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! ReadTheDocs should build successfully.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix issues before pushing.")
        return 1

if __name__ == "__main__":
    sys.exit(main())