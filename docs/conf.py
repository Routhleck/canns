# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------
project = 'CANNs'
# Note: Using 'copyright' variable name is required by Sphinx
# ruff: noqa: A001
copyright = '2025, Sichao He'  # noqa: A001
author = 'Sichao He'

# Get version from the installed package or git tags
# For GitHub Pages deployment, use the latest git tag (formal release version)
# For local development, use the package version (includes dev version info)
if os.environ.get('GITHUB_ACTIONS') == 'true':
    # In GitHub Actions, use the latest git tag for formal release version
    try:
        import subprocess
        result = subprocess.run(['git', 'describe', '--tags', '--abbrev=0'],
                              capture_output=True, text=True, cwd=os.path.dirname(__file__))
        if result.returncode == 0:
            git_version = result.stdout.strip().lstrip('v')
            version = git_version
            release = git_version
        else:
            version = '0.10.0'
            release = '0.10.0'
    except Exception:
        version = '0.10.0'
        release = '0.10.0'
else:
    # For local development, use the package version (includes dev info)
    try:
        # Try to import from the package
        import canns
        version = canns.__version__
        release = version
    except (ImportError, AttributeError):
        # Fallback: try to get from git tags
        try:
            import subprocess
            result = subprocess.run(['git', 'describe', '--tags', '--abbrev=0'],
                                  capture_output=True, text=True, cwd=os.path.dirname(__file__))
            if result.returncode == 0:
                git_version = result.stdout.strip().lstrip('v')
                version = git_version
                release = git_version
            else:
                version = '0.10.0'
                release = '0.10.0'
        except Exception:
            version = '0.10.0'
            release = '0.10.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
    'autoapi.extension',
    'myst_parser',
    'sphinx_design',
    'nbsphinx',  # Jupyter notebook support
    'sphinxcontrib.bibtex',  # Bibliography support
    'sphinx_copybutton',  # Copy button for code blocks
]

# -- Bibliography configuration (sphinxcontrib-bibtex) ----------------------
bibtex_bibfiles = ['refs/references.bib']
bibtex_default_style = 'plain'  # Citation format: [1], [2], [3] (numeric labels)
# Note: Don't set bibtex_reference_style - let it use default numeric labels

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Jupyter Notebook configuration (nbsphinx) --------------------------------
nbsphinx_execute = 'never'  # Don't execute notebooks during build (display code only)
nbsphinx_allow_errors = False  # Fail build if notebooks have errors
nbsphinx_timeout = 300  # 5 minutes timeout per notebook
nbsphinx_kernel_name = 'python3'

# -- Copy button configuration (sphinx-copybutton) ----------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_remove_prompts = True
copybutton_line_continuation_character = "\\"

# -- AutoAPI configuration ---------------------------------------------------
autoapi_dirs = ['../src/']
autoapi_type = 'python'
autoapi_template_dir = '_templates/autoapi'
autoapi_root = 'autoapi'
autoapi_options = [
    'members',
    'undoc-members', 
    'show-inheritance',
    'show-module-summary',
    'special-members',
    'imported-members',
]
autoapi_generate_api_docs = True
autoapi_add_toctree_entry = True
# Suppress duplicate object warnings
suppress_warnings = ['autosummary', 'autosummary.import_cycle']
autoapi_ignore = ['*/_version.py', '*/py.typed', '**/py.typed']
autoapi_python_class_content = 'both'  # Include both class and __init__ docstrings
autoapi_member_order = 'groupwise'
autoapi_keep_files = True
# Additional settings to avoid import resolution errors
autoapi_python_use_implicit_namespaces = True

# -- Autodoc configuration ---------------------------------------------------
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'

# -- Options for HTML output ------------------------------------------------
html_theme = 'furo'
html_static_path = ['_static']

# Favicon
html_favicon = '_static/logo.svg'

# Custom CSS files
html_css_files = [
    'custom.css',
]

# Furo theme options
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
    "source_repository": "https://github.com/routhleck/canns/",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_logo": "logo.svg",
    "dark_logo": "logo.svg",
    "light_css_variables": {
        "color-brand-primary": "#2980b9",
        "color-brand-content": "#2980b9",
        "color-admonition-background": "transparent",
    },
    "dark_css_variables": {
        "color-brand-primary": "#79afd1",
        "color-brand-content": "#79afd1",
    },
}

# -- Internationalization ---------------------------------------------------
language = 'en'
locale_dirs = ['locale/']
gettext_compact = False

# -- MyST Parser options ----------------------------------------------------
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# -- Intersphinx mapping ----------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
}

# -- Napoleon settings -------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Options for LaTeX/PDF output -------------------------------------------
latex_engine = 'pdflatex'  # Use pdflatex for PDF generation

latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '11pt',
    'preamble': r'''
        \usepackage{bookmark}
        \usepackage{microtype}
    ''',
    'fncychap': '\\usepackage[Bjornstrup]{fncychap}',
    'printindex': '\\footnotesize\\raggedright\\printindex',
}

# LaTeX document configuration
latex_documents = [
    ('index', 'canns.tex', 'CANNs Documentation',
     'Sichao He', 'manual'),
]

# -- Options for ePub output -------------------------------------------------
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright
epub_exclude_files = ['search.html']
