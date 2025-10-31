# Configuration file for the Sphinx documentation builder.
# -- Project information -----------------------------------------------------
project = 'Smile Extraction'
copyright = ''
author = 'Raeed Chowdhury'

# The full version, including alpha/beta/rc tags
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',
    'myst_parser',
]

autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']


# Ensure smile_extract is importable; fail with a clear error if not
try:
    import smile_extract
except ImportError:
    raise RuntimeError(
        "smile_extract must be installed in your environment to build docs. "
        "Run 'pip install -e .' or 'pip install .[doc]' in the repo root."
    )
