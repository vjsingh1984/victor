# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import datetime

# -- Path setup --------------------------------------------------------------

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Victor AI Framework'
copyright = f'{datetime.now().year}, Vijaykumar Singh'
author = 'Vijaykumar Singh'

# The short X.Y version
version = '0.5'
# The full version, including alpha/beta/rc tags
release = '0.5.8'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.coverage',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The suffix(es) of source filenames.
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The master toctree document.
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_title = 'Victor AI Framework'
html_logo = '_static/logo.png'
html_favicon = '_static/favicon.ico'

# Add any paths that contain custom static files (such as style sheets)
html_static_path = ['_static']

# -- Extension configuration -------------------------------------------------

# Napoleon settings (for Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
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

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Autosummary settings
autosummary_generate = True

# Type hints settings
set_type_checking_flag = True
always_document_param_types = True

# Intersphinx mapping (for linking to external documentation)
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'asyncio': ('https://docs.python.org/3/library/asyncio.html', None),
}

# -- Options for HTMLHelp output ---------------------------------------------

htmlhelp_basename = 'Victordoc'

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '10pt',
}

# Grouping the document tree into LaTeX files.
latex_documents = [
    (master_doc, 'Victor.tex', 'Victor AI Framework Documentation',
     'Vijaykumar Singh', 'manual'),
]

# -- Options for manual page output ------------------------------------------

man_pages = [
    (master_doc, 'victor', 'Victor AI Framework Documentation',
     [author], 1)
]

# -- Options for Texinfo output --------------------------------------------

texinfo_documents = [
    (master_doc, 'Victor', 'Victor AI Framework Documentation',
     author, 'Victor', 'Open-source agentic AI framework.',
     'Miscellaneous'),
]
