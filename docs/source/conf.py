# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))  # Points to the project root

project = 'filter-stations'
copyright = '2025, Austin Kaburia'
author = 'Austin Kaburia'
release = '0.7.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',      # Core library for generating docs from code
    'sphinx.ext.napoleon',     # Parses your "Parameters/Returns" style docstrings
    #'sphinx.ext.viewcode',     # Adds links to source code - Disabled to hide source
    'sphinx.ext.githubpages',  # Optional, good if you switch hosting later
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Disable "View Source" links
html_show_sourcelink = False
html_copy_source = False



# Mock ALL external libraries to prevent ImportError on ReadTheDocs
autodoc_mock_imports = [
    "rioxarray", 
    "rasterio", 
    "huggingface_hub", 
    "networkx", 
    "dask", 
    "xarray", 
    "pandas", 
    "numpy", 
    "requests", 
    "tqdm", 
    "python-dateutil",
    "haversine",
    "sklearn",
    "zarr"
]
