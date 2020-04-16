# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../src/synergy'))


# -- Project information -----------------------------------------------------

project = 'synergy'
copyright = '2020, David J Wooten'
author = 'David J Wooten'

# The full version, including alpha/beta/rc tags
release = 'v0.0.2'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "numpydoc",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'nature'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']




numpydoc_use_plots = False
#Whether to produce plot:: directives for Examples sections that contain import matplotlib or from matplotlib import.

numpydoc_show_class_members = False
#Whether to show all members of a class in the Methods and Attributes sections automatically. True by default.

numpydoc_show_inherited_class_members = True
#Whether to show all inherited members of a class in the Methods and Attributes sections automatically. If it’s false, inherited members won’t shown. True by default.

numpydoc_class_members_toctree = False
#Whether to create a Sphinx table of contents for the lists of class methods and attributes. If a table of contents is made, Sphinx expects each entry to have a separate page. True by default.

#numpydoc_citation_re : str
#A regular expression matching citations which should be mangled to avoid conflicts due to duplication across the documentation. Defaults to [\w-]+.

#numpydoc_use_blockquotes : bool
#Until version 0.8, parameter definitions were shown as blockquotes, rather than in a definition list. If your styling requires blockquotes, switch this config option to True. This option will be removed in version 0.10.

#numpydoc_attributes_as_param_list : bool
#Whether to format the Attributes section of a class page in the same way as the Parameter section. If it’s False, the Attributes section will be formatted as the Methods section using an autosummary table. True by default.

#numpydoc_xref_param_type : bool
#Whether to create cross-references for the parameter types in the Parameters, Other Parameters, Returns and Yields sections of the docstring. False by default.
#Note
#Depending on the link types, the CSS styles might be different. consider overriding e.g. span.classifier a span.xref and span.classifier a code.docutils.literal.notranslate CSS classes to achieve a uniform appearance.

#numpydoc_xref_aliases : dict
#Mappings to fully qualified paths (or correct ReST references) for the aliases/shortcuts used when specifying the types of parameters. The keys should not have any spaces. Together with the intersphinx extension, you can map to links in any documentation. The default is an empty dict.

#If you have the following intersphinx namespace configuration:

#    intersphinx_mapping = {
#        'python': ('https://docs.python.org/3/', None),
#        'numpy': ('https://docs.scipy.org/doc/numpy', None),
#        ...
#    }

#The default numpydoc_xref_aliases will supply some common Python standard library and NumPy names for you. Then for your module, a useful dict may look like the following (e.g., if you were documenting sklearn.model_selection):

#    numpydoc_xref_aliases = {
#        'LeaveOneOut': 'sklearn.model_selection.LeaveOneOut',
#        ...
#    }

#This option depends on the numpydoc_xref_param_type option being True.
#numpydoc_xref_ignore : set

#Words not to cross-reference. Most likely, these are common words used in parameter type descriptions that may be confused for classes of the same name. For example:

#    numpydoc_xref_ignore = {'type', 'optional', 'default'}

#    The default is an empty set.
#numpydoc_edit_link : bool

#    Deprecated since version edit: your HTML template instead

#    Whether to insert an edit link after docstrings.
