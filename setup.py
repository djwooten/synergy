import setuptools

import codecs
import os.path

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


with open("README.md", "r") as fh:
    long_description = fh.read()

def get_synergy_packages():
    packages=setuptools.find_packages(where='src')
    print("Hello", packages)
    return packages

setuptools.setup(
    name="synergy", # Replace with your own username
    version=get_version("src/synergy/__init__.py"),
    author="David J. Wooten",
    author_email="dwooten@psu.edu",
    description="Python package for calculating drug combination synergy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/djwooten/synergy",

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
#        'Programming Language :: Python :: 3.5', # not installed for testing
#        'Programming Language :: Python :: 3.6', # fails with import errors
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
    ],
#    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, <4',
    python_requires='>=3.7',
    keywords='synergy drug combination pharmacology cancer',
    #packages=setuptools.find_packages(where='src'),
    packages=get_synergy_packages(),
    package_dir={'': 'src'},
    install_requires=[
        "scipy", # 0.18.0 introduced curve_fit(jac=)
        "numpy" # 1.6.0 is first version compatible with python 3
    ],
    # package_data VS data_files VS ???
)
