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

setuptools.setup(
    name="synergy", # Replace with your own username
    version=get_version("synergy/__init__.py"),
    author="David J. Wooten",
    author_email="dwooten@psu.edu",
    description="Python package for calculating drug combination synergy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/djwooten/synergy",
    #packages=setuptools.find_packages("synergy"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
#    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, <4',
    python_requires='>=3.6',
    keywords='synergy drug combination pharmacology cancer',
#    package_dir={'synergy': 'src/'},
    packages=setuptools.find_packages(where='synergy'),
    install_requires=[
        "scipy", # 0.18.0 introduced curve_fit(jac=)
        "numpy" # 1.6.0 is first version compatible with python 3
    ],
    # package_data VS data_files VS ???
    
)
