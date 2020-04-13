import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="synergy", # Replace with your own username
    version="0.0.1",
    author="David J. Wooten",
    author_email="dwooten@psu.edu",
    description="Python package for calculating drug combination synergy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/djwooten/synergy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
#    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, !=3.4.*, <4',
    python_requires='>=3.6',
    keywords='synergy drug combination pharmacology cancer',
#    package_dir={'': 'src/'},
#    packages=find_packages(where='src'),
    install_requires=[
        "scipy",
        "numpy",
    ],
    # package_data VS data_files VS ???
    
)
