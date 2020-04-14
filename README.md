# synergy

## Installation

Using PIP
`not yet`

Using conda
`not yet`

Using git
`git clone ...`

## Requirements
* python 3
* numpy
* scipy

## Current features
* Calculate two-drug synergy using
  * Parametric
    * MuSyC
    * Zimmer (effective dose model)
  * Dose-dependent
    * Bliss
    * Loewe
    * Schindler
    * ZIP
    * HSA
* Single drug models
  * Four-parameter Hill equation
  * Two-parameter Hill equation
  * Median-effect equation
* Model scoring
  * R-squared
  * Akaike Information Criterion
  * Bayesian Information Criterion

## Planned features
* Additional models
  * Parametric
    * GPDI
    * BRAID
  * Dose-dependent
    * Combination Index
* Three+ drug combinations (when possible)
* Plotting functions
* Dose / experiment design tools
* Monte carlo sampling to obtain error bars
* Parallelization API for fitting high-throughput screen data

## Current version
0.0.1

## License
GNU General Public License v3 or later (GPLv3+)
