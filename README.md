<img src="logo.png" width="400" />

# synergy

A python package to calculate, analyze, and visualize drug combination synergy and antagonism. Currently supports multiple models of synergy, including MuSyC, Bliss, Loewe, Combination Index, ZIP, Zimmer, Schindler, and HSA.

## Installation

Using PIP
`pip install synergy`

Using conda
`not yet`

Using git
`git clone ...`

## Requirements
* python >= 3.5
* numpy >= 1.13.0
* scipy >= 0.18.0
* Optional for full plotting functionality
  * matplotlib
  * plotly

## Current features
* Calculate two-drug synergy using
  * Parametric
    * MuSyC
    * Zimmer (effective dose model)
    * BRAID
  * Dose-dependent
    * Bliss
    * Loewe
    * Schindler
    * ZIP
    * HSA
    * Combination Index
* Residual bootstrap re-sampling to obtain confidence intervals for parameters of parametric models
* Single drug models
  * Parametric
    * Four-parameter Hill equation
    * Two-parameter Hill equation
    * Median-effect equation
  * Non-parametric
    * Piecewise linear
* Model scoring
  * R-squared
  * Akaike Information Criterion
  * Bayesian Information Criterion
* Visualization
  * Heatmaps
  * 3D Plotly Surfaces
* Synthetic data tools
  * Drug dilutions using grid-based sampling
  * "Sham experiment" simulation

## Planned features
* Additional models
  * Parametric
    * GPDI
* Three+ drug combinations (when possible)
* Visualization
  * Highlight single-drug curves on 3D surface plots
  * matplotlib 3D surface plotting
  * Contour plots for heatmaps
  * Isobolgrams
* Additional dose / experiment design tools
  * Alternative dosing strategies
* Heteroskedastic re-sampling for datasets with >= 3 replicates at each dose
* Parallelization API for fitting high-throughput screen data

## License
GNU General Public License v3 or later (GPLv3+)
