<img src="https://djwooten.github.io/img/synergy_logo.png" width="400" />

# synergy

A python package to calculate, analyze, and visualize drug combination synergy and antagonism. Currently supports multiple models of synergy, including MuSyC, Bliss, Loewe, Combination Index, ZIP, Zimmer, BRAID, Schindler, and HSA.

## Installation

Using PIP
`pip install synergy`

Using conda
`not yet`

Using git
`git clone ...`

## Example Usage

### Parametric Models

#### Fit to data

```python
from synergy.combination import MuSyC # or BRAID, Zimmer
import pandas as pd

df = pd.read_csv("your_own_drug_response_data.csv")

model = MuSyC(E0_bounds=(0,1), E1_bounds=(0,1), E2_bounds=(0,1), E3_bounds=(0,1))
model.fit(df['drug1.conc'], df['drug2.conc'], df['effect'], bootstrap_iterations=100)
```

Bounds are optional, but will help the fitting algorithm if you know them. Each model has different parameters that may mean different things, so you may wish to check your choice model's `__init__()` arguments.

#### Get parameters and confidence intervals

```python
print(model)
print(model.get_parameter_range(confidence_interval=95).T)
```

Each synergy model has their own synergy parameters. Read their documentation and publications to understand what they mean. Confidence intervals are only generated if you set the number of bootstrap_iterations in `model.fit()`. The full results of the bootstrapping are stored in `model.bootstrap_parameters`.  If you request a 95% confidence interval, `get_parameter_range()` will calculate the 2.5% and 97.5% percentiles for each parameter.

#### Visualize

```python
# Requires matplotlib
model.plot_colormap(df['drug1.conc'], df['drug2.conc'], xlabel="Drug1", ylabel="Drug2")

# Requires plotly
model.plot_surface_plotly(df['drug1.conc'], df['drug2.conc'], xlabel="Drug1", 	\
                          ylabel="Drug2", zlabel="Effect", fname="plotly.html", \
                          scatter_points=df)
```

`scatter_points` is optional, but if given, it should be a pandas.DataFrame with (at least) columns "drug1.conc", "drug2.conc", and "effect".

#### Generate synthetic data

```python
from synergy.utils.dose_tools import grid
import numpy as np

model = MuSyC(E0=1, E1=0.6, E2=0.4, E3=0, h1=2, h2=0.8, C1=1e-2, C2=1e-1, \
              oalpha12=2, oalpha21=1, gamma12=2.5, gamma21=0.7)

# d1min, d1max, d2min, d2max, npoints1, npoints2
d1, d2 = grid(1e-3, 1e0, 1e-3, 1e0, 8, 8)

E = model.E(d1, d2)
```

### Nonparametric (dose dependent) synergy models

#### Fit to data

```python
from synergy.combination import Loewe # or Bliss, ZIP, HSA, Schindler, CombinationIndex
import pandas as pd

df = pd.read_csv("your_own_drug_response_data.csv")
model = Loewe()
model.fit(df['drug1.conc'], df['drug2.conc'], df['effect'])
```

#### Get synergy values

```python
print(model.synergy) # Will have size equal to d1, d2, and E passed to fit()
```

#### Visualize

```python
# Requires matplotlib
model.plot_colormap(df['drug1.conc'], df['drug2.conc'], xlabel="Drug1", ylabel="Drug2")

# Requires plotly
model.plot_surface_plotly(df['drug1.conc'], df['drug2.conc'], xlabel="Drug1",		\
                          ylabel="Drug2", zlabel="Loewe Synergy", fname="plotly.html")
```

## Requirements

* python >= 3.5
* numpy >= 1.13.0
* scipy >= 0.18.0
* Optional for full plotting functionality
  * matplotlib
  * plotly
  * pandas

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
  * MuSyC
  * Bliss
  * Loewe / CI
  * HSA
  * Schindler
  * Zimmer (at least incorporating pairwise synergies)
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
