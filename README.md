<img src="https://djwooten.github.io/img/synergy_logo.png" width="400" />

# synergy

A python package to calculate, analyze, and visualize drug combination synergy and antagonism. Currently supports multiple models of synergy, including MuSyC, Bliss, Loewe, Combination Index, ZIP, Zimmer, BRAID, Schindler, and HSA.

#### Citation

If you use, please cite:

Wooten, David J, and Albert, Réka. synergy - A Python library for calculating, analyzing, and visualizing drug combination synergy. (2020) Bioinformatics. https://doi.org/10.1093/bioinformatics/btaa826

## Installation

Using PIP
`pip install synergy`

Using conda
`not yet`

From source

```bash
git clone git@github.com:djwooten/synergy.git
cd synergy
pip install -e .
```



## Example Usage

### Parametric Models

#### Fit to data

For this, I assume you have access to a drug response data set that has (at least) the following columns.

| drug1.conc | drug2.conc | effect |
| ---------- | ---------- | ------ |
| 0          | 0          | 1      |
| 0          | 0.01       | 0.97   |
| 0          | 0.1        | 0.9    |
| 0          | 1          | 0.7    |
| ...        | ...        | ...    |

An example dataset can be found at https://raw.githubusercontent.com/djwooten/synergy/master/datasets/sample_data_1.csv

```python
from synergy.combination import MuSyC # or BRAID, Zimmer
import pandas as pd

df = pd.read_csv("your_own_drug_response_data.csv")

model = MuSyC(E0_bounds=(0,1), E1_bounds=(0,1), E2_bounds=(0,1), E3_bounds=(0,1))
model.fit(df['drug1.conc'], df['drug2.conc'], df['effect'], bootstrap_iterations=100)
```

Bounds are optional, but will help the fitting algorithm if you know them. Each model has different parameters that may mean different things, so you may wish to check your choice model's `__init__()` arguments.

#### Get parameters + confidence intervals

```python
model.get_parameters(confidence_interval=95)
```

Each synergy model has their own synergy parameters. Read their documentation and publications to understand what they mean. Confidence intervals are only generated if you set the number of `bootstrap_iterations` in `model.fit()`. The full results of the bootstrapping are stored in `model.bootstrap_parameters`.  If you request a 95% confidence interval, `get_parameters()` will calculate the 2.5% and 97.5% percentiles for each parameter.

#### Summarize synergy conclusions

```python
print(model.summary(confidence_interval=95))
```

This will report any parameters that are synergistic or antagonistic across the entire requested confidence interval. If the model was not fit with `bootstrap_iterations`, only the best fit value is used to determine synergism or antagonism.

#### Visualize

```python
# Requires matplotlib
model.plot_heatmap(df['drug1.conc'], df['drug2.conc'], xlabel="Drug1", ylabel="Drug2")

# Requires plotly
model.plot_surface_plotly(df['drug1.conc'], df['drug2.conc'], xlabel="Drug1", 	\
                          ylabel="Drug2", zlabel="Effect", fname="plotly.html", \
                          scatter_points=df)
```

Visualization requires the doses for drug 1 and drug 2 to be sampled on a complete rectangular grid. So for instance, if drug 1 is sampled at 0, 0.01, 0.1, 1, 10 uM (a total of 5 concentrations), and drug 1 is sampled at 0, 0.1, 1, 10 uM (a total of 4 concentrations), the doses and effects must cover all pairwise combinations (e.g., 5*4=20 points must be given). `scatter_points` is optional, but if given, it should be a pandas.DataFrame with (at least) columns "drug1.conc", "drug2.conc", and "effect".

#### Generate synthetic data

```python
from synergy.utils.dose_tools import grid
import numpy as np

model = MuSyC(E0=1, E1=0.6, E2=0.4, E3=0, h1=2, h2=0.8, C1=1e-2, C2=1e-1, \
              alpha12=2, alpha21=1, gamma12=2.5, gamma21=0.7)

# d1min, d1max, d2min, d2max, npoints1, npoints2
d1, d2 = grid(1e-3, 1e0, 1e-3, 1e0, 8, 8)

E = model.E(d1, d2)
```

### Nonparametric (dose dependent) synergy models

#### Fit to data

See above for an example dataset.

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
model.plot_heatmap(xlabel="Drug1", ylabel="Drug2")

# Requires plotly
model.plot_surface_plotly(xlabel="Drug1", ylabel="Drug2", 				\
                          zlabel="Loewe Synergy", fname="plotly.html")
```

### 3+ Drug Combination Synergy

Currently, only MuSyC, Loewe, Bliss, HSA, Combination Index, and Schindler can be used to calculate synergy of 3+ drug combinations. 3+ drug models are implemented separately (allowing for some optimizations for the two-drug case), so 3+ drug models are imported from `synergy.higher`

#### Fitting 3+ drug parametric models

```python
from synergy.higher import MuSyC
import pandas as pd

df = pd.read_csv("your_own_drug_response_data.csv")

model = MuSyC(E_bounds=(0,1), h_bounds=(1e-3,1e3)) # NOT E0_bounds, E1_bounds, etc...

model.fit(df[['drug1.conc','drug2.conc','drug3.conc']], df['effect'], 	\
          bootstrap_iterations=20)
```

Unlike the two drug case, doses are passed to the three+ drug case in a single matrix-like object, with M rows (number of samples) and N columns (number of drugs). Further, bounds are not specified for each individual drug, but rather `E_bounds`, `h_bounds`, etc, will be used for all drugs.

#### Visualization

For three drugs, E is a 3D scalar field. One option for visualization is to use `synergy.utils.plots.plot_heatmap(d1, d2, E)` where E is calculated across `d1` and `d2`, but on fixed slices of `d3`. An additional option is to use plotly isosurfaces, which render curved 2D surfaces of constant E in a 3D space. These isosurfaces are similar to stacks of heatmaps. This is implemented for `synergy.higher` models as

```
model.plotly_isosurfaces(d, drug_axes=[0,1,2])
```

`drug_axes` specifies which three drugs will be used for the axes of the isosurface plot. The other drugs will be kept at their minimum value, or can be manually fixed to specific values using the optional `other_drug_slices` argument.

#### Fitting and visualizing 3+ dose-dependent models

```python
from synergy.higher import Bliss # or Loewe, HSA, Schindler, CombinationIndex
import pandas as pd

df = pd.read_csv("your_own_drug_response_data.csv")

model = Bliss()

model.fit(df[['drug1.conc','drug2.conc','drug3.conc']], df['effect'])

# Visualizatoin
model.plotly_isosurfaces()
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
* Calculate 3+ drug synergy using
  * Parametric
    * MuSyC
  * Dose-dependent
    * Bliss
    * Loewe
    * Schindler
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
* Additional dose / experiment design tools
  * Alternative dosing strategies
* Heteroskedastic re-sampling for datasets with >= 3 replicates at each dose
* Parallelization API for fitting high-throughput screen data

## License
GNU General Public License v3 or later (GPLv3+)
