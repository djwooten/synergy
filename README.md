<img src="https://djwooten.github.io/img/synergy_logo.png" width="400" />

# synergy

A python package to calculate, analyze, and visualize drug combination synergy and antagonism. Supports multiple models of synergy:
- MuSyC
- Bliss
- Loewe
- Combination Index
- ZIP
- Zimmer
- BRAID
- Schindler
- HSA

#### Citation

If you use, please cite:

> Wooten, David J, and Albert, Réka. synergy - A Python library for calculating, analyzing, and visualizing drug combination synergy. (2020) Bioinformatics. https://doi.org/10.1093/bioinformatics/btaa826

## Installation

Using PIP
`pip install synergy`

From source

```bash
git clone git@github.com:djwooten/synergy.git
cd synergy
pip install -e .
```

## Documentation

You can find extensive documentation and examples at http://synergy.readthedocs.io/

## Example Usage

The general usage is the same for most synergy models, though refer to the documentation at
http://synergy.readthedocs.io/ for specifics.

### Fitting models

For this, I assume you have access to a drug response data set that has (at least) the following columns.

| drug1.conc | drug2.conc | effect |
| ---------- | ---------- | ------ |
| 0          | 0          | 1      |
| 0          | 0.01       | 0.97   |
| 0          | 0.1        | 0.9    |
| 0          | 1          | 0.7    |
| ...        | ...        | ...    |

```python
from synergy.combination import MuSyC  # or any other model
import pandas as pd

df = pd.read_csv("/path/to/your_own_drug_response_data.csv")

# Instantiate the model. Bounds are optional for parametric models, and will be used when fitting to data.
# In this case, imagine that the effect data we have is known to fall between 0 and 1. Further, the given h and alpha
# bounds here are reasonable for most datasets, as changing those parameters has the most impact near `1`.
model = MuSyC(E_bounds=(0, 1), h_bounds=(1e-3, 1e3), alpha_bounds=(1e-3, 1e3))

# Prepare the input data to be fit
d1 = df["drug1.conc"]
d2 = df["drug2.conc"]
E = df["effect"]

# Fit the model (bootstrap_iterations is an option for some models to estimate parameter
model.fit(d1, d2, E, bootstrap_iterations=100)

model.summarize()
```
The last call to `model.summarize()` will print a table summarizing the synergy findings. This is only available for
parametric synergy models. For example, the table may look like

```
Parameter  |  Value  |  95% CI          |  Comparison  |  Synergy
=====================================================================
beta       |  0.261  |  (0.175, 0.333)  |  > 0         |  synergistic
alpha12    |  3.54   |  (2.67, 4.93)    |  > 1         |  synergistic
alpha21    |  1.29   |  (0.845, 2.45)   |  ~= 1        |  additive
gamma12    |  0.947  |  (0.762, 1.2)    |  ~= 1        |  additive
gamma21    |  0.722  |  (0.487, 1.2)    |  ~= 1        |  additive
```

### Visualization

Many utilities exist under `synergy.utils.plots` to create heatmaps, 3D interactive dose-response surfaces, and 3D
interactive isosurfaces. These require intsalling `matplotlib` for the heatmaps, and `plotly`  for the 3d plots. Many
exmaples can be seen at http://synergy.readthedocs.io/models/synergy_models.html/.

The basic approach is

```python
from synergy.utils.plots import plot_heatmap, plot_plotly_surface

plot_heatmap(d1, d2, E, title="Dose response surface", fname="heatmap.png")
plot_surface_plotly(d1, d2, E, title="Dose response surface", fname="surface.html")
```

## License
GNU General Public License v3 or later (GPLv3+)
