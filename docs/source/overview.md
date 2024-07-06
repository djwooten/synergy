Overview
========

`synergy` can be used to fit dose-response data to single-drug or drug-combination datasets, as well as visualize the results. Detailed examples are given for each supported model below.

## Supported models

### Single-drug

- [Log-linear](models/single/log_linear)
- [Hill](models/single/hill)

### Synergy

- [Bliss](models/synergy/bliss)
- [BRAID](models/synergy/braid)
- [Combination Index](models/synergy/combination_index)
- [HSA](models/synergy/hsa)
- [Loewe](models/synergy/loewe)
- [MuSyC](models/synergy/musyc)
- [Schindler](models/synergy/schindler)
- [Zimmer](models/synergy/zimmer)
- [ZIP](models/synergy/zip)

## Choosing a model

In general you should understand the assumptions, limitations, and interpretation of any synergy model you use. Here is a quick guide that can get you started:

![Choosing a synergy model](_static/synergy_flowchart.svg "Choosing a model")