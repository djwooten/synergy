import numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objects as go

from synergy.utils import dose_tools
from synergy.higher import MuSyC
from synergy.utils import plots
    
    
E_params = [2,1,1,1,1,0,0,0]
h_params = [2,1,0.8]
C_params = [0.1,0.01,0.1]
#alpha_params = [2,3,1,1,0.7,0.5,2,1,1]
#gamma_params = [0.4,2,1,2,0.7,3,2,0.5,2]
alpha_params = [1,]*9
gamma_params = [1,]*9

params = E_params + h_params + C_params + alpha_params + gamma_params

truemodel = MuSyC()
truemodel.parameters = params


d = dose_tools.grid_multi((1e-3,1e-3,1e-3),(1,1,1),(10,10,10), include_zero=True)
#truemodel.plotly_isosurfaces(d, vmin=0, vmax=2, isomin=0.2, isomax=2)

d1 = d[:,0]
d2 = d[:,1]
d3 = d[:,2]
E = truemodel.E(d)
noise = 0.05
E_fit = E + noise*(E_params[0]-E_params[-1])*(2*np.random.rand(len(E))-1)

if False:
    fig = plt.figure(figsize=(15,8))
    for i,DD in enumerate(np.unique(d3)):
        mask = np.where(d3==DD)
        ax = fig.add_subplot(3,4,i+1)
        plots.plot_colormap(d1[mask], d2[mask], E[mask], ax=ax, vmin=0, vmax=2)

    plt.tight_layout()
    plt.show()


model = MuSyC(E_bounds=(0,2), h_bounds=(1e-3,1e3), alpha_bounds=(1e-5, 1e5), gamma_bounds=(1e-5,1e5))
model.fit(d, E_fit, bootstrap_iterations=10)
print(model.parameters)
if model.converged:
    print(model.get_parameter_range().T)



"""
[[ 2.00051866e+000  2.02743581e+000] E000   0
 [ 9.54405240e-001  1.02073778e+000] E001   1
 [ 1.00251605e+000  1.02151387e+000] E010   2
 [ 9.05183388e-001  1.00220782e+000] E011   3
 [ 9.58066963e-001  1.03765892e+000] E100   4
 [-4.06375348e+000 -5.87511711e-002] E101   5
 [-6.10862987e-002 -3.37465160e-002] E110   6
 [ 3.54793607e-002  4.05854702e+000] E111   7
 [ 1.92536161e+000  2.20938436e+000] h1
 [ 9.37268534e-001  1.00328285e+000] h2
 [ 7.52204732e-001  8.42893309e-001] h3
 [ 9.19156532e-002  1.04078766e-001] C1
 [ 9.07234395e-003  1.00609336e-002] C2
 [ 9.27698933e-002  1.15706844e-001] C3
 [ 5.38382992e-001  8.19610405e-001]  alpha_001_101  alpha_1_3
 [ 4.48683342e-010  4.48683342e-010]  alpha_001_011  alpha_1_2
 [ 8.44097691e-001  1.08844512e+000]  alpha_010_110  alpha_2_3
 [ 1.98455338e-001  6.54591925e-001]  alpha_010_011  alpha_2_1
 [ 9.76095359e-001  1.58906833e+000]  alpha_011_111  alpha_12_3     #Higher
 [ 4.17573510e-001  9.92527045e-001]  alpha_100_110  alpha_3_2
 [ 6.86679760e-001  1.67810200e+000]  alpha_100_101  alpha_3_1
 [ 3.13287026e-320              inf]  alpha_101_111  alpha_13_2     #Higher
 [ 1.14853096e+000  7.30596201e+000]  alpha_110_111  alpha_23_1     #Higher
 [ 8.90056072e-001  1.14328975e+000]  gamma_001_101  gamma_1_3
 [ 1.33579822e+003  1.33579822e+003]  gamma_001_011  gamma_1_2
 [ 8.52403439e-001  9.76121300e-001]  gamma_010_110  gamma_2_3
 [ 4.09087807e-001  1.65238193e+000]  gamma_010_011  gamma_2_1
 [ 9.16184174e-001  1.13820199e+000]  gamma_011_111  gamma_12_3     #Higher
 [ 9.81663688e-001  1.42852424e+000]  gamma_100_110  gamma_3_2
 [ 9.50343381e-001  1.04200025e+001]  gamma_100_101  gamma_3_1
 [ 0.00000000e+000  2.68846079e-001]  gamma_101_111  gamma_13_2     #Higher
 [ 6.98912096e-001  5.93276434e+000]] gamma_110_111  gamma_23_1     #Higher

"""