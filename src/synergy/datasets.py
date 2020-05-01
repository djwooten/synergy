#    Copyright (C) 2020 David J. Wooten
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np

from .combination import MuSyC
from .higher import MuSyC as MuSyC_higher
from .utils import dose_tools
from .utils import base as utils
from .single import Hill


def ds_A(noise=0.05):
    E0, E1, E2, E3 = 1, 0.4, 0.3, 0
    h1, h2 = 2.3, 0.8
    C1, C2 = 0.035, 0.15
    alpha12, alpha21 = 0.3, 4.5
    gamma12, gamma21 = 3.7, 0.95

    model = MuSyC(E0=E0, E1=E1, E2=E2, E3=E3, h1=h1, h2=h2, C1=C1, C2=C2, alpha12=alpha12, alpha21=alpha21, gamma12=gamma12, gamma21=gamma21)
    
    d1, d2 = dose_tools.grid(C1/20, C1*20, C2/20, C2*20, 6, 6, include_zero=True)

    E = model.E(d1, d2)
    noise = noise*(E0-E3)

    np.random.seed(0)
    E = E + noise*(2*np.random.rand(len(d1))-1)
    
    return d1, d2, E

def ds_B(noise=0.05):
    E0, E1, E2, E3 = 1, 0.4, 0.1, 0.2
    h1, h2 = 2.3, 0.8
    C1, C2 = 0.035, 0.15
    alpha12, alpha21 = 1, 2.3
    gamma12, gamma21 = 0.5, 1.02

    model = MuSyC(E0=E0, E1=E1, E2=E2, E3=E3, h1=h1, h2=h2, C1=C1, C2=C2, alpha12=alpha12, alpha21=alpha21, gamma12=gamma12, gamma21=gamma21)
    
    d1, d2 = dose_tools.grid(C1/20, C1*20, C2/20, C2*20, 6, 6, include_zero=True)

    E = model.E(d1, d2)
    noise = noise*(E0-E3)

    np.random.seed(10)
    E = E + noise*(2*np.random.rand(len(d1))-1)
    
    return d1, d2, E

def ds_C(noise=0.05):
    E0, E1, E2, E3 = 1, 0, 0, 0
    h1, h2 = 2.3, 0.5
    C1, C2 = 0.035, 0.15
    alpha12, alpha21 = 0.7, 2.3
    gamma12, gamma21 = 0.9, 1.02

    model = MuSyC(E0=E0, E1=E1, E2=E2, E3=E3, h1=h1, h2=h2, C1=C1, C2=C2, alpha12=alpha12, alpha21=alpha21, gamma12=gamma12, gamma21=gamma21)
    
    d1, d2 = dose_tools.grid(C1/20, C1*20, C2/20, C2*20, 6, 6, include_zero=True)

    E = model.E(d1, d2)
    noise = noise*(E0-E3)

    np.random.seed(20)
    E = E + noise*(2*np.random.rand(len(d1))-1)
    
    return d1, d2, E



def bliss_independent(noise=0.05):
    E0, E1, E2, E3 = 1, 0.4, 0.3, 0.12
    h1, h2 = 2.3, 0.8
    C1, C2 = 0.035, 0.15
    alpha12, alpha21 = 1, 1
    gamma12, gamma21 = 1, 1

    model = MuSyC(E0=E0, E1=E1, E2=E2, E3=E3, h1=h1, h2=h2, C1=C1, C2=C2, alpha12=alpha12, alpha21=alpha21, gamma12=gamma12, gamma21=gamma21)
    
    d1, d2 = dose_tools.grid(C1/20, C1*20, C2/20, C2*20, 6, 6, include_zero=True)

    E = model.E(d1, d2)
    noise = noise*(E0-E3)

    np.random.seed(230)
    E = E + noise*(2*np.random.rand(len(d1))-1)
    
    return d1, d2, E

def linear_isobole(noise=0.05):
    E0, E1, E2, E3 = 1, 0.4, 0., 0
    h1, h2 = 1, 1
    C1, C2 = 0.035, 0.15
    alpha12, alpha21 = 0, 0
    gamma12, gamma21 = 1, 1
    
    model = MuSyC(E0=E0, E1=E1, E2=E2, E3=E3, h1=h1, h2=h2, C1=C1, C2=C2, alpha12=alpha12, alpha21=alpha21, gamma12=gamma12, gamma21=gamma21)
    
    d1, d2 = dose_tools.grid(C1/20, C1*20, C2/20, C2*20, 6, 6, include_zero=True)

    E = model.E(d1, d2)
    noise = noise*(E0-E3)

    np.random.seed(1)
    E = E + noise*(2*np.random.rand(len(d1))-1)
    
    return d1, d2, E

def sham(noise=0.05):
    E0, Emax = 1, 0
    h = 1
    C = 0.035
    drug = Hill(E0=E0, Emax=Emax, h=h, C=C)
    
    d = np.linspace(0,1.5*C,num=10)
    d1, d2, E = utils.sham(d, drug)

    noise = noise*(E0-Emax)
    np.random.seed(22)
    E = E + noise*(2*np.random.rand(len(d1))-1)
    
    return d1, d2, E


# ################### HIGHER

def ds_A_3(noise=0.05):
    E_params = [1,0.5,0.4,0.2,0.3,0.03,0.12,0.1]
    h_params = [2.3,0.8,1.3]
    C_params = [0.1,0.01,0.1]
    alpha_params = [2,3,1,1,0.7,0.5,2,1,1]
    gamma_params = [0.4,2,1,2,0.7,3,2,0.5,2]

    model = MuSyC_higher()
    model.parameters = E_params + h_params + C_params + alpha_params + gamma_params
    
    d = dose_tools.grid_multi((1e-3,1e-3,1e-3),(1,1,1),(6,6,6), include_zero=True)

    E = model.E(d)
    noise = noise*(max(E_params)-min(E_params))

    np.random.seed(64)
    E = E + noise*(2*np.random.rand(d.shape[0])-1)
    
    return d, E

def ds_B_3(noise=0.05):
    E_params = [1,0.45,0.4,0.4,0.3,0.3,0.3,0.]
    h_params = [0.3,0.8,1.3]
    C_params = [0.1,0.01,0.1]
    alpha_params = [2,3,1,1,0.7,0.5,0.2,1,1.5]
    gamma_params = [1.4,2,0.1,2.5,0.7,3,2,0.5,2]

    model = MuSyC_higher()
    model.parameters = E_params + h_params + C_params + alpha_params + gamma_params
    
    d = dose_tools.grid_multi((1e-3,1e-3,1e-3),(1,1,1),(6,6,6), include_zero=True)

    E = model.E(d)
    noise = noise*(max(E_params)-min(E_params))

    np.random.seed(6400)
    E = E + noise*(2*np.random.rand(d.shape[0])-1)
    
    return d, E

def ds_C_3(noise=0.05):
    E_params = [1,0.45,0.4,0.55,0.2,0.1,0.3,0.]
    h_params = [0.3,0.8,1.3]
    C_params = [0.1,0.01,0.1]
    alpha_params = [2,3,0.02,1,2.7,1.5,0.2,0.12,3.7]
    gamma_params = [1.4,2,0.1,2.5,0.7,3,2,0.5,2]

    model = MuSyC_higher()
    model.parameters = E_params + h_params + C_params + alpha_params + gamma_params
    
    d = dose_tools.grid_multi((1e-3,1e-3,1e-3),(1,1,1),(6,6,6), include_zero=True)

    E = model.E(d)
    noise = noise*(max(E_params)-min(E_params))

    np.random.seed(6004)
    E = E + noise*(2*np.random.rand(d.shape[0])-1)
    
    return d, E

def bliss_independent_3(noise=0.05):
    E_params = [1,0.5,0.4,0.2,0.3,0.15,0.12,0.06]
    h_params = [2.3,0.8,1.3]
    C_params = [0.1,0.01,0.1]
    alpha_params = [1,]*9
    gamma_params = [1,]*9

    model = MuSyC_higher()
    model.parameters = E_params + h_params + C_params + alpha_params + gamma_params
    
    d = dose_tools.grid_multi((1e-3,1e-3,1e-3),(1,1,1),(6,6,6), include_zero=True)

    E = model.E(d)
    noise = noise*(max(E_params)-min(E_params))

    np.random.seed(4)
    E = E + noise*(2*np.random.rand(d.shape[0])-1)
    
    return d, E

def linear_isobole_3(noise=0.05):
    E_params = [1,0.5,0.4,0.5,0.2,0,0,0]
    h_params = [1,1,1]
    C_params = [0.1,0.01,0.1]
    alpha_params = [0,]*9
    gamma_params = [1,]*9

    model = MuSyC_higher()
    model.parameters = E_params + h_params + C_params + alpha_params + gamma_params
    
    d = dose_tools.grid_multi((1e-3,1e-3,1e-3),(1,1,1),(6,6,6), include_zero=True)

    with np.errstate(divide='ignore'):
        E = model.E(d)
    noise = noise*(max(E_params)-min(E_params))

    np.random.seed(44)
    E = E + noise*(2*np.random.rand(d.shape[0])-1)
    
    return d, E

def sham_3(noise=0.05):
    E0, Emax = 1, 0
    h = 1
    C = 0.035
    drug = Hill(E0=E0, Emax=Emax, h=h, C=C)
    
    d = np.linspace(0,1.5*C,num=6)
    d, E = utils.sham_higher(d, drug, 3)

    noise = noise*(E0-Emax)
    np.random.seed(288)
    E = E + noise*(2*np.random.rand(d.shape[0])-1)
    
    return d, E


E_params = [1,0.5,0.4,0.5,0.2,0,0,0]
#h_params = [2,1,0.8]
h_params = [1,1,1]
C_params = [0.1,0.01,0.1]
alpha_params = [2,3,1,1,0.7,0.5,2,1,1]
#gamma_params = [0.4,2,1,2,0.7,3,2,0.5,2]
#alpha_params = [0,]*9
gamma_params = [1,]*9

params = E_params + h_params + C_params + alpha_params + gamma_params

truemodel = MuSyC()
truemodel.parameters = params

d = dose_tools.grid_multi((1e-3,1e-3,1e-3),(1,1,1),(6,6,6), include_zero=True)
