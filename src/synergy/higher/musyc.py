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
from scipy.optimize import curve_fit

#from .. import utils
#from ..single import Hill

class MuSyC():
    def __init__(self, E_bounds=(-np.inf,np.inf), h_bounds=(0,np.inf), C_bounds=(0,np.inf), alpha_bounds=(0,np.inf), gamma_bounds=(0,np.inf), r=1.):
        self.params = None
        self.r = r

        # Given 3 drugs, there are 9 synergy edges, and so 9 synergistic potencies and cooperativities. Thus, alphas=[a,b,c,d,e,f,g,h,i]. _edge_index[2][6] will give the index of the alpha corresponding to going from state [010] to [110] (e.g., adding drug 3).
        self._edge_index = None

    def fit(self, d, E):
        p0 = self._get_initial_guess(d, E)
        popt, pcov = curve_fit(self._model, d, E, p0=p0)
        return popt

    def E(self, d):
        if len(d.shape) != 2:
            # d is not properly formatted
            return 0

        n = d.shape[1]
        if (n < 2):
            # MuSyC requires at least two drugs
            return 0

        self._build_edge_indices(n)
        params = MuSyC._transform_params_to_fit(self.params)
        return self._model(d, *params)

    def set_params(self, params):
        self.params = params

    @staticmethod
    def _transform_params_to_fit(params):
        params = np.asarray(params)
        n = MuSyC._get_n_drugs_from_params(params)
        if n<2:
            return 0
        
        h_param_offset = 2**n
        params[h_param_offset:] = np.log(params[h_param_offset:])
        return params

    @staticmethod
    def _transform_params_from_fit(popt):
        params = np.asarray(popt)
        n = MuSyC._get_n_drugs_from_params(params)
        if n<2:
            return 0
        
        h_param_offset = 2**n
        params[h_param_offset:] = np.exp(params[h_param_offset:])
        return params

    @staticmethod
    def _get_n_drugs_from_params(params):
        n = 2
        while len(params) > 2**n + n*2**n:
            n += 1
        if len(params) == 2**n+n*2**n:
            return n
        return 0

    def _get_initial_guess(self, d, E):
        # TODO Implement bounds sanitizing, and good guesses based on single-drug and pairwise drug combinations
        n = d.shape[1]

        E_params = [0,]*(2**n)
        
        for idx in range(2**n):
            # state = [0,1,1] means drug3=0, drug2=1, drug1=1
            state = MuSyC._idx_to_state(idx, n)
            mask = d[:,0]>0
            for drugnum in range(1,n+1): # 1, 2, 3, ...
                drugstate = state[n-drugnum] # 1->2, 2->1, 3->0
                if drugstate==0:
                    mask = mask & (d[:,drugnum-1]==np.min(d[:,drugnum-1]))
                else:
                    mask = mask & (d[:,drugnum-1]==np.max(d[:,drugnum-1]))
            E_params[idx] = np.median(E[mask])

        h_params = [1,]*n
        C_params = [np.median(d[:,i]) for i in range(n)]
        alpha_params = [1,]*(2**(n-1)*n-n)
        gamma_params = [1,]*(2**(n-1)*n-n)

        p0 = E_params + h_params + C_params + alpha_params + gamma_params
        return MuSyC._transform_params_to_fit(p0)
        
    @staticmethod
    def _hamming(a,b):
        s = 0
        for A,B in zip(a,b):
            if (A!=B): s+=1
        return s

    @staticmethod
    def _get_n_states(n):
        return 2**n

    @staticmethod
    def _get_n_edges(n):
        return 2**(n-1)*n

    @staticmethod
    def _idx_to_state(idx, n):
        """Returns state representing index=idx

        MuSyC models states for cells affected by each drug in all possible combinations. For instance, given three drugs, MuSyC will have states 000, 001, 010, 011, 100, 101, 110, 111. 110 means affected by drugs 3 and 2, but unaffected by drug 1.

        Parameters
        ----------
        idx : int
            Index for the state, ranging from 0 to 2**n-1.
        
        n : int
            Number of drugs in combination

        Returns
        -------
        state : list of int
            A list indicating which drugs are "active". [1,1,0] means drug 1 is not active (0), and drugs 2 and 3 are active.
        """
        return [int(i) for i in bin(idx)[2:].zfill(n)]

    @staticmethod
    def _state_to_idx(state):
        """Converts a drug state to its index.
        See _idx_to_state() for more info.

        Parameters
        ----------
        state : list of int
            
        Returns
        -------
        idx : int
        """
        return int(''.join([str(i) for i in state]), base=2)

    @staticmethod
    def _get_neighbors(idx, n):
        """Returns neighbors of the drug.

        Returns
        -------
        add_drugs : list
            List of tuples. For each tuple, the first element indicates \
            which drug is being added. The second element indicates the \
            idx of the state reached by adding that drug.

        remove_drugs : list
            As above, but for states reached by removing drugs.
        """
        state = MuSyC._idx_to_state(idx, n)
        add_drugs = []
        remove_drugs = []
        for i,val in enumerate(state):
            neighbor = [j for j in state]
            neighbor[i] = 1-val
            drug_idx = n-i-1
            if val==0:
                add_drugs.append((drug_idx, MuSyC._state_to_idx(neighbor)))
            else:
                remove_drugs.append((drug_idx, MuSyC._state_to_idx(neighbor)))
        return add_drugs, remove_drugs
    
    def _build_edge_indices(self, n):
        self._edge_index = dict()
        count = 0
        for i in range(1,2**n): 
            add_d, rem_d = MuSyC._get_neighbors(i,n)
            if len(add_d) > 0:
                self._edge_index[i] = dict()
            for j in add_d:
                self._edge_index[i][j[1]] = count
                count += 1

    def _model(self, doses, *args):
        """Model for higher dimensional MuSyC.

        Parameters
        ----------
        doses : numpy.ndarray
            M x N ndarray, where M is the number of samples, and N is the number of drugs.

        args
            Parameters for the model, in order of E, h, C, alpha, gamma. The number of each type of parameter is E (2**N), h and C (N), alpha and gamma (2**(N-1)*N-N).
        """
        n = doses.shape[1]
        matrix = np.zeros((doses.shape[0],2**n,2**n))
        
        h_param_offset = 2**n
        C_param_offset = h_param_offset + n
        alpha_param_offset = C_param_offset + n
        gamma_param_offset = alpha_param_offset + 2**(n-1)*n-n

        E_params = args[:2**n]
        logh_params = np.asarray(args[h_param_offset:C_param_offset])
        logC_params = np.asarray(args[C_param_offset:alpha_param_offset])
        logalpha_params = np.asarray(args[alpha_param_offset:gamma_param_offset])
        loggamma_params = np.asarray(args[gamma_param_offset:])

        h_params = np.exp(logh_params)
        C_params = np.exp(logC_params)
        alpha_params = np.exp(logalpha_params)
        gamma_params = np.exp(loggamma_params)

        # First do row 0, which corresponds to U
        # All edges are non-synergistic

        add_drugs, remove_drugs = MuSyC._get_neighbors(0, n)
        for drugnum, jidx in add_drugs:
            d = doses[:,drugnum]
            h = h_params[drugnum]
            C = C_params[drugnum]
            r = 1
            r1r = r*np.power(C,h)
            
            matrix[:,0,0] -= r*np.power(d,h)
            matrix[:,0,jidx] = r1r

        # Loop over all other states/rows (except the last one)
        for idx in range(1,2**n-1):
            row = [0,]*(2**n)
            add_drugs, remove_drugs = MuSyC._get_neighbors(idx, n)
            for drugnum, jidx in add_drugs:
                gamma = 1
                alpha = 1
                # If the neighbor is not 0, this is a synergy edge
                if jidx > 0:
                    edge_idx = self._edge_index[idx][jidx]
                    gamma = gamma_params[edge_idx]
                    alpha = alpha_params[edge_idx]
                d = doses[:,drugnum]
                h = h_params[drugnum]
                C = C_params[drugnum]
                r = 1
                r1r = r*np.power(C,h)
                
                # This state gains from reverse transitions out of jidx
                matrix[:,idx,jidx] += r1r**gamma

                # This state loses from transitions toward jidx
                matrix[:,idx,idx] -= np.power(r*np.power(alpha*d,h),gamma)

            for drugnum, jidx in remove_drugs:
                gamma = 1
                alpha = 1
                # If the neighbor is not 0, this is a synergy edge
                if jidx > 0:
                    edge_idx = self._edge_index[jidx][idx]
                    gamma = gamma_params[edge_idx]
                    alpha = alpha_params[edge_idx]
                d = doses[:,drugnum]
                h = h_params[drugnum]
                C = C_params[drugnum]
                r = 1
                r1r = r*np.power(C,h)
                
                # This state loses from reverse transitions toward jidx
                matrix[:, idx, idx] -= r1r**gamma

                # This state gaines from transitions from jidx
                matrix[:, idx, jidx] += np.power(r*np.power(alpha*d,h),gamma)
            
        # The final constraint is that U+A1+A2+... = 1
        matrix[:,-1,:]=1
        matrix_inv = np.linalg.inv(matrix)

        # All other rows should multiply to zero. Only the last row goes to 1.
        b = np.zeros(2**n)
        b[-1]=1
        return np.dot(np.dot(matrix_inv,b), np.asarray(E_params))


if __name__=="__main__":
    from matplotlib import pyplot as plt
    
    
    E_params = [2,1,1,1,1,0,0,0]
    h_params = [2,1,0.8]
    C_params = [0.1,0.01,0.1]
    #alpha_params = [2,3,1,1,0.7,0.5,2,1,1]
    #gamma_params = [0.4,2,1,2,0.7,3,2,0.5,2]
    alpha_params = [1,]*9
    gamma_params = [1,]*9

    params = E_params + h_params + C_params + alpha_params + gamma_params

    model = MuSyC()
    model.params = params

    n_points = 10
    D = np.logspace(-3,0,n_points)
    d1, d2, d3 = np.meshgrid(D,D,D)
    d1 = d1.flatten()
    d2 = d2.flatten()
    d3 = d3.flatten()

    d = np.zeros((n_points**3,3))
    d[:,0] = d1
    d[:,1] = d2
    d[:,2] = d3

    E = model.E(d)
    for i in D:
        for j in D:
            mask = np.where((d2==i)&(d3==j))
            plt.plot(d1[mask], E[mask])
    plt.xscale('log')
    plt.show()


    import plotly.graph_objects as go

    fig = go.Figure(data=go.Isosurface(
        x=np.log10(d1),
        y=np.log10(d2),
        z=np.log10(d3),
        value=E,
        isomin=0.1,
        isomax=2,
        opacity=0.6,
        colorscale='PRGn_r',
        surface_count=10, # number of isosurfaces, 2 by default: only min and max
        colorbar_nticks=10, # colorbar ticks correspond to isosurface values
        caps=dict(x_show=False, y_show=False)
        ))
    fig.show()


    from synergy.utils import plots
    fig = plt.figure(figsize=(15,6))
    for i,DD in enumerate(D):
        mask = np.where(d2==DD)
        ax = fig.add_subplot(2,5,i+1)
        plots.plot_colormap(d1[mask], d3[mask], E[mask], ax=ax, vmin=0, vmax=2)
    plt.show()

    
