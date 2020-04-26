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

from .. import utils
from ..single import Hill

class MuSyC():
    def __init__(self, E_bounds=(-np.inf,np.inf), h_bounds=(0,np,inf), C_bounds=(0,np.inf), alpha_bounds=(0,np.inf), gamma_bounds=(0,np.inf), r=1.):
        self.params = None
        self.n_dim = None
        
        # Given 3 drugs, there are 9 synergy edges, and so 9 synergistic potencies and cooperativities. Thus, alphas=[a,b,c,d,e,f,g,h,i]. _edge_index[2][6] will give the index of the alpha corresponding to going from state [010] to [110] (e.g., adding drug 3).
        self._edge_index = None

    def E(self, d):
        if len(d.shape) != 2:
            # d is not properly formatted
            return 0

        n_dim = d.shape[1]
        if (n_dim < 2):
            # MuSyC requires at least two drugs
            return 0

        return MuSyC._model(d, *self.params)
        

    def _model(d, *args):
        n_dim = d.shape[1]

        n_E = MuSyC._get_n_states(n_dim)
        n_h = n_dim
        n_C = n_dim
        n_alpha = MuSyC._get_n_edges(n_dim)-n_dim
        n_gamma = n_alpha

        matrix = np.zeros((n_E, n_E))

        return n_E + n_h + n_C + n_alpha + n_gamma

    def _is_neighbor(i,j,n_dim):
        a = MuSyC._get_drug_state(i, n_dim)
        b = MuSyC._get_drug_state(j, n_dim)

        if MuSyC._hamming(a,b) != 1: return 0
        for idx in range(len(a)):
            if a[idx] != b[idx]:
                if a[idx]=="1" return -(idx+1)
                return idx+1
        return 0
        
    def _hamming(a,b):
        s = 0
        for A,B in zip(a,b):
            if (A!=B): s+=1
        return s

    def _get_drug_state(idx, n_dim):
        return bin(idx)[2:].zfill(n_dim)

    def _get_idx(state):
        return int(state,base=2)

    def _get_n_states(n_dim):
        return 2**n_dim

    def _get_n_edges(n_dim):
        return 2**(n_dim-1)*n_dim


def idx_to_state(idx, n):
    return [int(i) for i in bin(idx)[2:].zfill(n)]

def state_to_idx(state):
    return int(''.join([str(i) for i in state]), base=2)

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
    state = idx_to_state(idx, n)
    add_drugs = []
    remove_drugs = []
    for i,val in enumerate(state):
        neighbor = [j for j in state]
        neighbor[i] = 1-val
        drug_idx = n-i-1
        if val==0:
            add_drugs.append((drug_idx, state_to_idx(neighbor)))
        else:
            remove_drugs.append((drug_idx, state_to_idx(neighbor)))
    return add_drugs, remove_drugs
    
def _build_edge_indices(n):
    _edge_index = dict()
    count = 0
    for i in range(1,2**n): 
        add_d, rem_d = get_neighbors(i,n)
        if len(add_d) > 0:
            _edge_index[i] = dict()
        for j in add_d:
            _edge_index[i][j[1]] = count
            count += 1
    return _edge_index

def build_matrix(doses):
    n = doses.shape[1]
    matrix = np.zeros((doses.shape[0],2**n,2**n))
    E_params = [2,1,1,1,1,0,0,0]
    h_params = [2,1,0.8]
    C_params = [0.1,0.01,0.1]
    alpha_params = [2,3,1,1,0.7,0.5,2,1,1]
    gamma_params = [0.4,2,1,2,0.7,3,2,0.5,2]

    params = E_params + h_params + C_params + alpha_params + gamma_params
    h_param_offset = 2**n
    C_param_offset = h_param_offset + n
    alpha_param_offset = C_param_offset + n
    gamma_param_offset = alpha_param_offset + 2**(n-1)*n-n

    E_params = params[:2**n]
    h_params = params[h_param_offset:C_param_offset]
    C_params = params[C_param_offset:alpha_param_offset]
    alpha_params = params[alpha_param_offset:gamma_param_offset]
    gamma_params = params[gamma_param_offset:]
    
    _edge_index = _build_edge_indices(n)
    # First do row 0, which corresponds to U
    # All edges are non-synergistic

    add_drugs, remove_drugs = _get_neighbors(0, n)
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
        add_drugs, remove_drugs = _get_neighbors(idx, n)
        for drugnum, jidx in add_drugs:
            gamma = 1
            alpha = 1
            # If the neighbor is not 0, this is a synergy edge
            if jidx > 0:
                edge_idx = _edge_index[idx][jidx]
                gamma = gamma_params[edge_idx]
                alpha = alpha_params[edge_idx]
            C = C_params[drugnum]
            r = 1
            r1r = r*np.power(C,h)
            d = doses[:,drugnum]
            
            # This state gains from reverse transitions out of jidx
            matrix[:,idx,jidx] += r1r**gamma

            # This state loses from transitions toward jidx
            matrix[:,idx,idx] -= np.power(r*np.power(alpha*d,h),gamma)

        for drugnum, jidx in remove_drugs:
            gamma = 1
            alpha = 1
            # If the neighbor is not 0, this is a synergy edge
            if jidx > 0:
                edge_idx = _edge_index[jidx][idx]
                gamma = gamma_params[edge_idx]
                alpha = alpha_params[edge_idx]
            C = C_params[drugnum]
            r = 1
            r1r = r*np.power(C,h)
            d = doses[:,drugnum]
            
            # This state loses from reverse transitions toward jidx
            matrix[:, idx, idx] -= r1r**gamma

            # This state gaines from transitions from jidx
            matrix[:, idx, jidx] += np.power(r*np.power(alpha*d,h),gamma)
        
    matrix[:,-1,:]=1
    matrix_inv = np.linalg.inv(matrix)

    return np.dot(np.dot(matrix_inv,np.asarray([0,0,0,0,0,0,0,1])), np.asarray(E_params))