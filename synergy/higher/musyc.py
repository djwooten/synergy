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

from typing import Sequence
import numpy as np

from synergy import utils
from synergy.exceptions import ModelNotParameterizedError
from synergy.higher.synergy_model_Nd import ParametricSynergyModelND
from synergy.single.dose_response_model_1d import DoseResponseModel1D
from synergy.single.hill import Hill


class MuSyC(ParametricSynergyModelND):
    """-"""

    # Bounds will depened on the number of dimensions, so will be filled out in _get_initial_guess()
    def __init__(
        self,
        single_drug_models: Sequence[DoseResponseModel1D] = None,
        num_drugs: int = -1,
        r=1.0,
        fit_gamma=False,
        **kwargs,
    ):
        """Ctor."""
        # Given 3 drugs, there are 9 synergy edges, and so 9 synergistic potencies and cooperativities.
        # Thus, alphas=[a,b,c,d,e,f,g,h,i]. _edge_index[2][6] will give the index of the alpha corresponding to going
        # from state [010] to [110] (e.g., adding drug 3).
        if single_drug_models:
            self.N = len(single_drug_models)
        else:
            self.N = num_drugs
        self._edge_index = MuSyC._get_edge_indices(self.N)
        self.fit_gamma = fit_gamma

        super().__init__(single_drug_models=single_drug_models, num_drugs=num_drugs, **kwargs)

        self.r = r

        if not self.fit_gamma:
            self.fit_function = self._model_no_gamma
        else:
            self.fit_function = self._model

    def _transform_params_to_fit(self, params):
        """Transform linear parameters to log-scale for fitting.

        Params come in order E, h, C, alpha, gamma. So everything past E should be log-scaled.
        """
        params = np.array(params, copy=True)
        h_param_offset = self._num_E_params
        params[h_param_offset:] = np.log(params[h_param_offset:])
        return params

    def _transform_params_from_fit(self, params):
        """Transform logscaled parameters to linear scale.

        Params come in order E, h, C, alpha, gamma. So everything past E should be exponentiated.
        """
        params = np.array(params, copy=True)
        h_param_offset = self._num_E_params
        params[h_param_offset:] = np.exp(params[h_param_offset:])
        return params

    def _todo_do_bounds(self):
        """TODO: I'm not sure if I can use this method to get bounds?

        Currently I use ParametricModelMixins.set_bounds()
        Which relies on names
        """
        n = self.N

        n_E = 2**n
        n_h = n
        n_C = n
        n_alpha = 2 ** (n - 1) * n - n
        n_gamma = 2 ** (n - 1) * n - n

        # TODO can I use this for main bounds?
        E_bounds = [
            self.E_bounds,
        ] * n_E
        logh_bounds = [
            self.logh_bounds,
        ] * n_h
        logC_bounds = [
            self.logC_bounds,
        ] * n_C
        logalpha_bounds = [
            self.logalpha_bounds,
        ] * n_alpha

        if not self.fit_gamma:
            bounds = E_bounds + logh_bounds + logC_bounds + logalpha_bounds
        else:
            loggamma_bounds = [
                self.loggamma_bounds,
            ] * n_gamma
            bounds = E_bounds + logh_bounds + logC_bounds + logalpha_bounds + loggamma_bounds

        self.bounds = tuple(zip(*bounds))

    def _get_initial_guess(self, d, E, p0):
        """-"""
        if p0 is None:
            E_params = [0] * self._num_E_params  # These will all be overridden by (1) single drug fits or (2) E(dmax)
            h_params = [1] * self._num_h_params  # These may all be overridden by single drug fits
            C_params = list(np.exp(np.median(np.log(d), axis=0)))  # These may all be overridden by single drug fits
            alpha_params = [1] * self._num_alpha_params  # These will not be overrideen
            gamma_params = [1] * self._num_gamma_params  # These will not be overrideen

            # Make guesses of E for each drug state
            for idx in range(self._num_E_params):
                # state = [0,1,1] means drug3=0, drug2=1, drug1=1
                state = MuSyC._idx_to_state(idx, self.N)
                mask = d[:, 0] > -1  # d is always > 0, so initializes to array of True
                for drugnum in range(self.N):  # e.g., 0, 1, 2 (N=3)
                    drugstate = state[self.N - drugnum - 1]  # e.g., 0->2, 1->1, 2->0  (N=3)
                    if drugstate == 0:
                        mask = mask & (d[:, drugnum] == np.min(d[:, drugnum]))
                    else:
                        mask = mask & (d[:, drugnum] == np.max(d[:, drugnum]))
                mask = np.where(mask)
                E_params[idx] = np.median(E[mask])

            # Make guesses for E, h, C of undrugged and single-drugged states
            E0_avg = 0

            for i, single_drug_model in enumerate(self.single_drug_models):
                if single_drug_model.is_specified:
                    E0_avg += single_drug_model.E0 / self.N
                    E_params[i] = single_drug_model.Emax
                    h_params[i] = single_drug_model.h
                    C_params[i] = single_drug_model.C
                else:
                    E0_avg += E_params[0] / self.N
            E_params[0] = E0_avg

            if not self.fit_gamma:
                p0 = E_params + h_params + C_params + alpha_params
            else:
                p0 = E_params + h_params + C_params + alpha_params + gamma_params

        p0 = list(self._transform_params_to_fit(p0))
        utils.sanitize_initial_guess(p0, self.bounds)
        return p0

    @staticmethod
    def _hamming(a, b):
        s = 0
        for A, B in zip(a, b):
            if A != B:
                s += 1
        return s

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
        return int("".join([str(i) for i in state]), base=2)

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
        for i, val in enumerate(state):
            neighbor = [j for j in state]
            neighbor[i] = 1 - val
            drug_idx = n - i - 1
            if val == 0:
                add_drugs.append((drug_idx, MuSyC._state_to_idx(neighbor)))
            else:
                remove_drugs.append((drug_idx, MuSyC._state_to_idx(neighbor)))
        return add_drugs, remove_drugs

    @staticmethod
    def _get_edge_indices(n: int) -> dict[int, dict[int, int]]:
        """Return a map of start state, end state, to index of associated alpha (gamma) parameter

        When evaluating the model, we need to know which alpha (gamma) parameter to use for a given edge. This method
        calculates those indices.

        For example, with 3 drugs, there are 9 synergy edges. So alpha_params will have 9 elements. The edge_index from
        this map will indicate which parameter to use.

        :param int n: Number of drugs
        :return dict[int, dict[int, int]]: Map of start state, end state, to index of associated alpha (gamma) parameter
        """
        edge_index: dict[int, dict[int, int]] = dict()
        count = 0
        for i in range(1, 2**n):
            add_d, _rem_d = MuSyC._get_neighbors(i, n)  # only map edges corresponding to adding drugs, so ignore rem_d.
            if len(add_d) > 0:
                edge_index[i] = dict()
            for j in add_d:
                edge_index[i][j[1]] = count
                count += 1
        return edge_index

    @property
    def _parameter_names(self) -> list[str]:
        """-"""
        param_names = []
        gamma_names = []
        for i in range(self._num_E_params):
            drug_string = MuSyC._get_drug_string_from_state(MuSyC._idx_to_state(i, self.N))
            param_names.append(f"E_{drug_string}")
        for i in range(self._num_h_params):
            param_names.append(f"h_{i}")
        for i in range(self._num_C_params):
            param_names.append(f"C_{i}")
        for start_state in self._edge_index:
            for end_state in self._edge_index[start_state]:
                edge_string = MuSyC._get_drug_string_from_edge(
                    MuSyC._idx_to_state(start_state, self.N), MuSyC._idx_to_state(end_state, self.N)
                )
                param_names.append(f"alpha_{edge_string}")
                gamma_names.append(f"gamma_{edge_string}")

        if self.fit_gamma:
            return param_names + gamma_names
        return param_names

    @property
    def _default_fit_bounds(self) -> dict[str, tuple[float, float]]:
        """-"""
        bounds: dict[str, tuple[float, float]] = dict()
        for param in self._parameter_names:
            if param.startswith("h") or param.startswith("C") or param.startswith("alpha") or param.startswith("gamma"):
                bounds[param] = (0, np.inf)
        return bounds

    @property
    def _num_E_params(self):
        """-"""
        return 2**self.N

    @property
    def _num_h_params(self):
        """-"""
        return self.N

    @property
    def _num_C_params(self):
        """-"""
        return self.N

    @property
    def _num_alpha_params(self):
        """-"""
        return 2 ** (self.N - 1) * self.N - self.N

    @property
    def _num_gamma_params(self):
        """-"""
        return 2 ** (self.N - 1) * self.N - self.N

    def _model_no_gamma(self, d, *args):
        """-"""
        loggammas = [0] * self._num_gamma_params
        return self._model(d, *args, *loggammas)

    def _model(self, d, *args):
        """-"""
        # `matrix` is the state transition matrix for the MuSyC model
        # matrix[i, :, :] is the state transition matrix at d[i]
        # That is to say, the matrix is handled completely numerically, rather than symbolically solving and then
        # plugging in doses.
        if len(d.shape) == 1:
            d = np.reshape(d, (-1, len(d)))
        matrix = np.zeros((d.shape[0], 2**self.N, 2**self.N))

        E_param_offset = 0
        h_param_offset = E_param_offset + self._num_E_params
        C_param_offset = h_param_offset + self._num_h_params
        alpha_param_offset = C_param_offset + self._num_C_params
        gamma_param_offset = alpha_param_offset + self._num_alpha_params

        E_params = args[E_param_offset:h_param_offset]
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

        add_drugs, remove_drugs = MuSyC._get_neighbors(0, self.N)
        # drugnum indicates which drug is added (like drug 0, drug 1, or drug 2 (0-based indexing))
        # jidx is the matrix column index of the state reached by adding that drug
        for drugnum, jidx in add_drugs:
            d_row = d[:, drugnum]
            h = h_params[drugnum]
            C = C_params[drugnum]
            r1r = self.r * np.power(C, h)

            # Transitions away from U due to dose d
            matrix[:, 0, 0] -= self.r * np.power(d_row, h)
            # Transitions into U from neighboring states
            matrix[:, 0, jidx] = r1r

        # Loop over all other states/rows (except the last one, since we know An = 1 - (U + A1 + A2 + ...))
        for idx in range(1, self._num_E_params - 1):
            add_drugs, remove_drugs = MuSyC._get_neighbors(idx, self.N)
            for drugnum, jidx in add_drugs:
                gamma = 1
                alpha = 1
                # If the neighbor is not 0, this is a synergy edge
                if jidx > 0:
                    edge_idx = self._edge_index[idx][jidx]
                    gamma = gamma_params[edge_idx]
                    alpha = alpha_params[edge_idx]
                d_row = d[:, drugnum]
                h = h_params[drugnum]
                C = C_params[drugnum]
                r1r = self.r * np.power(C, h)

                # This state gains from reverse transitions out of jidx
                matrix[:, idx, jidx] += r1r**gamma

                # This state loses from transitions toward jidx
                matrix[:, idx, idx] -= np.power(self.r * np.power(alpha * d_row, h), gamma)

            for drugnum, jidx in remove_drugs:
                gamma = 1
                alpha = 1
                # If the neighbor is not 0, this is a synergy edge
                if jidx > 0:
                    edge_idx = self._edge_index[jidx][idx]
                    gamma = gamma_params[edge_idx]
                    alpha = alpha_params[edge_idx]
                d_row = d[:, drugnum]
                h = h_params[drugnum]
                C = C_params[drugnum]
                r1r = self.r * np.power(C, h)

                # This state loses from reverse transitions toward jidx
                matrix[:, idx, idx] -= r1r**gamma

                # This state gaines from transitions from jidx
                matrix[:, idx, jidx] += np.power(self.r * np.power(alpha * d_row, h), gamma)

        # The final constraint is that U + A1 + A2 + ... = 1
        matrix[:, -1, :] = 1
        matrix_inv = np.linalg.inv(matrix)

        # M . [U A1 A2 ...]^T = [0 0 0 ... 1]^T
        # [U A1 A2 ...]^T = M^-1 . [0 0 0 ... 1]^T
        # [E0 E1 E2 ...] . [U A1 A2 ...] = E
        # All other rows should multiply to zero. Only the last row goes to 1.
        b = np.zeros(2**self.N)
        b[-1] = 1
        return np.dot(np.dot(matrix_inv, b), np.asarray(E_params))

    @staticmethod
    def _get_drug_string_from_state(state: list[int]) -> str:
        """Converts state (e.g., [1, 1, 0]) to drug-string (e.g., "2,3")

        State in this context indicates which drugs are present (1) vs absent (0). The 0th state index corresponds to
        the Nth drug. (e.g., [1, 0] -> "2", while [0, 1] -> "1")

        The drug-string is a list of the present drugs.
        """
        # If state is undrugged, return 0
        if 1 not in state:
            return "0"
        n = len(state)

        return ",".join(sorted([str(n - i) for i, val in enumerate(state) if val == 1]))

    @staticmethod
    def _get_drug_string_from_edge(state_a: list[int], state_b: list[int]) -> str:
        """Return a string representing the edge between two states

        The string is {start_state}_{added_drugs}. So for example, if drugs 2 and 3 are present in state_a, and
        drug 1 is added to get to state_b, the edge string would be "2,3_1".

        :param list[int] state_a: List of drugs in state_a (0 for absent, 1 for present)
        :param list[int] state_b: List of drugs in state_b
        :return str: String representing the edge between state_a and state_b
        """
        drugstr_a = MuSyC._get_drug_string_from_state(state_a)
        drugstr_b = MuSyC._get_drug_difference_string(state_a, state_b)
        return "_".join([drugstr_a, drugstr_b])

    @staticmethod
    def _get_drug_difference_string(state_a: list[int], state_b: list[int]) -> str:
        """Return the difference in drugs between two states.

        The drug difference string is a comma-separated list of drugs that are added. Removed drugs are ignored

        :param list[int] state_a: List of drugs in state_a (0 for absent, 1 for present)
        :param list[int] state_b: List of drugs in state_b
        :return str: Comma-separated list of drugs that are added
        """
        n = len(state_a)
        return ",".join(
            sorted([str(n - i) for i, drugab in enumerate(zip(state_a, state_b)) if drugab[1] == drugab[0] + 1])
        )

    @staticmethod
    def _get_beta(state, parameters):
        """Calculates synergistic efficacy, a synergy parameter derived from E parameters."""

        # beta is only defined for states associated with 2 or more drugs
        if state.count(1) < 2:
            return 0

        n = len(state)

        idx = MuSyC._state_to_idx(state)
        E = parameters[idx]

        # parents are states with one fewer drug active (e.g., parents of 011 are 001 and 010). beta is calculated by comparing E, the strongest of E_parents, and E0.
        E_parents = []
        for i in range(n):
            if state[i] == 1:
                parent = [j for j in state]
                parent[i] = 0
                pidx = MuSyC._state_to_idx(parent)
                E_parents.append(parameters[pidx])

        E0 = parameters[0]
        # TODO Add support for positive E orientation
        E_best_parent = np.amin(np.asarray(E_parents), axis=0)

        beta = (E_best_parent - E) / (E0 - E_best_parent)
        return beta

    def _default_single_drug_class(self):
        """-"""
        return Hill

    def _required_single_drug_class(self):
        """-"""
        return Hill

    def E_reference(self, d):
        """-"""
        if not self.is_specified:
            return ModelNotParameterizedError()
        parameters = self._transform_params_to_fit(self._get_parameters())
        alpha_offset = self._num_E_params + self._num_C_params + self._num_h_params
        parameters[alpha_offset:] = 1.0
        return self._model(d, *parameters)
