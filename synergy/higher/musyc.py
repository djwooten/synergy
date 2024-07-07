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

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from synergy import utils
from synergy.exceptions import ModelNotParameterizedError
from synergy.higher.synergy_model_Nd import ParametricSynergyModelND
from synergy.single.dose_response_model_1d import DoseResponseModel1D
from synergy.single.hill import Hill
from synergy.utils.model_mixins import ParametricModelMixins


class MuSyC(ParametricSynergyModelND):
    """The MuSyC model for n-dimensional drug combinations.

    In MuSyC, synergy is parametrically defined as shifts in potency (alpha), efficacy (beta), or cooperativity (gamma).

    Two modes are supported:

    - fit_gamma=False (default) - fits only alpha and beta, with gamma fixed to 1.0
    - fit_gamma=True - fits all synergy parameters (alpha, beta, and gamma)

    .. csv-table:: Interpretation of synergy parameters
       :header: "Parameter", "Values", "Synergy/Antagonism", "Interpretation"

       "``alpha_a_b``", "[0, 1)", "Antagonistic Potency",       "Drug(s) ``a`` decrease the potency of drug ``b``"
       ,                "> 1",    "Synergistic Potency",        "Drug(s) ``a`` increase the potency of drug ``b``"
       "``beta_a``",    "< 0",    "Antagonistic Efficacy",      "Combination ``a`` is weaker than with one fewer drug"
       ,                "> 0",    "Synergistic Efficacy",       "Combination ``a`` is stronger than with one fewer drug"
       "``gamma_a_b``", "[0, 1)", "Antagonistic Cooperativity", "Drug(s) ``a`` decrease the cooperativity of drug ``b``"
       ,                "> 1",    "Synergistic Cooperativity",  "Drug(s) ``a`` increase the cooperativity of drug ``b``"
    """

    # Bounds will depened on the number of dimensions, so will be filled out in _get_initial_guess()
    def __init__(
        self,
        single_drug_models: Optional[Sequence[DoseResponseModel1D]] = None,
        num_drugs: int = -1,
        r_r=1.0,
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

        self.r_r = r_r

        if not self.fit_gamma:
            self.fit_function = self._model_no_gamma
        else:
            self.fit_function = self._model

    def _transform_params_to_fit(self, params):
        """Transform linear parameters to log-scale for fitting.

        Params come in order E, h, C, alpha, gamma. So everything past E should be log-scaled.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
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

    def _get_default_single_drug_kwargs(self, drug_idx: int) -> Dict[str, Any]:
        """Default keyword arguments for single drug models.

        This is used for each single drug unless an already instantiated version is provided in __init__().

        1) Translate bounds for the entire model into bounds for the specified single-drug model
        """
        linear_lower_bounds = self._transform_params_from_fit(self._bounds[0])
        linear_upper_bounds = self._transform_params_from_fit(self._bounds[1])
        parameter_bounds = list(
            zip(linear_lower_bounds, linear_upper_bounds)
        )  # convert [(lb, lb, ...), (ub, ub, ...)] to [(lb, ub), (lb, ub), ...]
        return {
            "E0_bounds": parameter_bounds[0],
            "Emax_bounds": parameter_bounds[self._parameter_names.index(f"E_{drug_idx + 1}")],
            "h_bounds": parameter_bounds[self._parameter_names.index(f"h_{drug_idx + 1}")],
            "C_bounds": parameter_bounds[self._parameter_names.index(f"C_{drug_idx + 1}")],
        }

    def _get_initial_guess(self, d, E, p0):
        """Get the initial guess.

        MuSyC will look at doses around the minimum, maximum, and median doses for each drug and use those to make
        guesses for E and C parameters. h, alpha, and gamma will be set to 1.0.
        """
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
        utils.sanitize_initial_guess(p0, self._bounds)
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

        MuSyC models states for cells affected by each drug in all possible combinations. For instance, given three
        drugs, MuSyC will have states 000, 001, 010, 011, 100, 101, 110, 111. 110 means affected by drugs 3 and 2,
        but unaffected by drug 1.

        Parameters
        ----------
        idx : int
            Index for the state, ranging from 0 to 2**n-1.

        n : int
            Number of drugs in combination

        Returns
        -------
        state : list of int
            A list indicating which drugs are "active". [1,1,0] means drug 1 is not active (0), and drugs 2 and 3 are
            active.
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
    def _get_edge_indices(n: int) -> Dict[int, Dict[int, int]]:
        """Return a map of start state, end state, to index of associated alpha (gamma) parameter

        When evaluating the model, we need to know which alpha (gamma) parameter to use for a given edge. This method
        calculates those indices.

        For example, with 3 drugs, there are 9 synergy edges. So alpha_params will have 9 elements. The edge_index from
        this map will indicate which parameter to use.

        :param int n: Number of drugs
        :return Dict[int, Dict[int, int]]: Map of start state, end state, to index of associated alpha (gamma) parameter
        """
        edge_index: Dict[int, Dict[int, int]] = dict()
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
    def _parameter_names(self) -> List[str]:
        param_names = []
        gamma_names = []
        for i in range(self._num_E_params):
            drug_string = MuSyC._get_drug_string_from_state(MuSyC._idx_to_state(i, self.N))
            param_names.append(f"E_{drug_string}")
        for i in range(self._num_h_params):
            param_names.append(f"h_{i + 1}")
        for i in range(self._num_C_params):
            param_names.append(f"C_{i + 1}")
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
    def _default_fit_bounds(self) -> Dict[str, Tuple[float, float]]:
        return {
            param: (0, np.inf)
            for param in self._parameter_names
            if param.startswith("h") or param.startswith("C") or param.startswith("alpha") or param.startswith("gamma")
        }

    @property
    def _num_E_params(self):
        """One per drug state."""
        return 2**self.N

    @property
    def _num_h_params(self):
        """One per drug."""
        return self.N

    @property
    def _num_C_params(self):
        """One per drug."""
        return self.N

    @property
    def _num_alpha_params(self):
        """One per edge from a drugged state to another drugged state."""
        return 2 ** (self.N - 1) * self.N - self.N

    @property
    def _num_gamma_params(self):
        """One per edge from a drugged state to another drugged state."""
        return 2 ** (self.N - 1) * self.N - self.N

    def _model_no_gamma(self, d, *args):
        """MuSyC model assuming gamma == 1."""
        loggammas = [0] * self._num_gamma_params
        return self._model(d, *args, *loggammas)

    def _model(self, d, *args):
        """MuSyC model.

        This creates a transition matrix for the MuSyC model and then solves for the equilibrium state by inverting it.
        """
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
            # r1r = self.r * np.float_power(C, h)
            r = self.r_r / np.float_power(C, h)

            # Transitions away from U due to dose d
            matrix[:, 0, 0] -= r * np.float_power(d_row, h)
            # Transitions into U from neighboring states
            matrix[:, 0, jidx] = self.r_r

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
                r = self.r_r / np.float_power(C, h)

                # This state gains from reverse transitions out of jidx
                matrix[:, idx, jidx] += self.r_r**gamma

                # This state loses from transitions toward jidx
                matrix[:, idx, idx] -= np.float_power(r * np.float_power(alpha * d_row, h), gamma)

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
                r = self.r_r / np.float_power(C, h)

                # This state loses from reverse transitions toward jidx
                matrix[:, idx, idx] -= self.r_r**gamma

                # This state gaines from transitions from jidx
                matrix[:, idx, jidx] += np.float_power(r * np.float_power(alpha * d_row, h), gamma)

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
    def _get_drug_string_from_state(state: Sequence[int]) -> str:
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
    def _get_drug_string_from_edge(state_a: Sequence[int], state_b: Sequence[int]) -> str:
        """Return a string representing the edge between two states

        The string is {start_state}_{added_drugs}. So for example, if drugs 2 and 3 are present in state_a, and
        drug 1 is added to get to state_b, the edge string would be "2,3_1".

        :param Sequence[int] state_a: List of drugs in state_a (0 for absent, 1 for present)
        :param Sequence[int] state_b: List of drugs in state_b
        :return str: String representing the edge between state_a and state_b
        """
        drugstr_a = MuSyC._get_drug_string_from_state(state_a)
        drugstr_b = MuSyC._get_drug_difference_string(state_a, state_b)
        return "_".join([drugstr_a, drugstr_b])

    @staticmethod
    def _get_drug_difference_string(state_a: Sequence[int], state_b: Sequence[int]) -> str:
        """Return the difference in drugs between two states.

        The drug difference string is a comma-separated list of drugs that are added. Removed drugs are ignored

        :param Sequence[int] state_a: List of drugs in state_a (0 for absent, 1 for present)
        :param Sequence[int] state_b: List of drugs in state_b
        :return str: Comma-separated list of drugs that are added
        """
        n = len(state_a)
        return ",".join(
            sorted([str(n - i) for i, drugab in enumerate(zip(state_a, state_b)) if drugab[1] == drugab[0] + 1])
        )

    @property
    def beta(self) -> Dict[str, float]:
        """Synergistic efficacy, a synergy parameter derived from E parameters.

        :return Dict[str, float]: A map of which drugs are present in each state to the beta value for that state.
        """
        if not self.is_specified:
            raise ModelNotParameterizedError("Cannot calculate beta if model is not specified.")

        parameters = [self.get_parameters()[param] for param in self._parameter_names]
        state_count = 2**self.N
        beta = {}
        for i in range(state_count):
            state = MuSyC._idx_to_state(i, self.N)
            drug_string = MuSyC._get_drug_string_from_state(state)
            value = MuSyC._get_beta(parameters, state)
            if not np.isnan(value):
                beta[f"beta_{drug_string}"] = value
        return beta

    @staticmethod
    def _get_beta(parameters, state):
        """Calculates synergistic efficacy, a synergy parameter derived from E parameters.

        beta is defined for states associated with 2 or more present drugs.
        The state's parents are states with one fewer drug active (e.g., parents of 011 are 001 and 010).
        Examples:
            The parents of [0, 1, 1] are [0, 0, 1] and [0, 1, 0].
            The parents of [1, 1, 1] are [0, 1, 1], [1, 0, 1], and [1, 1, 0]
        beta is calculated by comparing E, the strongest of E_parents, and E0.

        `beta = (E_best_parent - E) / (E0 - E_best_parent)`

        If beta > 0, it indicates the combination of all of the drugs together is more efficacious than the
        than any of the parent combinations.

        If beta < 0, it indicates the combination is less efficatious than the parent combinations.
        """
        if state.count(1) < 2:  # beta is only defined for states associated with 2 or more drugs
            return np.nan

        n = len(state)

        idx = MuSyC._state_to_idx(state)
        E = parameters[idx]

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

    @property
    def _default_single_drug_class(self):
        return Hill

    @property
    def _required_single_drug_class(self):
        return Hill

    def E_reference(self, d):
        if not self.is_specified and (
            not self.single_drug_models or not all([model.is_specified for model in self.single_drug_models])
        ):
            return ModelNotParameterizedError()

        parameters: Dict[str, float] = {}
        strongest_E = np.inf
        E0 = 0

        # If the model is not specified yet, get single-drug parameters from single-drug models
        if not self.is_specified:
            for i, model in enumerate(self.single_drug_models):
                E0 += model.E0 / self.N
                Emax = model.Emax
                strongest_E = min(strongest_E, Emax)  # TODO: Add support for positive E orientation
                parameters[f"E_{i + 1}"] = Emax
                parameters[f"h_{i + 1}"] = model.h
                parameters[f"C_{i + 1}"] = model.C
        # Otherwise get single-drug parameters directly from the model
        else:
            for i in range(self.N):
                E0 += self.E_0 / self.N
                Emax = self.__getattribute__(f"E_{i + 1}")
                strongest_E = min(strongest_E, Emax)
                parameters[f"E_{i + 1}"] = Emax
                parameters[f"h_{i + 1}"] = self.__getattribute__(f"h_{i + 1}")
                parameters[f"C_{i + 1}"] = self.__getattribute__(f"C_{i + 1}")
        parameters["E_0"] = E0

        # Default all other parameters to no-synergy values
        parameters_list = []
        for param_key in self._parameter_names:
            if param_key in parameters:  # single-drug parameters
                parameters_list.append(parameters[param_key])
            elif param_key.startswith("E"):  # E_{i,j,...} combinations
                parameters_list.append(strongest_E)
            else:  # alpha or gamma
                parameters_list.append(1.0)

        return self._model(d, *self._transform_params_to_fit(parameters_list))

    def get_confidence_intervals(self, confidence_interval: float = 95) -> Dict[str, Tuple[float, float]]:
        """Returns the lower bound and upper bound estimate for each parameter.

        This also calculates confidence intervals for beta, which is derived from the E parameters.

        Parameters
        ----------
        confidence_interval : float, default=95
            % confidence interval to return. Must be between 0 and 100.

        Return
        ------
        Dict[str, Tuple[float, float]]
            A dictionary of parameter names to a tuple of the lower and upper bounds of the confidence interval.
        """
        ci = super().get_confidence_intervals(confidence_interval=confidence_interval)

        if self.bootstrap_parameters is None:
            return ci

        lb = (100 - confidence_interval) / 2.0
        ub = 100 - lb

        for i in range(self._num_E_params):
            state = MuSyC._idx_to_state(i, self.N)
            if state.count(1) < 2:  # beta is only defined for states associated with 2 or more drugs
                continue
            bootstrap_beta = MuSyC._get_beta(self.bootstrap_parameters.transpose(), state)
            drug_string = MuSyC._get_drug_string_from_state(state)
            ci[f"beta_{drug_string}"] = np.percentile(bootstrap_beta, [lb, ub])
        return ci

    def summarize(self, confidence_interval: float = 95, tol: float = 0.01):
        pars = self.get_parameters()

        header = ["Parameter", "Value", "Comparison", "Synergy"]
        ci: Dict[str, Tuple[float, float]] = {}
        if self.bootstrap_parameters is not None:
            ci = self.get_confidence_intervals(confidence_interval=confidence_interval)
            header.insert(2, f"{confidence_interval:0.3g}% CI")

        rows = [header]

        # beta
        for idx in range(self._num_E_params):
            state = MuSyC._idx_to_state(idx, self.N)
            if state.count(1) < 2:
                continue
            drug_string = MuSyC._get_drug_string_from_state(state)
            beta = MuSyC._get_beta(list(pars.values()), state)
            rows.append(
                ParametricModelMixins.make_summary_row(
                    f"beta_{drug_string}", 0, beta, ci, tol, False, "synergistic", "antagonistic"
                )
            )

        # alpha and gamma
        for key in pars.keys():
            if "alpha" in key or "gamma" in key:
                rows.append(
                    ParametricModelMixins.make_summary_row(
                        key, 1, pars[key], ci, tol, True, "synergistic", "antagonistic"
                    )
                )

        print(utils.format_table(rows))

    def __repr__(self):
        if self.is_specified:
            parameters = self.get_parameters()
            parameters.update(self.beta)
            param_vals = ", ".join([f"{param}={val:0.3g}" for param, val in parameters.items()])  # typing: ignore
        else:
            param_vals = ""
        return f"MuSyC({param_vals})"
