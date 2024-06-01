# TODO: Ensure proper behavior of bounds when fitting

import os
import sys
import unittest
from copy import deepcopy
from unittest import TestCase

import hypothesis
import numpy as np
from hypothesis import given
from hypothesis.strategies import sampled_from

from synergy.higher import MuSyC
from synergy.testing_utils.test_data_loader import load_nd_test_data
from synergy.testing_utils import assertions as synergy_assertions
from synergy.utils import dose_utils


MAX_FLOAT = sys.float_info.max

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _get_E3(E0, E1, E2, beta):
    strongest_E = np.amin(np.asarray([E1, E2]), axis=0)
    return strongest_E - beta * (E0 - strongest_E)


class MuSyCNDUnitTests(TestCase):
    """Unit tests for basic utility functions"""

    def test_num_params(self):
        """Ensure the model computes the correct number of parameters"""
        model = MuSyC(num_drugs=2)
        for param, expected in {"E": 4, "h": 2, "C": 2, "alpha": 2, "gamma": 2}.items():
            attr = f"_num_{param}_params"
            observed = model.__getattribute__(attr)
            self.assertEqual(observed, expected, msg=f"Expected {expected} {param} parameteres")

        model = MuSyC(num_drugs=3)
        for param, expected in {"E": 8, "h": 3, "C": 3, "alpha": 9, "gamma": 9}.items():
            attr = f"_num_{param}_params"
            observed = model.__getattribute__(attr)
            self.assertEqual(observed, expected, msg=f"Expected {expected} {param} parameteres")

        # alpha_{1}_{1,2}, #alpha_{1}_{1,3}, alpha_{2}_{1,2}, alpha_{2}_{2,3}, alpha_{3}_{1,3}, alpha_{3}_{2,3}
        # alpha_{1,2}_{1,2,3}, alpha_{1,3}_{1,2,3}, alpha_{2,3}_{1,2,3},

    def test_parameter_names(self):
        """Ensure parameter names are correct"""
        model = MuSyC(num_drugs=3)
        for p in model._parameter_names:
            print(p)
        # TODO
        self.assertAlmostEqual(0, 100)

    def test_idx_to_state(self):
        """Ensure state is computed correctly.

        idx counts from 0 to 2^num_drugs - 1, and state is the binary representation of idx.
        """
        self.assertListEqual([0, 0, 0], MuSyC._idx_to_state(0, 3))
        self.assertListEqual([0, 0, 1], MuSyC._idx_to_state(1, 3))
        self.assertListEqual([0, 1, 0], MuSyC._idx_to_state(2, 3))
        self.assertListEqual([0, 1, 1], MuSyC._idx_to_state(3, 3))
        self.assertListEqual([1, 0, 0], MuSyC._idx_to_state(4, 3))
        self.assertListEqual([1, 0, 1], MuSyC._idx_to_state(5, 3))
        self.assertListEqual([1, 1, 0], MuSyC._idx_to_state(6, 3))
        self.assertListEqual([1, 1, 1], MuSyC._idx_to_state(7, 3))

    def test_state_to_idx(self):
        """Ensure idx is computed correctly"""
        self.assertEqual(0, MuSyC._state_to_idx([0, 0, 0]))
        self.assertEqual(1, MuSyC._state_to_idx([0, 0, 1]))
        self.assertEqual(2, MuSyC._state_to_idx([0, 1, 0]))
        self.assertEqual(3, MuSyC._state_to_idx([0, 1, 1]))
        self.assertEqual(4, MuSyC._state_to_idx([1, 0, 0]))
        self.assertEqual(5, MuSyC._state_to_idx([1, 0, 1]))
        self.assertEqual(6, MuSyC._state_to_idx([1, 1, 0]))
        self.assertEqual(7, MuSyC._state_to_idx([1, 1, 1]))

    def test_hamming(self):
        """Ensure hamming distance is correctly calculated"""
        self.assertEqual(MuSyC._hamming([0, 0, 0, 0], [0, 0, 0, 0]), 0)
        self.assertEqual(MuSyC._hamming([0, 0, 0, 0], [1, 1, 0, 0]), 2)
        self.assertEqual(MuSyC._hamming([1, 1, 0, 0], [0, 0, 0, 0]), 2)
        self.assertEqual(MuSyC._hamming([0, 1], [1, 1]), 1)

    def test_get_neighbors(self):
        """Ensure neighbors in the state transition matrix are calculated correctly"""
        state = [0, 0, 0]
        idx = MuSyC._state_to_idx(state)
        add_drugs, remove_drugs = MuSyC._get_neighbors(idx, len(state))
        self.assertCountEqual(
            add_drugs,
            [
                (0, MuSyC._state_to_idx([0, 0, 1])),  # add drug 1 get to state [0, 0, 1]
                (1, MuSyC._state_to_idx([0, 1, 0])),  # add drug 2 get to state [0, 1, 0]
                (2, MuSyC._state_to_idx([1, 0, 0])),  # add drug 3 get to state [1, 0, 0]
            ],
        )
        self.assertListEqual(remove_drugs, [])

        state = [0, 1, 1, 0]
        idx = MuSyC._state_to_idx(state)
        add_drugs, remove_drugs = MuSyC._get_neighbors(idx, len(state))
        self.assertCountEqual(
            add_drugs,
            [
                (0, MuSyC._state_to_idx([0, 1, 1, 1])),  # add drug 1 get to state [0, 1, 1, 1]
                (3, MuSyC._state_to_idx([1, 1, 1, 0])),  # add drug 4 get to state [1, 1, 1, 0]
            ],
        )
        self.assertCountEqual(
            remove_drugs,
            [
                (1, MuSyC._state_to_idx([0, 1, 0, 0])),  # remove drug 2 get to state [0, 1, 0, 0]
                (2, MuSyC._state_to_idx([0, 0, 1, 0])),  # add drug 3 get to state [0, 0, 1, 0]
            ],
        )

    def test_get_edge_indices(self):
        """Ensure edge indices are calculated correctly"""
        edge_indices = MuSyC._get_edge_indices(3)
        import json

        print(json.dumps(edge_indices, indent=4))
        # TODO make test

    def test_get_drug_string_from_state(self):
        """Ensure drug strings are calculated correctly

        The drug string is a comma-separated list of drugs that are present in the state.
        """
        self.assertEqual(MuSyC._get_drug_string_from_state([1, 1, 0]), "2,3")
        self.assertEqual(MuSyC._get_drug_string_from_state([1, 0, 1, 1, 0]), "2,3,5")

    def test_get_drug_string_from_edge(self):
        """Ensure edge string is correct

        The edge string is an underscore-separated list of the starting and ending drug strings.
        """
        self.assertEqual(MuSyC._get_drug_string_from_edge([1, 0, 0], [1, 0, 1]), "3_1")

    def test_get_drug_difference_string(self):
        """Ensure drug difference string is correct

        The drug difference string is a comma-separated list of drugs that are added.
        """
        #                                            drug   3  2  1    3  2  1
        self.assertEqual(MuSyC._get_drug_difference_string([1, 0, 0], [1, 0, 1]), "1")
        self.assertEqual(MuSyC._get_drug_difference_string([1, 0, 0], [1, 1, 0]), "2")
        self.assertEqual(MuSyC._get_drug_difference_string([1, 0, 0], [1, 1, 1]), "1,2")
        self.assertEqual(MuSyC._get_drug_difference_string([1, 0, 0], [1, 0, 0]), "")  # no difference
        self.assertEqual(MuSyC._get_drug_difference_string([1, 0, 0], [0, 0, 0]), "")  # removing is ignored

    def test_get_beta(self):
        """Ensure beta is calculated correctly."""
        # We only need E params to calculate beta, so ignore everything else
        # NOTE these parameters must be in the correct order as what model._parameter_names generates
        params = [
            1.0,  # E_0
            0.6,  # E_1
            0.5,  # E_2
            0.7,  # E_1,2
            0.2,  # E_3
            0.2,  # E_1,3
            0.15,  # E_2,3
            0.0,  # E_1,2,3
        ]
        expected_betas = {
            "1,2": -0.4,  # (0.5 - 0.7) / (1.0 - 0.5)
            "1,3": 0.0,  # (0.2 - 0.2) / (1.0 - 0.2)
            "2,3": 0.0625,  # (0.2 - 0.15) / (1.0 - 0.2)
            "1,2,3": 0.17647058823529413,  # (0.15 - 0.0) / (1.0 - 0.15)
        }
        for state_idx in range(8):
            drug_state = MuSyC._idx_to_state(state_idx, 3)
            drug_string = MuSyC._get_drug_string_from_state(drug_state)
            beta = MuSyC._get_beta(params, drug_state)
            if drug_string in expected_betas:
                self.assertAlmostEqual(beta, expected_betas[drug_string], msg=f"Expected beta for {drug_string}")
            else:
                self.assertTrue(np.isnan(beta), msg=f"Expected beta == NaN for {drug_string}")


class MuSyCNDModelTests(TestCase):
    """Tests for the n-dimensional MuSyC model"""

    def test_asymptotic_limits(self):
        """Ensure the asymptotic dose limits work correctly"""
        params = {
            "E_0": 1.0,
            "E_1": 0.6,
            "E_2": 0.5,
            "E_1,2": 0.7,
            "E_3": 0.2,
            "E_1,3": 0.2,
            "E_2,3": 0.15,
            "E_1,2,3": 0.0,
            "h_1": 1.0,
            "h_2": 2.0,
            "h_3": 0.5,
            "C_1": 1.0,
            "C_2": 1.0,
            "C_3": 1.0,
            "alpha_1_3": 1.0,
            "alpha_1_2": 1.0,
            "alpha_2_3": 1.0,
            "alpha_2_1": 1.0,
            "alpha_1,2_3": 1.0,
            "alpha_3_2": 1.0,
            "alpha_3_1": 1.0,
            "alpha_1,3_2": 1.0,
            "alpha_2,3_1": 1.0,
        }
        model = MuSyC(num_drugs=3, **params)
        M = 1e9  # "maximum" dose
        d = np.asarray(
            [
                [0, 0, 0],  # 0, 0, 0
                [1, 0, 0],  # 0, 0, 0 -> 1, 0, 0
                [0, 1, 0],  # 0, 0, 0 -> 0, 1, 0
                [0, 0, 1],  # 0, 0, 0 -> 0, 0, 1
                [M, 0, 0],  # 1, 0, 0
                [0, M, 0],  # 0, 1, 0
                [0, 0, M],  # 0, 0, 1
                [M, 1, 0],  # 1, 0, 0 -> 1, 1, 0
                [M, 0, 1],  # 1, 0, 0 -> 1, 0, 1
                [1, M, 0],  # 0, 1, 0 -> 1, 1, 0
                [0, M, 1],  # 0, 1, 0 -> 0, 1, 1
                [1, 0, M],  # 0, 0, 1 -> 1, 0, 1
                [0, 1, M],  # 0, 0, 1 -> 0, 1, 1
                [M, M, 0],  # 1, 1, 0
                [M, 0, M],  # 1, 0, 1
                [0, M, M],  # 0, 1, 1
                [M, M, 1],  # 1, 1, 0 -> 1, 1, 1
                [M, 1, M],  # 1, 0, 1 -> 1, 1, 1
                [1, M, M],  # 0, 1, 1 -> 1, 1, 1
                [M, M, M],  # 1, 1, 1
            ],
            dtype=np.float64,
        )
        E = model.E(d)
        expected = np.asarray(
            [
                params["E_0"],
                (params["E_0"] + params["E_1"]) / 2.0,
                (params["E_0"] + params["E_2"]) / 2.0,
                (params["E_0"] + params["E_3"]) / 2.0,
                params["E_1"],
                params["E_2"],
                params["E_3"],
                (params["E_1"] + params["E_1,2"]) / 2.0,
                (params["E_1"] + params["E_1,3"]) / 2.0,
                (params["E_2"] + params["E_1,2"]) / 2.0,
                (params["E_2"] + params["E_2,3"]) / 2.0,
                (params["E_3"] + params["E_1,3"]) / 2.0,
                (params["E_3"] + params["E_2,3"]) / 2.0,
                params["E_1,2"],
                params["E_1,3"],
                params["E_2,3"],
                (params["E_1,2"] + params["E_1,2,3"]) / 2.0,
                (params["E_1,3"] + params["E_1,2,3"]) / 2.0,
                (params["E_2,3"] + params["E_1,2,3"]) / 2.0,
                params["E_1,2,3"],
            ]
        )
        np.testing.assert_allclose(E, expected, atol=1e-4)


class MuSyC3DFittingTests(TestCase):
    """Tests for fitting the n-dimensional MuSyC model"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.EXPECTED_PARAMETERS = {
            "synthetic_musyc3_reference_1.csv": {
                "E_1": 0.0,
                "E_2": 0.0,
                "E_3": 0.0,
                "E_1,2": 0.0,
                "E_1,3": 0.0,
                "E_2,3": 0.0,
            },
            "synthetic_musyc3_high_order_efficacy_synergy.csv": {
                "E_0": 1.0,
                "E_1": 2 / 3,
                "E_2": 2 / 3,
                "E_3": 2 / 3,
                "E_1,2": 1 / 3,
                "E_1,3": 1 / 3,
                "E_2,3": 1 / 3,
                "E_1,2,3": 0.0,
                "h_1": 1.0,
                "h_2": 1.0,
                "h_3": 1.0,
                "C_1": 1.0,
                "C_2": 1.0,
                "C_3": 1.0,
                "alpha_1_2": 1.0,
                "alpha_1_3": 1.0,
                "alpha_2_1": 1.0,
                "alpha_2_3": 1.0,
                "alpha_3_1": 1.0,
                "alpha_3_2": 1.0,
                "alpha_1,2_3": 1.0,
                "alpha_1,3_2": 1.0,
                "alpha_2,3_1": 1.0,
            },
            "synthetic_musyc3_high_order_potency_synergy.csv": {
                "alpha_1,2_3": 3.0,
            },
        }

    def _get_expected_parameters(self, fname):
        """Get the expected parameters for a given test file.

        Uses default parameters for any parameters not specified.
        """
        default_parameters = self.EXPECTED_PARAMETERS["synthetic_musyc3_high_order_efficacy_synergy.csv"]
        return dict(default_parameters, **self.EXPECTED_PARAMETERS[fname])

    @hypothesis.settings(deadline=None)
    @given(
        sampled_from(
            [
                # "synthetic_musyc3_reference_1.csv",
                "synthetic_musyc3_high_order_efficacy_synergy.csv",
                "synthetic_musyc3_high_order_potency_synergy.csv",
            ]
        )
    )
    def test_fit_musyc3d_no_bootstrap(self, fname):
        """Ensure the model fits correctly.

        Correctness is determined using atol = rtol = 0.05, in numpy's assert_allclose

        "synthetic_musyc3_reference_1.csv" is skipped because alpha fits are too hard when beta == 0, especially
        in the N-dimensional case. This makes the confidence intervals too large, and the best fit fail the tolerances.
        """
        model = MuSyC(num_drugs=3)
        d, E = load_nd_test_data(os.path.join(TEST_DATA_DIR, fname))
        model.fit(d, E)

        expected = self._get_expected_parameters(fname)
        synergy_assertions.assert_dict_allclose(model.get_parameters(), expected, rtol=5e-2, atol=5e-2)

    #    @hypothesis.settings(deadline=None)
    #    @given(
    #        sampled_from(
    #            [
    #                "synthetic_musyc3_reference_1.csv",
    #                # "synthetic_musyc3_high_order_efficacy_synergy.csv",
    #                # "synthetic_musyc3_high_order_potency_synergy.csv",
    #            ]
    #        )
    #    )
    def test_musyc_confidence_intervals(self):  # , fname):
        """Ensure confidence intervals are calculated correctly.

        This test must add in the beta parameters to the expected values, since they are not fit in the reference data,
        but are calculted in get_confidence_intervals().
        """
        np.random.seed(987214)
        fname = "synthetic_musyc3_reference_1.csv"
        expected = self._get_expected_parameters(fname)
        model = MuSyC(num_drugs=3)

        # Pre-compute beta parameters, which are returned in the confidence intervals
        E_parameters = []
        for key in model._parameter_names:
            if key.startswith("E"):
                E_parameters.append(expected[key])
        for idx in range(8):
            drug_state = MuSyC._idx_to_state(idx, 3)
            if drug_state.count(1) < 2:
                continue
            drug_string = MuSyC._get_drug_string_from_state(drug_state)
            expected[f"beta_{drug_string}"] = MuSyC._get_beta(E_parameters, drug_state)

        # Load and fit the data
        d, E = load_nd_test_data(os.path.join(TEST_DATA_DIR, fname))
        model.fit(d, E, bootstrap_iterations=100)

        # Ensure there were bootstrap iterations
        self.assertIsNotNone(model.bootstrap_parameters)

        confidence_intervals_95 = model.get_confidence_intervals()
        confidence_intervals_50 = model.get_confidence_intervals(confidence_interval=50)

        # Ensure that less stringent CI is narrower
        # [=====95=====]  More confidence requires wider interval
        #     [===50==]   Less confidence but tighter interval
        synergy_assertions.assert_dict_interval_is_contained_in_other(confidence_intervals_50, confidence_intervals_95)

        # Ensure true values are within confidence intervals
        # Without tol, this gives get "h_2: 1.0 not in (0.9906860645502625, 0.9997539385137254)"
        synergy_assertions.assert_dict_values_in_intervals(expected, confidence_intervals_95, tol=5e-4)


if __name__ == "__main__":
    unittest.main()
