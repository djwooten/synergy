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
from synergy.testing_utils.test_data_loader import load_test_data
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
        self.assertAlmostEqual(0, 100)

    def test_get_drug_string_from_state(self):
        """Ensure drug strings are calculated correctly"""
        self.assertEqual(MuSyC._get_drug_string_from_state([1, 1, 0]), "2,3")

    def test_get_drug_string_from_edge(self):
        """Ensure edge string is correct"""
        self.assertEqual(MuSyC._get_drug_string_from_edge([1, 0, 0], [1, 0, 1]), "3_1,3")


if __name__ == "__main__":
    unittest.main()
