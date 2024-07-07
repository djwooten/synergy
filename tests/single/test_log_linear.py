import os
import unittest
from unittest import TestCase

import numpy as np

from synergy.single import LogLinear

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class TestLogLinear(TestCase):
    """Tests for the log linear model."""

    def test_is_specified(self):
        """Ensure that is_specified is false until model.fit is called"""
        model = LogLinear()

        # Ensure the model is not yet marked as specified
        self.assertFalse(model.is_specified)

        d = np.asarray([1, 10])
        E = np.asarray([0, 1])
        model.fit(d, E)

        # Ensure the model is now specified
        self.assertTrue(model.is_specified)

    def test_log_linearization(self):
        """Ensure the model correctly computes log-linearization."""
        model = LogLinear()
        #   |         X
        #   |
        #   |
        #   |
        #   |     X
        #   +-X------------
        #     1   10  100
        d = np.asarray([1, 10, 100])
        E = np.asarray([0, 1, 5])
        model.fit(d, E)

        segment_1_d = np.logspace(0, 1)
        segment_1_E = np.linspace(0, 1)

        segment_2_d = np.logspace(1, 2)
        segment_2_E = np.linspace(1, 5)

        test_d = np.hstack([segment_1_d, segment_2_d])
        expected_E = np.hstack([segment_1_E, segment_2_E])
        observed_E = model.E(test_d)
        np.testing.assert_allclose(observed_E, expected_E)

    def test_nans_outside_of_bounds(self):
        """Ensure E(d) is NaN at doses above and below the trained range."""
        model = LogLinear()
        d = np.asarray([1, 10])
        E = np.asarray([0, 10])

        model.fit(d, E)
        test_d = np.asarray([0.9, 10.1])
        obs_E = model.E(test_d)

        self.assertTrue(np.isnan(obs_E).all(), msg=f"LogLinear should be NaN outside of trained range ({obs_E})")

    def test_inverse_monotonic(self):
        """Ensure monotonic response curves are invertible in the entire range."""
        model = LogLinear()
        #   |         X
        #   |
        #   |
        #   |
        #   |     X
        #   +-X------------
        #     1   10  100
        d = np.asarray([1, 10, 100])
        E = np.asarray([0, 1, 5])
        model.fit(d, E)

        segment_1_d = np.logspace(0, 1)
        segment_1_E = np.linspace(0, 1)

        segment_2_d = np.logspace(1, 2)
        segment_2_E = np.linspace(1, 5)

        expected_d = np.hstack([segment_1_d, segment_2_d])
        test_E = np.hstack([segment_1_E, segment_2_E])
        observed_d = model.E_inv(test_E)
        np.testing.assert_allclose(observed_d, expected_d)

    def test_inverse_nans_outside_of_bounds(self):
        """Ensure E_inv(E) is NaN at E above and below the trained range."""
        model = LogLinear()
        d = np.asarray([1, 10, 100])
        E = np.asarray([0, 1, 5])
        model.fit(d, E)

        test_E = np.hstack([-0.1, 5.1])
        obs_d = model.E_inv(test_E)
        self.assertTrue(np.isnan(obs_d).all(), msg=f"LogLinear should be NaN outside of trained range ({obs_d})")

    def test_inverse_linearize_nans_in_middle(self):
        """Ensure non-invertible regions in a response curve are smoothed via an invertible linear approximation."""
        #    |---||----||---|
        #   |               X
        # _ |             X
        #   |       X   X
        # _ |     X   X
        #   |   X
        #   +-X------------------------
        #     0 1 2 3 4 5 6 7
        d = np.logspace(0, 7, 8)
        E = np.asarray([0, 1, 2, 3, 2, 3, 4, 5])
        model = LogLinear()
        model.fit(d, E)

        segment_1_d = np.logspace(0, 2)
        segment_1_E = np.linspace(0, 2)

        # Details of the middle region are lost.
        # Instead it linearly interpolates from the left side of the middle region to the right side.
        segment_2_d = np.logspace(2, 5)
        segment_2_E = np.linspace(2, 3)

        segment_3_d = np.logspace(5, 7)
        segment_3_E = np.linspace(3, 5)

        expected_d = np.hstack([segment_1_d, segment_2_d, segment_3_d])
        test_E = np.hstack([segment_1_E, segment_2_E, segment_3_E])
        observed_d = model.E_inv(test_E)
        np.testing.assert_allclose(observed_d, expected_d)

    def test_inverse_linearize_nans_on_edges(self):
        """Ensure non-invertible regions at the boundary use their inner edges for inverses.

        TODO: It seems like there could be a more clever solution, like linearizing from the boundary to the median?
        """
        #    |---||----||---|
        #   |             X
        # _ |           X   X
        #   |         X
        # _ |       X
        #   | X   X
        #   +---X----------------------
        #     0 1 2 3 4 5 6 7
        d = np.logspace(0, 7, 8)
        E = np.asarray([1, 0, 1, 2, 3, 4, 5, 4])
        model = LogLinear()
        model.fit(d, E)

        # This segment is on the left edge. We only use the right boundary for an inverse (d=10^2)
        segment_1_d = np.ones(50) * 100.0
        segment_1_E = np.linspace(0, 1)

        segment_2_d = np.logspace(2, 5)
        segment_2_E = np.linspace(1, 4)

        # This segment is on the right edge. We only use the left boundary for an inverse (d=10^5)
        segment_3_d = np.ones(50) * 100000.0
        segment_3_E = np.linspace(4, 5)

        expected_d = np.hstack([segment_1_d, segment_2_d, segment_3_d])
        test_E = np.hstack([segment_1_E, segment_2_E, segment_3_E])
        observed_d = model.E_inv(test_E)
        np.testing.assert_allclose(observed_d, expected_d)

    def test_inverse_linearize_with_joined_intervals(self):
        """Ensure  non-invertible regions that overlap are linearized like a single non-invertible region"""
        #    |--||---||--|
        # _ |           X
        #   |       X
        #   |   X
        #   |         X
        # _ |     X
        #   +-X------------
        #     0 1 2 3 4 5
        d = np.logspace(0, 5, 6)
        E = np.asarray([0, 3, 1, 4, 2, 5])
        # E intervals (1, 3) and (2, 4) should get merged to (1, 4)
        model = LogLinear()
        model.fit(d, E)

        # Use this to find logd at the boundary between non-invertible and invertible regions
        # It just uses y - y0 = m * (x - x0), solve for x (which is logd)
        def get_boundary_logd(logd0, logd1, E0, E1, E):
            return (logd1 - logd0) * (E - E0) / (E1 - E0) + logd0

        logd_boundary_left = get_boundary_logd(0, 1, 0, 3, 1)  # (0, 0) to (1, 3), solve y=1
        logd_boundary_right = get_boundary_logd(4, 5, 2, 5, 4)  # (4, 2) to (5, 5), solve y=4

        segment_1_d = np.logspace(0, logd_boundary_left)
        segment_1_E = np.linspace(0, 1)

        # Exclude the boundaries, because they are invertible
        segment_2_d = np.logspace(logd_boundary_left, logd_boundary_right)[1:-1]
        segment_2_E = np.linspace(1, 4)[1:-1]

        segment_3_d = np.logspace(logd_boundary_right, 5)
        segment_3_E = np.linspace(4, 5)

        expected_d = np.hstack([segment_1_d, segment_2_d, segment_3_d])
        test_E = np.hstack([segment_1_E, segment_2_E, segment_3_E])
        observed_d = model.E_inv(test_E)
        np.testing.assert_allclose(observed_d, expected_d)

    def test_inverse_with_nan_inverses(self):
        """Ensure that when nan_inverses==True, non-invertible regions return NaN for their inverse"""
        #    |---||----||---|
        #   |             X
        # _ |           X   X
        #   |         X
        # _ |       X
        #   | X   X
        #   +---X----------------------
        #     0 1 2 3 4 5 6 7
        d = np.logspace(0, 7, 8)
        E = np.asarray([1, 0, 1, 2, 3, 4, 5, 4])
        model = LogLinear(nan_inverses=True)
        model.fit(d, E)

        # This segment is non-invertible.
        # Exclude the boundary points (E=0 and E=1) because boundaries ARE invertible
        segment_1_d = np.ones(48) * np.nan
        segment_1_E = np.linspace(0, 1)[1:-1]

        # This segment can be inverted.
        segment_2_d = np.logspace(2, 5)
        segment_2_E = np.linspace(1, 4)

        # This segment is non-invertible.
        # Exclude the boundary points (E=4 and E=5) because boundaries ARE invertible
        segment_3_d = np.ones(48) * np.nan
        segment_3_E = np.linspace(4, 5)[1:-1]

        #                        NaN           Values      NaN
        expected_d = np.hstack([segment_1_d, segment_2_d, segment_3_d])
        test_E = np.hstack([segment_1_E, segment_2_E, segment_3_E])
        observed_d = model.E_inv(test_E)

        np.testing.assert_allclose(observed_d, expected_d)

    def test_dose_scale(self):
        """Ensure the model behaves well at high dose ranges."""
        model = LogLinear()
        #   |         X
        #   |
        #   |
        #   |
        #   |     X
        #   +-X------------
        #     1   10  100
        scale = 1e9
        d = np.asarray([1, 10, 100]) * scale
        E = np.asarray([0, 1, 5])
        model.fit(d, E)

        segment_1_d = np.logspace(0, 1) * scale
        segment_1_E = np.linspace(0, 1)

        segment_2_d = np.logspace(1, 2) * scale
        segment_2_E = np.linspace(1, 5)

        test_d = np.hstack([segment_1_d, segment_2_d])
        expected_E = np.hstack([segment_1_E, segment_2_E])
        observed_E = model.E(test_d)
        np.testing.assert_allclose(observed_E, expected_E)

        self.assertAlmostEqual(model._dose_scale, scale * 10, places=1)


if __name__ == "__main__":
    unittest.main()
