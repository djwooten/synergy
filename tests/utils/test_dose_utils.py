import numpy as np
import pytest
from numpy import testing as npt

from synergy.testing_utils import unique_tol
from synergy.utils import dose_utils


class TestRemoveZeros:
    """Tests for the remove_zeros function."""

    def test_remove_zeros_no_zero(self):
        """Ensure no changes if the array has no zeros."""
        data = np.asarray([1, 2, 3])
        result = dose_utils.remove_zeros(data)
        assert (result == data).all()

    def test_remove_zeros_dilutions(self):
        """Ensure zeros are replaced based on dilution."""
        data = np.asarray([0, 1, 10])
        result_1 = dose_utils.remove_zeros(data)
        result_2 = dose_utils.remove_zeros(data, num_dilutions=2)
        npt.assert_allclose(result_1, np.asarray([0.1, 1, 10]))
        npt.assert_allclose(result_2, np.asarray([0.01, 1, 10]))

    def test_remove_zeros_min_buffer(self):
        """Ensure zeros are replaced based on min_buffer."""
        data = np.asarray([0, 1, 1.001, 2])
        result = dose_utils.remove_zeros(data, min_buffer=1)

        # when resorting to min_buffer, the min and max doses are used to get the dilution
        data_expected = np.asarray([1.0 / 2.0, 1, 1.001, 2])
        npt.assert_allclose(result, data_expected)


class TestDoseGrids:
    """Tests for dose grid creation tools."""

    def test_dose_grid(self):
        """Ensure dose grid is created correctly."""
        d1min = 1
        d1max = 10
        d2min = 2
        d2max = 20
        n_points1 = 3
        n_points2 = 4

        d1, d2 = dose_utils.make_dose_grid(d1min, d1max, d2min, d2max, n_points1, n_points2)

        # check shape
        assert len(d1) == n_points1 * n_points2
        assert len(d2) == n_points1 * n_points2

        # check counts
        assert len(np.unique(d1)) == n_points1
        assert len(np.unique(d2)) == n_points2

        # check for duplicates (each d1 dose should show up n_points2 times, and vice versa)
        assert np.unique(np.unique(d1, return_counts=True)[1]) == np.asarray([n_points2])
        assert np.unique(np.unique(d2, return_counts=True)[1]) == np.asarray([n_points1])

        # check min/max
        assert pytest.approx(np.min(d1)) == d1min
        assert pytest.approx(np.max(d1)) == d1max
        assert pytest.approx(np.min(d2)) == d2min
        assert pytest.approx(np.max(d2)) == d2max

    def test_dose_grid_include_zero(self):
        """Ensure 0 is included correctly."""
        d1min = 1
        d1max = 10
        d2min = 2
        d2max = 20
        n_points1 = 3
        n_points2 = 4

        d1, d2 = dose_utils.make_dose_grid(d1min, d1max, d2min, d2max, n_points1, n_points2, include_zero=True)

        # check shape
        assert len(d1) == n_points1 * n_points2
        assert len(d2) == n_points1 * n_points2

        # check counts
        assert len(np.unique(d1)) == n_points1
        assert len(np.unique(d2)) == n_points2

        # check min/max (dmin should still show up, even though the true min is 0)
        assert pytest.approx(np.min(d1[d1 > 0])) == d1min
        assert pytest.approx(np.max(d1)) == d1max
        assert pytest.approx(np.min(d2[d2 > 0])) == d2min
        assert pytest.approx(np.max(d2)) == d2max

        # check zero
        assert pytest.approx(np.min(d1)) == 0
        assert pytest.approx(np.min(d2)) == 0

    def test_dose_grid_linear(self):
        """Ensure linear dose grid is created correctly."""
        d1min = 1
        d1max = 10
        d2min = 2
        d2max = 20
        n_points1 = 4
        n_points2 = 5

        d1, d2 = dose_utils.make_dose_grid(d1min, d1max, d2min, d2max, n_points1, n_points2, logscale=False)

        # check shape
        assert len(d1) == n_points1 * n_points2
        assert len(d2) == n_points1 * n_points2

        # check counts
        assert len(np.unique(d1)) == n_points1
        assert len(np.unique(d2)) == n_points2

        # check for duplicates (each d1 dose should show up n_points2 times, and vice versa)
        assert np.unique(np.unique(d1, return_counts=True)[1]) == np.asarray([n_points2])
        assert np.unique(np.unique(d2, return_counts=True)[1]) == np.asarray([n_points1])

        # check min/max
        assert pytest.approx(np.min(d1)) == d1min
        assert pytest.approx(np.max(d1)) == d1max
        assert pytest.approx(np.min(d2)) == d2min
        assert pytest.approx(np.max(d2)) == d2max

        # ensure linear scale (differences between doses should be constant)
        assert len(unique_tol(np.diff(np.unique(d1)))) == 1
        assert len(unique_tol(np.diff(np.unique(d2)))) == 1

    def test_dose_grid_multi(self):
        """Ensure multi-dose grid is created correctly."""
        dmin = [1, 2, 3]
        dmax = [10, 20, 30]
        n_points = [3, 4, 5]

        doses = dose_utils.make_dose_grid_multi(dmin, dmax, n_points)

        # check shape
        assert doses.shape[0] == np.prod(n_points)
        assert doses.shape[1] == len(dmin)

        # Check each drug
        for d, dmin_, dmax_, n_points_ in zip(doses.transpose(), dmin, dmax, n_points):
            assert len(np.unique(d)) == n_points_
            assert pytest.approx(np.min(d)) == dmin_
            assert pytest.approx(np.max(d)) == dmax_

    def test_dose_grid_multi_include_zero(self):
        """Ensure 0 is included correctly in multi-dose grid."""
        dmin = [1, 2, 3]
        dmax = [10, 20, 30]
        n_points = [3, 4, 5]

        doses = dose_utils.make_dose_grid_multi(dmin, dmax, n_points, include_zero=True)

        # check shape
        assert doses.shape[0] == np.prod(n_points)
        assert doses.shape[1] == len(dmin)

        # Check each drug
        for d, dmin_, dmax_, n_points_ in zip(doses.transpose(), dmin, dmax, n_points):
            assert len(np.unique(d)) == n_points_
            assert pytest.approx(np.min(d[d > 0])) == dmin_
            assert pytest.approx(np.max(d)) == dmax_

            # check zero
            assert pytest.approx(np.min(doses)) == 0

    def test_dose_grid_multi_linear(self):
        """Ensure linear multi-dose grid is created correctly."""
        dmin = [1, 2, 3]
        dmax = [10, 20, 30]
        n_points = [4, 5, 6]

        doses = dose_utils.make_dose_grid_multi(dmin, dmax, n_points, logscale=False)

        # check shape
        assert doses.shape[0] == np.prod(n_points)
        assert doses.shape[1] == len(dmin)

        # Check each drug
        for d, dmin_, dmax_, n_points_ in zip(doses.transpose(), dmin, dmax, n_points):
            assert len(np.unique(d)) == n_points_
            assert pytest.approx(np.min(d)) == dmin_
            assert pytest.approx(np.max(d)) == dmax_

            # ensure linear scale (differences between doses should be constant)
            assert len(unique_tol(np.diff(np.unique(d)))) == 1


class TestMonotherapyUtils:
    """Tests for monotherapy utility functions."""

    def test_is_monotherapy_ND(self):
        """Ensure monotherapy is detected correctly."""
        d = np.asarray([0, 0, 0, 0])
        assert dose_utils.is_monotherapy_ND(d)

        d = np.asarray([0, 0, 0, 1])
        assert dose_utils.is_monotherapy_ND(d)

        d = np.asarray([0, 0, 1, 1])
        assert not dose_utils.is_monotherapy_ND(d)

    def test_get_monotherapy_mask_ND(self):
        """Ensure monotherapy mask is computed correctly."""
        d = np.asarray(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 2, 0],
                [3, 0, 0],
                [1, 1, 1],
                [1, 1, 0],
                [0, 1, 1],
            ]
        )
        mask = dose_utils.get_monotherapy_mask_ND(d)
        assert (mask[0] == np.asarray([0, 1, 2, 3])).all()

    def test_get_drug_alone_mask_ND(self):
        """Ensure the single drug's monotherpay doses are extracted correctly."""
        d = np.asarray(
            [
                [0, 0, 0],
                [0, 0, 1],  # drug_idx 2
                [0, 2, 0],  # drug_idx 1
                [3, 0, 0],  # drug_idx 0
                [1, 1, 1],
                [1, 1, 0],
                [0, 1, 1],
            ]
        )
        mask = dose_utils.get_drug_alone_mask_ND(d, 0)
        assert (mask[0] == np.asarray([0, 3])).all()

        mask = dose_utils.get_drug_alone_mask_ND(d, 1)
        assert (mask[0] == np.asarray([0, 2])).all()

        mask = dose_utils.get_drug_alone_mask_ND(d, 2)
        assert (mask[0] == np.asarray([0, 1])).all()


def test_is_on_grid():
    """Test the is_on_grid function."""
    d = np.asarray(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 2],
            [0, 0, 3],
            [0, 1, 0],
            [0, 1, 1],
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 0],
            [0, 2, 1],
            [0, 2, 2],
            [0, 2, 3],
            [1, 0, 0],
            [1, 0, 1],
            [1, 0, 2],
            [1, 0, 3],
            [1, 1, 0],
            [1, 1, 1],
            [1, 1, 2],
            [1, 1, 3],
            [1, 2, 0],
            [1, 2, 1],
            [1, 2, 2],
            [1, 2, 3],
        ]
    )
    assert dose_utils.is_on_grid(d)

    d = np.asarray(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 2],
            [0, 0, 3],
            [0, 1, 0],
            [0, 1, 1],
            # [0, 1, 2],
            [0, 1, 3],
            [0, 2, 0],
            [0, 2, 1],
            [0, 2, 2],
            [0, 2, 3],
            [1, 0, 0],
            [1, 0, 1],
            [1, 0, 2],
            [1, 0, 3],
            [1, 1, 0],
            [1, 1, 1],
            [1, 1, 2],
            [1, 1, 3],
            [1, 2, 0],
            [1, 2, 1],
            [1, 2, 2],
            [1, 2, 3],
        ]
    )
    assert not dose_utils.is_on_grid(d)

    d = np.asarray(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 2],
            [0, 0, 3],
            [0, 1, 0],
            [0, 1, 1],
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 0],
            [0, 2, 1],
            [0, 2, 2],
            [0, 2, 3],
            [1, 0, 0],
            [1, 0, 1],
            [10, 0, 2],  # [1, 0, 2],
            [1, 0, 3],
            [1, 1, 0],
            [1, 1, 1],
            [1, 1, 2],
            [1, 1, 3],
            [1, 2, 0],
            [1, 2, 1],
            [1, 2, 2],
            [1, 2, 3],
        ]
    )
    assert not dose_utils.is_on_grid(d)


def test_aggregate_replicates():
    """Ensure replcate doses are aggregated correctly."""
    d = np.asarray(
        [
            [0, 0, 0],
            # replicates
            [0, 0, 1],
            [0, 0, 1],
            # replicates
            [0, 1, 1],
            [0, 1, 1],
            [0, 1, 1],
        ]
    )
    d_unique = np.asarray(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
        ]
    )
    E = np.asarray([0, 1, 2, 3, 4, 50])

    d_agg, E_agg = dose_utils.aggregate_replicates(d, E)

    assert (d_agg == d_unique).all()
    assert np.allclose(E_agg, np.asarray([0, 1.5, 4]))

    d_agg, E_agg = dose_utils.aggregate_replicates(d, E, aggfunc=np.mean)
    assert (d_agg == d_unique).all()
    assert np.allclose(E_agg, np.asarray([0, 1.5, 19]))
