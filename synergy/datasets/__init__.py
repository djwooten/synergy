from importlib import resources
from typing import Sequence

import numpy as np

from synergy.single.hill import Hill
from synergy.testing_utils.synthetic_data_generators import (
    HillDataGenerator,
    MultiplicativeSurvivalReferenceDataGenerator,
    MuSyCDataGenerator,
    ShamDataGenerator,
)

DEFAULT_DATA_MODULE = "synergy.datasets.data"


def _load_data(fname: str) -> np.typing.NDArray[np.float64]:
    lines = []
    path = resources.files(DEFAULT_DATA_MODULE).joinpath(fname)
    with open(path) as infile:  # type: ignore
        infile.readline()  # Get rid of header
        for line in infile:
            line_split = line.strip().split(sep=",")
            lines.append([float(val) for val in line_split])
    return np.asarray(lines)


def _write_data(fname: str, column_names: Sequence[str], *args):
    path = resources.files(DEFAULT_DATA_MODULE).joinpath(fname)
    with open(path, "w") as outfile:  # type: ignore
        outfile.write(",".join(column_names) + "\n")
        for row in zip(*args):
            outfile.write(",".join([f"{col:0.4g}" for col in row]) + "\n")


def load_hill_example():
    """Load example data for a single drug."""
    data = _load_data("hill.csv")
    return data[:, 0], data[:, 1]


def load_2d_sham_example():
    """Load example data for a 2-drug sham combination."""
    return _load_data("sham_2d.csv")


def load_3d_sham_example():
    """Load example data for a 3-drug sham combination."""
    return _load_data("sham_3d.csv")


def load_2d_example():
    """Load example data for 2-drug combination."""
    data = _load_data("2d.csv")
    return data[:, 0], data[:, 1], data[:, 2]


def load_3d_example():
    """Load example data for 3-drug combination"""
    data = _load_data("3d.csv")
    return data[:, :-1], data[:, -1]


def load_4d_example():
    """Load example data for 4-drug combination."""
    data = _load_data("4d.csv")
    return data[:, :-1], data[:, -1]


def main():
    """Generate example datasets."""
    np.random.seed(531135)
    _write_data("hill.csv", ["drug.conc", "E"], *HillDataGenerator.get_data(replicates=3))

    np.random.seed(89645)
    _write_data(
        "sham_2d.csv",
        ["drug1.conc", "drug2.conc", "E"],
        *ShamDataGenerator.get_combination(Hill(E0=1, Emax=0, C=1, h=1), 0.05, 20, 6, replicates=3),
    )

    np.random.seed(1234)
    d, E = ShamDataGenerator.get_ND_combination(Hill(E0=1, Emax=0, C=1, h=1), 3, 0.05, 20, n_points=6, replicates=3)
    _write_data("sham_3d.csv", ["drug1.conc", "drug2.conc", "drug3.conc", "E"], *d.transpose(), E)

    np.random.seed(490)
    _write_data(
        "2d.csv",
        ["drug1.conc", "drug2.conc", "E"],
        *MuSyCDataGenerator.get_2drug_combination(E1=0.5, E2=0.2, E3=0.0, h1=1.2, h2=0.8, alpha12=2.8, replicates=3),
    )

    np.random.seed(3245)
    params = {
        "E_0": 1.0,
        "E_1": 0.7,
        "E_2": 0.5,
        "E_3": 0.4,
        "E_1,2": 0.5,
        "E_1,3": 0.45,
        "E_2,3": 0.2,
        "E_1,2,3": 0.0,
        "h_1": 2.0,
        "h_2": 0.5,
        "h_3": 1.0,
        "C_1": 1.0,
        "C_2": 1.0,
        "C_3": 1.0,
        "alpha_1_2": 1.0,
        "alpha_1_3": 1.0,
        "alpha_2_1": 1.0,
        "alpha_2_3": 1.0,
        "alpha_3_1": 1.0,
        "alpha_3_2": 0.5,
        "alpha_1,2_3": 1.0,
        "alpha_1,3_2": 1.0,
        "alpha_2,3_1": 3.0,
    }
    d, E = MuSyCDataGenerator.get_ND_combination(replicates=3, **params)
    _write_data("3d.csv", ["drug1.conc", "drug2.conc", "drug3.conc", "E"], *d.transpose(), E)

    np.random.seed(8989)
    hill1 = Hill(E0=1.0, Emax=0.5, h=2.0, C=1.0)
    hill2 = Hill(E0=1.0, Emax=0.5, h=0.5, C=1.0)
    hill3 = Hill(E0=1.0, Emax=0.3, h=3.0, C=1.0)
    hill4 = Hill(E0=1.0, Emax=0.2, h=1.0, C=1.0)
    d, E = MultiplicativeSurvivalReferenceDataGenerator.get_ND_combination([hill1, hill2, hill3, hill4], replicates=3)
    _write_data("4d.csv", ["drug1.conc", "drug2.conc", "drug3.conc", "drug4.conc", "E"], *d.transpose(), E)


if __name__ == "__main__":
    main()
