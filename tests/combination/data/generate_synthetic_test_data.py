import inspect
import os
from typing import Dict

import numpy as np

from synergy.testing_utils.synthetic_data_generators import (
    BraidDataGenerator,
    EffectiveDoseModelDataGenerator,
    MuSyCDataGenerator,
)

TEST_DATA_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


def write_data(fname, column_names, *args):
    with open(fname, "w") as outfile:
        outfile.write(",".join(column_names) + "\n")
        for row in zip(*args):
            outfile.write(",".join([f"{col:0.4g}" for col in row]) + "\n")


def main():
    np.random.seed(490)
    write_data(
        os.path.join(TEST_DATA_DIR, "synthetic_musyc_reference_1.csv"),
        ["d1", "d2", "E"],
        *MuSyCDataGenerator.get_2drug_combination(
            E1=0.5, E2=0.0, E3=0.0, h1=1.2, h2=0.8, replicates=3, d_noise=0.01, E_noise=0.01
        ),
    )

    np.random.seed(3245)
    write_data(
        os.path.join(TEST_DATA_DIR, "synthetic_musyc_potency_1.csv"),
        ["d1", "d2", "E"],
        *MuSyCDataGenerator.get_2drug_combination(
            E1=0.5, E2=0.3, E3=0.0, h1=0.8, h2=1.4, alpha12=0.5, alpha21=2.0, replicates=3, d_noise=0.01, E_noise=0.01
        ),
    )

    np.random.seed(8989)
    write_data(
        os.path.join(TEST_DATA_DIR, "synthetic_musyc_efficacy_1.csv"),
        ["d1", "d2", "E"],
        *MuSyCDataGenerator.get_2drug_combination(
            E1=0.5, E2=0.3, E3=0.0, h1=0.8, h2=1.4, replicates=3, d_noise=0.01, E_noise=0.01
        ),
    )

    np.random.seed(28921892)
    write_data(
        os.path.join(TEST_DATA_DIR, "synthetic_musyc_efficacy_2.csv"),
        ["d1", "d2", "E"],
        *MuSyCDataGenerator.get_2drug_combination(
            E1=0.5, E2=0.0, E3=0.3, h1=0.8, h2=1.4, replicates=3, d_noise=0.01, E_noise=0.01
        ),
    )

    np.random.seed(83289)
    write_data(
        os.path.join(TEST_DATA_DIR, "synthetic_musyc_cooperativity_1.csv"),
        ["d1", "d2", "E"],
        *MuSyCDataGenerator.get_2drug_combination(
            E1=0.5,
            E2=0.3,
            E3=0.0,
            h1=0.8,
            h2=1.4,
            gamma12=0.5,
            gamma21=2.0,
            replicates=3,
            d_noise=0.01,
            E_noise=0.01,
            n_points1=6,
            n_points2=6,
        ),
    )

    np.random.seed(21746)
    write_data(
        os.path.join(TEST_DATA_DIR, "synthetic_musyc_linear_isobole_1.csv"),
        ["d1", "d2", "E"],
        *MuSyCDataGenerator.get_2drug_linear_isoboles(E1=0.2, E2=0.0, replicates=3, d_noise=0.01, E_noise=0.01),
    )

    np.random.seed(6587325)
    write_data(
        os.path.join(TEST_DATA_DIR, "synthetic_musyc_bliss_independent_1.csv"),
        ["d1", "d2", "E"],
        *MuSyCDataGenerator.get_2drug_bliss(E1=0.5, E2=0.3, h1=0.8, h2=1.4, replicates=3, d_noise=0.01, E_noise=0.01),
    )

    EDM_kwargs = {"h1": 0.8, "h2": 1.4, "replicates": 1, "n_points1": 10, "n_points2": 10, "E_noise": 0, "d_noise": 0}

    np.random.seed(34245)
    write_data(
        os.path.join(TEST_DATA_DIR, "synthetic_EDM_reference_1.csv"),
        ["d1", "d2", "E"],
        *EffectiveDoseModelDataGenerator.get_2drug_combination(**EDM_kwargs),
    )

    np.random.seed(453425)
    write_data(
        os.path.join(TEST_DATA_DIR, "synthetic_EDM_synergy_1.csv"),
        ["d1", "d2", "E"],
        *EffectiveDoseModelDataGenerator.get_2drug_combination(a12=-0.5, **EDM_kwargs),
    )

    np.random.seed(653634)
    write_data(
        os.path.join(TEST_DATA_DIR, "synthetic_EDM_synergy_2.csv"),
        ["d1", "d2", "E"],
        *EffectiveDoseModelDataGenerator.get_2drug_combination(a21=-0.5, **EDM_kwargs),
    )

    np.random.seed(123546)
    write_data(
        os.path.join(TEST_DATA_DIR, "synthetic_EDM_synergy_3.csv"),
        ["d1", "d2", "E"],
        *EffectiveDoseModelDataGenerator.get_2drug_combination(a12=-0.5, a21=-0.5, **EDM_kwargs),
    )

    BRAID_kwargs: Dict[str, float] = {
        "E2": 0.1,
        "E3": 0,
        "replicates": 1,
        "n_points1": 10,
        "n_points2": 10,
        "d1min": 1 / 50,
        "d1max": 50,
        "d2min": 1 / 50,
        "d2max": 50,
        "E_noise": 0.05,
        "d_noise": 0,
    }

    np.random.seed(921489)
    write_data(
        os.path.join(TEST_DATA_DIR, "synthetic_BRAID_reference_1.csv"),
        ["d1", "d2", "E"],
        *BraidDataGenerator.get_2drug_combination(**BRAID_kwargs),
    )

    np.random.seed(718947)
    write_data(
        os.path.join(TEST_DATA_DIR, "synthetic_BRAID_delta_synergy_1.csv"),
        ["d1", "d2", "E"],
        *BraidDataGenerator.get_2drug_combination(delta=2, **BRAID_kwargs),
    )

    np.random.seed(6234774)
    write_data(
        os.path.join(TEST_DATA_DIR, "synthetic_BRAID_delta_antagonism_1.csv"),
        ["d1", "d2", "E"],
        *BraidDataGenerator.get_2drug_combination(delta=0.5, **BRAID_kwargs),
    )

    np.random.seed(85279572)
    write_data(
        os.path.join(TEST_DATA_DIR, "synthetic_BRAID_kappa_synergy_1.csv"),
        ["d1", "d2", "E"],
        *BraidDataGenerator.get_2drug_combination(kappa=1, **BRAID_kwargs),
    )

    np.random.seed(8752375)
    write_data(
        os.path.join(TEST_DATA_DIR, "synthetic_BRAID_kappa_antagonism_1.csv"),
        ["d1", "d2", "E"],
        *BraidDataGenerator.get_2drug_combination(kappa=-1, **BRAID_kwargs),
    )

    np.random.seed(238179)
    write_data(
        os.path.join(TEST_DATA_DIR, "synthetic_BRAID_delta_kappa_synergy_1.csv"),
        ["d1", "d2", "E"],
        *BraidDataGenerator.get_2drug_combination(delta=2, kappa=1, **BRAID_kwargs),
    )

    np.random.seed(12372174)
    write_data(
        os.path.join(TEST_DATA_DIR, "synthetic_BRAID_delta_kappa_antagonism_1.csv"),
        ["d1", "d2", "E"],
        *BraidDataGenerator.get_2drug_combination(delta=0.5, kappa=-1, **BRAID_kwargs),
    )

    np.random.seed(546784)
    write_data(
        os.path.join(TEST_DATA_DIR, "synthetic_BRAID_asymmetric_1.csv"),
        ["d1", "d2", "E"],
        *BraidDataGenerator.get_2drug_combination(delta=0.5, kappa=1, **BRAID_kwargs),
    )

    np.random.seed(2121)
    write_data(
        os.path.join(TEST_DATA_DIR, "synthetic_BRAID_asymmetric_2.csv"),
        ["d1", "d2", "E"],
        *BraidDataGenerator.get_2drug_combination(delta=2, kappa=-1, **BRAID_kwargs),
    )


if __name__ == "__main__":
    main()
