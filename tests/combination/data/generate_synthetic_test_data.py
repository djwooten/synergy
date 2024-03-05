import inspect
import os

import numpy as np

from synergy.testing_utils.synthetic_data_generators import MuSyCDataGenerator, EffectiveDoseModelDataGenerator

TEST_DATA_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


def write_data(fname, column_names, *args):
    with open(fname, "w") as outfile:
        outfile.write(",".join(column_names) + "\n")
        for row in zip(*args):
            outfile.write(",".join([f"{col:0.4g}" for col in row]) + "\n")


def main():
    """-"""
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

    np.random.seed(34245)
    write_data(
        os.path.join(TEST_DATA_DIR, "synthetic_EDM_reference_1.csv"),
        ["d1", "d2", "E"],
        *EffectiveDoseModelDataGenerator.get_2drug_combination(
            h1=0.8, h2=1.4, replicates=3, d_noise=0.0001, E_noise=0.0001, n_points1=10, n_points2=10
        ),
    )

    np.random.seed(453425)
    write_data(
        os.path.join(TEST_DATA_DIR, "synthetic_EDM_synergy_1.csv"),
        ["d1", "d2", "E"],
        *EffectiveDoseModelDataGenerator.get_2drug_combination(
            h1=0.8, h2=1.4, a12=-0.5, replicates=3, d_noise=0.0001, E_noise=0.0001, n_points1=10, n_points2=10
        ),
    )

    np.random.seed(653634)
    write_data(
        os.path.join(TEST_DATA_DIR, "synthetic_EDM_synergy_2.csv"),
        ["d1", "d2", "E"],
        *EffectiveDoseModelDataGenerator.get_2drug_combination(
            h1=0.8, h2=1.4, a21=-0.5, replicates=3, d_noise=0.0001, E_noise=0.0001, n_points1=10, n_points2=10
        ),
    )

    np.random.seed(123546)
    write_data(
        os.path.join(TEST_DATA_DIR, "synthetic_EDM_synergy_3.csv"),
        ["d1", "d2", "E"],
        *EffectiveDoseModelDataGenerator.get_2drug_combination(
            h1=0.8, h2=1.4, a12=-0.5, a21=-0.5, replicates=3, d_noise=0.0001, E_noise=0.0001, n_points1=10, n_points2=10
        ),
    )


if __name__ == "__main__":
    main()
