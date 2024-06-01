import inspect
import os

import numpy as np

from synergy.single.hill import Hill
from synergy.single.log_linear import LogLinear
import synergy.testing_utils.synthetic_data_generators as generators

TEST_DATA_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

DEFAULT_REFERENCE_PARAMETERS_MUSYC_3 = {
    "E_0": 1.0,
    "E_1": 0.0,
    "E_2": 0.0,
    "E_3": 0.0,
    "E_1,2": 0.0,
    "E_1,3": 0.0,
    "E_2,3": 0.0,
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
}

DEFAULT_EFFICACY_PARAMETERS_MUSYC_3 = {
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
}


def write_data(fname, column_names, *args):
    with open(fname, "w") as outfile:
        outfile.write(",".join(column_names) + "\n")
        for d, E in zip(*args):
            row = list(d) + [E]
            outfile.write(",".join([f"{col:0.4g}" for col in row]) + "\n")


def main():
    """Generate synthetic data for testing N-dimensional synergy models."""
    np.random.seed(412)
    write_data(
        os.path.join(TEST_DATA_DIR, "synthetic_musyc3_reference_1.csv"),
        ["d1", "d2", "d3", "E"],
        *generators.MuSyCDataGenerator.get_ND_combination(
            num_drugs=3, replicates=3, d_noise=0.01, E_noise=0.01, **DEFAULT_REFERENCE_PARAMETERS_MUSYC_3
        ),
    )

    np.random.seed(12121212)
    write_data(
        os.path.join(TEST_DATA_DIR, "synthetic_musyc3_high_order_efficacy_synergy.csv"),
        ["d1", "d2", "d3", "E"],
        *generators.MuSyCDataGenerator.get_ND_combination(
            num_drugs=3, replicates=3, d_noise=0.01, E_noise=0.01, **DEFAULT_EFFICACY_PARAMETERS_MUSYC_3
        ),
    )

    np.random.seed(120948)
    # alpha is easier to identify when beta > 0, so use the efficacy parameters
    parameters = dict(DEFAULT_EFFICACY_PARAMETERS_MUSYC_3, **{"alpha_1,2_3": 3.0})
    write_data(
        os.path.join(TEST_DATA_DIR, "synthetic_musyc3_high_order_potency_synergy.csv"),
        ["d1", "d2", "d3", "E"],
        *generators.MuSyCDataGenerator.get_ND_combination(
            num_drugs=3, replicates=3, d_noise=0.01, E_noise=0.01, **parameters
        ),
    )

    np.random.seed(492148)
    hill0 = Hill(E0=1.0, Emax=0.0, h=1.2, C=1.0)
    hill01 = Hill(E0=1.0, Emax=0.1, h=1.0, C=1.0)
    hill02 = Hill(E0=1.0, Emax=0.2, h=0.9, C=1.0)
    write_data(
        os.path.join(TEST_DATA_DIR, "synthetic_schindler3_reference.csv"),
        ["d1", "d2", "d3", "E"],
        *generators.SchindlerReferenceDataGenerator.get_ND_combination([hill0, hill01, hill02], replicates=3),
    )

    np.random.seed(34892)
    loglinear0 = LogLinear.create_fit([1 / 20.0, 20.0], [1.0, 0.0])
    loglinear01 = LogLinear.create_fit([1 / 20.0, 20.0], [1.0, 0.1])
    loglinear02 = LogLinear.create_fit([1 / 20.0, 20.0], [1.0, 0.2])
    write_data(
        os.path.join(TEST_DATA_DIR, "synthetic_bliss3_linear_reference.csv"),
        ["d1", "d2", "d3", "E"],
        *generators.MultiplicativeSurvivalReferenceDataGenerator.get_ND_combination(
            [loglinear0, loglinear01, loglinear02], replicates=3
        ),
    )


if __name__ == "__main__":
    main()
