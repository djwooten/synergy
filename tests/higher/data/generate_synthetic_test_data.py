import inspect
import os

import numpy as np

import synergy.testing_utils.synthetic_data_generators as generators

TEST_DATA_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


def write_data(fname, column_names, *args):
    with open(fname, "w") as outfile:
        outfile.write(",".join(column_names) + "\n")
        for d, E in zip(*args):
            row = list(d) + [E]
            outfile.write(",".join([f"{col:0.4g}" for col in row]) + "\n")


def main():
    """-"""
    np.random.seed(412)
    write_data(
        os.path.join(TEST_DATA_DIR, "synthetic_musyc3_reference_1.csv"),
        ["d1", "d2", "d3", "E"],
        *generators.MuSyCDataGenerator.get_ND_combination(num_drugs=3, replicates=3),
    )


if __name__ == "__main__":
    main()
