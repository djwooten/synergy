import inspect
import os

import numpy as np

from synergy.testing_utils.synthetic_data_generators import HillDataGenerator

TEST_DATA_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


def write_data(fname, column_names, *args):
    with open(fname, "w") as outfile:
        outfile.write(",".join(column_names) + "\n")
        for row in zip(*args):
            outfile.write(",".join([f"{col:0.4g}" for col in row]) + "\n")


def main():
    """-"""
    np.random.seed(2)
    write_data(
        os.path.join(TEST_DATA_DIR, "synthetic_hill_1.csv"),
        ["d", "E"],
        *HillDataGenerator.get_data(replicates=3, E_noise=0.1, d_noise=0.1),
    )


if __name__ == "__main__":
    main()
