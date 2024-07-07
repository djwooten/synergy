import numpy as np


def load_test_data(fname, sep=",") -> tuple:
    lines = []
    with open(fname) as infile:
        infile.readline()  # Get rid of header
        for line in infile:
            line_split = line.strip().split(sep=sep)
            lines.append([float(val) for val in line_split])
    return tuple(np.asarray(lines).transpose())


def load_nd_test_data(fname, sep=",") -> tuple:
    data = load_test_data(fname, sep)
    return np.asarray(data[:-1]).transpose(), data[-1]
