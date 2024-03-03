from synergy.combination import MuSyC
from synergy.testing_utils.test_data_loader import load_test_data

model = MuSyC()
d1, d2, E = load_test_data("synthetic_musyc_reference_1.csv")

model.fit(d1, d2, E, bootstrap_iterations=100)
