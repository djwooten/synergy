import numpy as np
import numpy.testing as npt
import pytest

import synergy.testing_utils.synthetic_data_generators as generators
from synergy.combination import HSA, Loewe, Bliss, ZIP, CombinationIndex, Schindler
from synergy.combination.synergy_model_2d import DoseDependentSynergyModel2D
from synergy.single import Hill, Hill_CI, LogLinear
from synergy.single.dose_response_model_1d import DoseResponseModel1D

# Used for all models that require Hill
_SINGLE_HILL_MODELS: list[DoseResponseModel1D] = [
    Hill(E0=1.0, Emax=0.5, h=0.8, C=1.0),
    Hill(E0=1.0, Emax=0.0, h=1.2, C=1.0),
]
# Used for CombinationIndex
_SINGLE_HILL_CI_MODELS: list[DoseResponseModel1D] = [Hill_CI(h=0.8, C=1.0), Hill_CI(h=1.2, C=1.0)]
# Used for any model that allows LogLinear
_SINGLE_LOG_LINEAR_MODELS: list[DoseResponseModel1D] = [
    LogLinear.create_fit([1e-3, 1e3], [1.0, 0.1]),
    LogLinear.create_fit([1e-3, 1e3], [1.0, 0.2]),
]
# Map of models to reference data generators
_MODEL_TO_GENERATOR: dict[DoseDependentSynergyModel2D, generators.DoseDependentReferenceDataGenerator] = {
    HSA: generators.HSAReferenceDataGenerator,
    Loewe: generators.ShamDataGenerator,
    Bliss: generators.MultiplicativeSurvivalReferenceDataGenerator,
    ZIP: generators.MultiplicativeSurvivalReferenceDataGenerator,
    CombinationIndex: generators.ShamDataGenerator,
    Schindler: generators.ShamDataGenerator,
}
# Map of models to single drugs to use for test
_MODEL_TO_SINGLE_MODELS: dict[DoseDependentSynergyModel2D, list[DoseResponseModel1D]] = {
    HSA: _SINGLE_LOG_LINEAR_MODELS,
    Loewe: _SINGLE_LOG_LINEAR_MODELS,
    Bliss: _SINGLE_LOG_LINEAR_MODELS,
    ZIP: _SINGLE_HILL_MODELS,
    CombinationIndex: _SINGLE_HILL_CI_MODELS,
    Schindler: _SINGLE_HILL_MODELS,
}
# Map of models to tolerances for synergy status
_MODEL_TO_TOL: dict[DoseDependentSynergyModel2D, float] = {
    HSA: 1e-15,
    Loewe: 1e-5,
    Bliss: 1e-15,
    ZIP: 2e-2,
    CombinationIndex: 1e-1,
    Schindler: 1e-7,
}


class TestDoseDependentSynergy:
    """Tests for 2D dose-dependent synergy models."""

    @pytest.mark.parametrize("ModelClass", _MODEL_TO_GENERATOR.keys())
    def test_reference_data_has_no_synergy(self, ModelClass: type[DoseDependentSynergyModel2D]):
        """Ensure reference data has no synergy."""
        dmin, dmax = 1e-3, 1e3
        drug1_model, drug2_model = _MODEL_TO_SINGLE_MODELS[ModelClass]
        generator = _MODEL_TO_GENERATOR[ModelClass]

        d1, d2, E = generator.get_combination(
            drug1_model, drug2_model, dmin, dmax, dmin, dmax, n_points1=10, n_points2=10, E_noise=0.0, d_noise=0.0
        )

        d1 = d1[~np.isnan(E)]
        d2 = d2[~np.isnan(E)]
        E = E[~np.isnan(E)]

        model = ModelClass()
        model.fit(d1, d2, E)

        assert model.is_fit, "Model should be fit"
        assert model.is_specified, "Model should be specified after fitting"

        tol = _MODEL_TO_TOL[ModelClass]

        synergy_status = model.get_synergy_status(tol=tol)
        if ModelClass in [Loewe, Schindler, CombinationIndex]:
            synergy_status = synergy_status[1:]
        assert (synergy_status == "Additive").all(), "Expected all points to be additive"

    @pytest.mark.parametrize("ModelClass", _MODEL_TO_GENERATOR.keys())
    def test_synergistic_data_is_synergistic(self, ModelClass: type[DoseDependentSynergyModel2D]):
        """-"""
        dmin, dmax = 1e-3, 1e3
        drug1_model, drug2_model = _MODEL_TO_SINGLE_MODELS[ModelClass]
        generator = _MODEL_TO_GENERATOR[ModelClass]

        d1, d2, E = generator.get_combination(
            drug1_model, drug2_model, dmin, dmax, dmin, dmax, n_points1=30, n_points2=30, E_noise=0.0, d_noise=0.0
        )

        d1 = d1[~np.isnan(E)]
        d2 = d2[~np.isnan(E)]
        E = E[~np.isnan(E)]

        combination_mask = np.where((d1 > np.min(d1)) & (d2 > np.min(d2)))  # this is where drugs are in true combo
        E[combination_mask] *= E[combination_mask]  # make the drug stronger in combination

        model = ModelClass()
        synergy = model.fit(d1, d2, E)

        assert model.is_fit, "Model should be fit"
        assert model.is_specified, "Model should be specified after fitting"

        # TODO there are still some issues with ZIP

        synergy_status = model.get_synergy_status()
        synergy_status = synergy_status[combination_mask]
        synergy = synergy[combination_mask]

        non_nan_mask = np.where(~np.isnan(synergy))
        assert (synergy_status[non_nan_mask] == "Synergistic").all(), "Expected all points to be synergistic"

    @pytest.mark.parametrize("ModelClass", _MODEL_TO_GENERATOR.keys())
    def test_antagonistic_data_is_antagonistic(self, ModelClass: type[DoseDependentSynergyModel2D]):
        """-"""
        dmin, dmax = 1e-3, 1e3
        drug1_model, drug2_model = _MODEL_TO_SINGLE_MODELS[ModelClass]
        generator = _MODEL_TO_GENERATOR[ModelClass]

        d1, d2, E = generator.get_combination(
            drug1_model, drug2_model, dmin, dmax, dmin, dmax, n_points1=30, n_points2=30, E_noise=0.0, d_noise=0.0
        )

        d1 = d1[~np.isnan(E)]
        d2 = d2[~np.isnan(E)]
        E = E[~np.isnan(E)]

        combination_mask = np.where((d1 > np.min(d1)) & (d2 > np.min(d2)))  # this is where drugs are in true combo
        E[combination_mask] = np.sqrt(E[combination_mask])  # make the drug weaker in combination

        model = ModelClass()
        synergy = model.fit(d1, d2, E)

        assert model.is_fit, "Model should be fit"
        assert model.is_specified, "Model should be specified after fitting"

        synergy_status = model.get_synergy_status()
        synergy_status = synergy_status[combination_mask]
        synergy = synergy[combination_mask]

        non_nan_mask = np.where(~np.isnan(synergy))
        assert (synergy_status[non_nan_mask] == "Antagonistic").all(), "Expected all points to be antagonistic"
