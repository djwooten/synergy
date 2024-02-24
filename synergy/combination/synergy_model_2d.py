from abc import ABC, abstractmethod, abstractproperty
from copy import deepcopy

from synergy.single.dose_response_model_1d import DoseResponseModel1D
from synergy.utils import base as utils


class SynergyModel2D(ABC):
    """-"""

    def __init__(self, drug1_model: DoseResponseModel1D, drug2_model: DoseResponseModel1D):
        """-"""
        default_type = self._default_single_drug_class
        required_type = self._required_single_drug_class

        drug1_model = deepcopy(drug1_model)
        drug2_model = deepcopy(drug2_model)
        self.drug1_model: DoseResponseModel1D = utils.sanitize_single_drug_model(
            drug1_model, default_type, required_type
        )
        self.drug2_model: DoseResponseModel1D = utils.sanitize_single_drug_model(
            drug2_model, default_type, required_type
        )

    @abstractmethod
    def fit(self, d1, d2, E, **kwargs):
        """-"""

    @abstractmethod
    def E_reference(self, d1, d2):
        """-"""

    @abstractproperty
    def _required_single_drug_class(self) -> type[DoseResponseModel1D]:
        """-"""

    @abstractproperty
    def _default_single_drug_class(self) -> type[DoseResponseModel1D]:
        """-"""

    @abstractproperty
    def is_specified(self):
        """-"""

    @abstractproperty
    def is_fit(self):
        """-"""


class DoseDependentSynergyModel2D(SynergyModel2D):
    """-"""

    def __init__(self, drug1_model: DoseResponseModel1D, drug2_model: DoseResponseModel1D):
        """-"""
        super().__init__(drug1_model, drug2_model)
        self.synergy = None
        self.d1 = None
        self.d2 = None
        self.reference = None

    @abstractmethod
    def _get_synergy(self, d1, d2, E):
        """-"""

    def _sanitize_synergy(self, d1, d2, synergy, default_val: float):
        if hasattr(synergy, "__iter__"):
            synergy[(d1 == 0) | (d2 == 0)] = default_val
        elif d1 == 0 or d2 == 0:
            synergy = default_val
        return synergy


class ParametricSynergyModel2D(SynergyModel2D):
    """-"""

    @abstractmethod
    def E(self, d1, d2):
        """-"""

    @abstractproperty
    def parameters(self):
        """-"""

    @abstractmethod
    def get_confidence_intervals(self, confidence_interval: float = 95):
        """-"""
