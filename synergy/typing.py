from typing import Union

from synergy.single.dose_response_model_1d import DoseResponseModel1D


Model1D = Union[DoseResponseModel1D, type[DoseResponseModel1D]]
