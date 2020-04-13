"""
    Copyright (C) 2020 David J. Wooten

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import synergy.utils.utils as utils

class ParameterizedModel:
    """

    """
    def __init__(self):

        self.sum_of_squares_residuals = None
        self.r_squared = None
        self.aic = None
        self.bic = None
#        self.converged = False

    def _score(self, d1, d2, E):
        if (self._is_parameterized()):

            n_parameters = len(self.get_parameters())

            self.sum_of_squares_residuals = utils.residual_ss(d1, d2, E, self.E)
            self.r_squared = utils.r_squared(E, self.sum_of_squares_residuals)
            self.aic = utils.AIC(self.sum_of_squares_residuals, n_parameters, len(E))
            self.bic = utils.BIC(self.sum_of_squares_residuals, n_parameters, len(E))

    def get_parameters():
        return []

    def _is_parameterized(self):
        return not (None in self.get_parameters() or True in np.isnan(np.asarray(self.get_parameters())))

class ModelNotParameterizedError(Exception):
    """
    The model must be parameterized prior to use. This can be done by calling
    fit(), or setParameters().
    """
    def __init__(self, msg='The model must be parameterized prior to use. This can be done by calling fit(), or setParameters().', *args, **kwargs):
        super().__init__(msg, *args, **kwargs)

class FeatureNotImplemented(Warning):
    """
    """
    def __init__(self, msg="This feature is not yet implemented", *args, **kwargs):
        super().__init__(msg, *args, **kwargs)
