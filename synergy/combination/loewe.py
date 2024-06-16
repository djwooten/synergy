#    Copyright (C) 2020 David J. Wooten
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np

from synergy.single import Hill, LogLinear
from synergy.combination.synergy_model_2d import DoseDependentSynergyModel2D
from synergy.single.dose_response_model_1d import DoseResponseModel1D

# Used for delta mode of Loewe
from scipy.optimize import minimize_scalar


class Loewe(DoseDependentSynergyModel2D):
    """The Loewe additivity non-parametric synergy model for combinations of two drugs.

    Loewe's model of drug combination additivity expects a linear tradeoff, such that withholding ```X``` parts of
    drug 1 can be compensated for by adding ```Y``` parts of drug 2. In Loewe's model, ```X``` and ```Y``` are constant
    (e.g., withholding 5X parts of drug 1 will be compensated for with 5Y parts of drug 2.)

    The Loewe model is used to calculate a dose-dependent scalar value of synergy. Two modes are supported: "CI" and "delta"

    CI mode (default)
    =================
    This mode calculates synergy using an equation equivalent to combination index. But unlike CI, Loewe allows arbitrary
    single-drug models.

    In CI mode, 0 <= Loewe < 1 indicates synergism, while Loewe > 1 indicates antagonism


    delta mode
    ==========
    This mode calculates an expected dose response surface, and defines synergy as the difference between the measured
    values and the expected values. This mode requires single-drugs be fitted using a Hill model.

    In delta mode, Loewe > 0 indicates synergism, while Loewe < 0 indicates antagonism
    """

    def __init__(self, mode: str = "CI", drug1_model=None, drug2_model=None, **kwargs):
        """Ctor."""
        mode = mode.lower()
        if mode not in ["ci", "delta", "delta_hsa", "delta_nan", "delta_synergyfinder"]:
            raise ValueError("Unrecognized mode for Loewe ({mode})")
        self.mode = mode

        super().__init__(drug1_model=drug1_model, drug2_model=drug2_model)

    @property
    def _required_single_drug_class(self) -> type[DoseResponseModel1D]:
        """-"""
        if self.mode == "ci":
            return DoseResponseModel1D
        return Hill

    @property
    def _default_single_drug_class(self) -> type[DoseResponseModel1D]:
        """-"""
        if self.mode == "ci":
            return LogLinear
        return Hill

    def _get_synergy(self, d1, d2, E):
        if self.mode == "ci":
            return self._get_synergy_CI(d1, d2, E)
        return self._get_synergy_delta(d1, d2, E)

    def _get_synergy_delta(self, d1, d2, E):
        # Save reference and synergy
        if self.reference is None:
            self.reference = self.E_reference(d1, d2)
        synergy = self.reference - E

        return self._sanitize_synergy(d1, d2, synergy, 0)

    def _get_synergy_CI(self, d1, d2, E):
        with np.errstate(divide="ignore", invalid="ignore"):
            d1_alone = self.drug1_model.E_inv(E)
            d2_alone = self.drug2_model.E_inv(E)
            synergy = d1 / d1_alone + d2 / d2_alone

        return self._sanitize_synergy(d1, d2, synergy, 1.0)

    def _fit_Loewe_reference(self, d1, d2):
        """Calculates a reference (null) model for Loewe for drug1 and drug2 at concentrations X_r and X_c

        drug1_model and drug2_model MUST be some form of Hill equation

        Credsits: Mark Russo and David Wooten

        Returns scipy.optimize.minimize_scalar object
        """
        # TODO: Add E_range to the base single drug model, because the only reason we need
        # a Hill here is to get E0 and Emax. But we could get similar from LogLinear.
        if not (isinstance(self.drug1_model, Hill) and isinstance(self.drug2_model, Hill)):
            raise ValueError("Drug models are incorrect")
        # Compute the bounds within which Y_Loewe is valid.
        # Any value outside these bounds causes algorithm to take a root of a negative.
        bounds1 = sorted([self.drug1_model.E0, self.drug1_model.Emax])
        bounds2 = sorted([self.drug2_model.E0, self.drug2_model.Emax])
        bounds = [max(bounds1[0], bounds2[0]), min(bounds1[1], bounds2[1])]

        # Quadratic function with a minimum at CI=1
        # f = lambda Y: ((d1 / self.drug1_model.E_inv(Y)) + (d2 / self.drug2_model.E_inv(Y)) - 1.0) ** 2.0

        # Perform constrained minimization to find Y as close as possible to CI == 1 and return result.
        return minimize_scalar(
            self._quadratic_reference_objective_function, args=(d1, d2), method="bounded", bounds=bounds
        )

    def _quadratic_reference_objective_function(self, E, d1, d2):
        """Quadratic curve used to find the value of the reference null model

        Based on the combination index definition that
            d1 / E_inv(E) + d2 / E_inv(E) = 1
        for Loewe additivity.

        Credits: Mark Russo
        """
        return np.float_power((d1 / self.drug1_model.E_inv(E)) + (d2 / self.drug2_model.E_inv(E)) - 1.0, 2.0)

    def E_reference(self, d1, d2):
        """Calculates a reference (null) model for Loewe for drug1 and drug2.

        Credits: Mark Russo, David Wooten
        """
        if not (isinstance(self.drug1_model, Hill) and isinstance(self.drug2_model, Hill)):
            # TODO: Log a warning
            raise ValueError("E_reference() for this model requires individual drugs to be Hill models")

        with np.errstate(divide="ignore", invalid="ignore"):
            E_ref = 0 * d1

            Emax_1 = self.drug1_model.Emax
            Emax_2 = self.drug2_model.Emax

            weakest_E = max(Emax_1, Emax_2)

            stronger_drug = self.drug1_model
            if Emax_1 > Emax_2:
                stronger_drug = self.drug2_model

            # Loewe becomes undefined for effects past the weaker drug's Emax
            # We implement several modes to handle this case:
            #  1) mode="delta" - this will set E_reference to weakest_E
            #  2) mode="delta_HSA" - this will set E_r to min(E1, E2)
            #  3) mode="delta_nan" - this will set E_r to nan
            #  4) mode="synergyfinder" - sets E_r to stronger.E(d1+d2)
            if self.mode == "delta":
                option = 1
            elif self.mode == "delta_hsa":
                option = 2
            elif self.mode == "delta_nan":
                option = 3
            elif self.mode == "delta_synergyfinder":
                option = 4
            else:
                option = 1

            for i in range(len(E_ref)):
                D1, D2 = d1[i], d2[i]
                E1_alone = self.drug1_model.E(D1)
                E2_alone = self.drug2_model.E(D2)

                # No drug so E1 should equal E2, but let's take the average to be safer
                if D1 == 0.0 and D2 == 0.0:
                    E_ref[i] = 0.5 * (E2_alone + E1_alone)

                # Single drugs
                elif D2 == 0:
                    E_ref[i] = E1_alone
                elif D1 == 0:
                    E_ref[i] = E2_alone

                # Use a predefined reference behavior for when E1_alone or E2_alone exceed the weaker of the pair
                elif E1_alone < weakest_E or E2_alone < weakest_E:
                    if option == 1:
                        E_ref[i] = weakest_E
                    elif option == 2:
                        E_ref[i] = min(E1_alone, E2_alone)
                    elif option == 3:
                        E_ref[i] = np.nan
                    elif option == 4:
                        E_ref[i] = stronger_drug.E(D1 + D2)
                    else:
                        E_ref[i] = np.nan
                else:
                    # Numerically solve the value for Loewe
                    res = self._fit_Loewe_reference(D1, D2)
                    if res.success:
                        E_ref[i] = res.x
                    else:
                        E_ref[i] = min(E2_alone, E1_alone)
        return E_ref

    def plot_heatmap(self, cmap="PRGn", neglog=None, center_on_zero=True, **kwargs):
        if neglog is None:
            if self.mode == "delta":
                neglog = False
            else:
                neglog = True
        # super().plot_heatmap(cmap=cmap, neglog=neglog, center_on_zero=center_on_zero, **kwargs)

    def plot_surface_plotly(self, cmap="PRGn", neglog=None, center_on_zero=True, **kwargs):
        if neglog is None:
            if self.mode == "delta":
                neglog = False
            else:
                neglog = True
        # super().plot_surface_plotly(cmap=cmap, neglog=neglog, center_on_zero=center_on_zero, **kwargs)
