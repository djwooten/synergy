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

from typing import List, Type

import numpy as np

from synergy.combination.synergy_model_2d import DoseDependentSynergyModel2D
from synergy.single import Hill
from synergy.single.dose_response_model_1d import DoseResponseModel1D


class ZIP(DoseDependentSynergyModel2D):
    """A model used to fit the Zero Interaction Potency (ZIP) model (doi: 10.1016/j.csbj.2015.09.001).

    This model is based on the multiplicative survival principal (i.e., Bliss). All across the dose-response surface,
    it fits Hill-equations either holding d1==constant or d2==constant.

    ZIP quantifies changes in the EC50 (C) and Hill-slope (h) of each drug, as the other is increased. For instance, at
    a given d1==D1, d2==D2, ZIP fits two Hill equations: one for drug1 (fixing d2==D2), and one for drug2 (fixing
    d1==D1). These Hill equations are averaged to get a fit value of E (y_c in their paper) at these doses. This is then
    subtracted from the expected result (y_zip in their paper) obtained by assuming these Hill equations have equivalent
    h and C to their single-drug counterparts. This difference (delta in their paper) becomes the metric of synergy.

    ZIP models store these delta values as model._synergy, but also store the Hill equation fits for drug1 and drug2
    across the whole surface, allowing investigation of how h and C change across the surface


    Members
    -------
    synergy : array_like
        (-inf,0)=antagonism, (0,inf)=synergism. The "delta" synergy score from ZIP

    _h_21 : array_like
        The hill slope of drug 1 obtained by holding D2==constant

    _h_12 : array_like
        The hill slope of drug 2 obtained by holding D1==constant

    _C_21 : array_like
        The EC50 of drug 1 obtained by holding D2==constant

    _C_12 : array_like
        The EC50 of drug 2 obtained by holding D1==constant
    """

    def __init__(self, use_jacobian: bool = True, drug1_model=None, drug2_model=None, **kwargs):
        super().__init__(drug1_model=drug1_model, drug2_model=drug2_model, **kwargs)
        self.use_jacobian = use_jacobian

        self._h_21: List[float] = []  # h of drug 1, holding drug 2 fixed
        self._h_12: List[float] = []  # h of drug 2, holding drug 1 fixed
        self._C_21: List[float] = []  # C of drug 1, holding drug 2 fixed
        self._C_12: List[float] = []  # C of drug 2, holding drug 1 fixed
        self._Emax_21: List[float] = []  # Emax of drug 1, holding drug 2 fixed
        self._Emax_12: List[float] = []  # Emax of drug 2, holding drug 1 fixed

    @property
    def _required_single_drug_class(self) -> Type[DoseResponseModel1D]:
        return Hill

    @property
    def _default_single_drug_class(self) -> Type[DoseResponseModel1D]:
        return Hill

    def _get_synergy(self, d1, d2, E):
        drug1_model = self.drug1_model
        drug2_model = self.drug2_model

        if not (isinstance(drug1_model, Hill) and isinstance(drug2_model, Hill)):
            raise ValueError("Drug models are incorrect")

        E0_1, Emax_1, h1, C1 = drug1_model.E0, drug1_model.Emax, drug1_model.h, drug1_model.C
        E0_2, Emax_2, h2, C2 = drug2_model.E0, drug2_model.Emax, drug2_model.h, drug2_model.C
        E0 = (E0_1 + E0_2) / 2.0
        Emax = (Emax_1 + Emax_2) / 2.0

        if E0 < Emax:
            drug1_model.E0 = Emax_1
            drug2_model.E0 = Emax_2
            drug1_model.Emax = E0_1
            drug2_model.Emax = E0_2

            tmp = E0
            E0 = Emax
            Emax = tmp

        self._h_21 = []
        self._h_12 = []
        self._C_21 = []
        self._C_12 = []
        self._Emax_21 = []
        self._Emax_12 = []

        zip_model = _Hill_3P(Emax_bounds=(0, 1.5))

        for D1, D2 in zip(d1, d2):
            # Fix d2==D2, and fit hill for D1
            mask = np.where(d2 == D2)
            y2 = drug2_model.E(D2)
            zip_model.E0 = y2
            zip_model.fit(d1[mask], E[mask], use_jacobian=self.use_jacobian, p0=[Emax_1, h1, C1])
            self._h_21.append(zip_model.h)
            self._C_21.append(zip_model.C)
            self._Emax_21.append(zip_model.Emax)

            # Fix d1==D1, and fit hill for D2
            mask = np.where(d1 == D1)
            y1 = drug1_model.E(D1)
            zip_model.E0 = y1
            zip_model.fit(d2[mask], E[mask], use_jacobian=self.use_jacobian, p0=[Emax_2, h2, C2])
            self._h_12.append(zip_model.h)
            self._C_12.append(zip_model.C)
            self._Emax_12.append(zip_model.Emax)

        self._h_21 = np.asarray(self._h_21)
        self._h_12 = np.asarray(self._h_12)
        self._C_21 = np.asarray(self._C_21)
        self._C_12 = np.asarray(self._C_12)
        self._Emax_21 = np.asarray(self._Emax_21)
        self._Emax_12 = np.asarray(self._Emax_12)

        synergy = self._delta_score(d1, d2)

        return self._sanitize_synergy(d1, d2, synergy, 0.0)

    def E_reference(self, d1, d2):
        E1_alone, E2_alone = self._get_single_drug_Es(d1, d2)
        return E1_alone * E2_alone

    def _delta_score(self, d1, d2):
        """Calculate the difference between the Bliss reference surface and the (averaged) fit 1D slices used by ZIP

        drug2_alone  drug2
             |         |
             |         |
             |=========X==== drug1
             |         |
             +--------------- drug1_alone

        Notice that E0 of "drug1" starts at E of drug2_alone, and vice versa for "drug2"
        "X" marks (d1, d2)
        """
        E1_alone, E2_alone = self._get_single_drug_Es(d1, d2)

        hill = Hill()
        zip_drug_1 = hill._model(d1, E2_alone, self._Emax_21, self._h_21, self._C_21)
        zip_drug_2 = hill._model(d2, E1_alone, self._Emax_12, self._h_12, self._C_12)
        zip_fit = (zip_drug_1 + zip_drug_2) / 2.0

        return self.reference - zip_fit

    def _get_single_drug_Es(self, d1, d2):
        """Calculate these manually so that E0 uses the average, rather than E0_1 and E0_2"""
        if not (isinstance(self.drug1_model, Hill) and isinstance(self.drug2_model, Hill)):
            raise ValueError("Drug models are incorrect")
        E0_1 = self.drug1_model.E0
        E0_2 = self.drug2_model.E0
        E0 = (E0_1 + E0_2) / 2.0

        hill = Hill()
        E1_alone = hill._model(d1, E0, self.drug1_model.Emax, self.drug1_model.h, self.drug1_model.C)
        E2_alone = hill._model(d2, E0, self.drug2_model.Emax, self.drug2_model.h, self.drug2_model.C)

        return E1_alone, E2_alone


class _Hill_3P(Hill):
    """ZIP uses a three parameter Hill equation which fixes E0, but fits Emax, C, and h"""

    def __init__(self, **kwargs):
        self.E0 = 1.0  # this gets overwritten over the course of the algorithm
        super().__init__(**kwargs)

    @property
    def _parameter_names(self) -> List[str]:
        return ["Emax", "h", "C"]

    def _model_to_fit(self, d, Emax, logh, logC):
        return self._model(d, self.E0, Emax, np.exp(logh), np.exp(logC))

    def _model_jacobian_for_fit(self, d, Emax, logh, logC):
        dh = d ** (np.exp(logh))
        Ch = (np.exp(logC)) ** (np.exp(logh))
        logd = np.log(d)
        E0 = self.E0
        jEmax = dh / (Ch + dh)
        jC = (E0 - Emax) * dh * np.exp(logh + logC) * (np.exp(logC)) ** (np.exp(logh) - 1) / ((Ch + dh) * (Ch + dh))
        jh = (Emax - E0) * dh * np.exp(logh) * ((Ch + dh) * logd - (logC * Ch + logd * dh)) / ((Ch + dh) * (Ch + dh))
        jac = np.hstack((jEmax.reshape(-1, 1), jh.reshape(-1, 1), jC.reshape(-1, 1)))
        jac[np.isnan(jac)] = 0
        return jac

    def _get_initial_guess(self, d, E, p0):
        if p0 is None:
            p0 = [np.nanmin(E), 1, np.median(d)]

        return super()._get_initial_guess(d, E, p0)

    def _set_parameters(self, popt):
        Emax, h, C = popt

        self.Emax = Emax
        self.h = h
        self.C = C

    def _transform_params_from_fit(self, params):
        return params[0], np.exp(params[1]), np.exp(params[2]) * self._dose_scale

    def _transform_params_to_fit(self, params):
        with np.errstate(divide="ignore"):
            return params[0], np.log(params[1]), np.log(params[2] / self._dose_scale)

    def __repr__(self):
        if not self.is_specified:
            return "Hill_3P()"

        return "Hill_3P(E0=%0.3g, Emax=%0.3g, h=%0.3g, C=%0.3g)" % (self.E0, self.Emax, self.h, self.C)
