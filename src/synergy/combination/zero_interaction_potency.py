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
import warnings
from scipy.optimize import curve_fit
from .. import utils
from ..single import hill as hill
from .base import *

class ZIP(DoseDependentModel):
    """The Zero Interaction Potency (ZIP) model (doi: 10.1016/j.csbj.2015.09.001). This model is based on the multiplicative survival principal (i.e., Bliss). All across the dose-response surface, it fits Hill-equations either holding d1==constant or d2==constant.
    
    ZIP quantifies changes in the EC50 (C) and Hill-slope (h) of each drug, as the other is increased. For instance, at a given d1==D1, d2==D2, ZIP fits two Hill equations: one for drug1 (fixing d2==D2), and one for drug2 (fixing d1==D1). These Hill equations are averaged to get a fit value of E (y_c in their paper) at these doses. This is then subtracted from the expected result (y_zip in their paper) obtained by assuming these Hill equations have equivalent h and C to their single-drug counterparts. This difference (delta in their paper) becomes the metric of synergy.

    ZIP models store these delta values as model._synergy, but also store the Hill equation fits for drug1 and drug2 across the whole surface, allowing investigation of how h and C change across the surface

    --------

    synergy : array-like, (-inf,0)=antagonism, (0,inf)=synergism
        The "delta" synergy score from ZIP

    _h_21 : array-like
        The hill slope of drug 1 obtained by holding D2==constant

    _h_12 : array-like
        The hill slope of drug 2 obtained by holding D1==constant

    _C_21 : array-like
        The EC50 of drug 1 obtained by holding D2==constant

    _C_12 : array-like
        The EC50 of drug 2 obtained by holding D1==constant
    """
    def __init__(self, E0_bounds=(-np.inf,np.inf), Emax_bounds=(-np.inf,np.inf), h_bounds=(0,np.inf), C_bounds=(0,np.inf)):

        super().__init__(h1_bounds=h_bounds, h2_bounds=h_bounds, C1_bounds=C_bounds, C2_bounds=C_bounds, E0_bounds=E0_bounds, E1_bounds=Emax_bounds, E2_bounds=Emax_bounds)

        self.E0_bounds = E0_bounds
        self.Emax_bounds = Emax_bounds
        self.C_bounds = C_bounds
        self.h_bounds = h_bounds

        self._h_21 = []
        self._h_12 = []
        self._C_21 = []
        self._C_12 = []

    def fit(self, d1, d2, E, drug1_model=None, drug2_model=None, use_jacobian=True, **kwargs):
    
        super().fit(d1, d2, E)
        
        if drug1_model is None:
            mask = np.where(d2==min(d2))
            drug1_model = hill.Hill.create_fit(d1[mask], E[mask], E0_bounds=self.E0_bounds, Emax_bounds=self.Emax_bounds, h_bounds=self.h_bounds, C_bounds=self.C_bounds, use_jacobian=use_jacobian)
            
        if drug2_model is None:
            mask = np.where(d1==min(d1))
            drug2_model = hill.Hill.create_fit(d2[mask], E[mask], E0_bounds=self.E0_bounds, Emax_bounds=self.Emax_bounds,h_bounds=self.h_bounds, C_bounds=self.C_bounds, use_jacobian=use_jacobian)

        self.drug1_model = drug1_model
        self.drug2_model = drug2_model
            
        E0_1, Emax_1, h1, C1 = drug1_model.get_parameters()
        E0_2, Emax_2, h2, C2 = drug2_model.get_parameters()
        E0 = (E0_1+E0_2)/2.
        Emax = (Emax_1+Emax_2)/2.
        if (E0 < Emax):
            drug1_model.E0 = Emax_1
            drug2_model.E0 = Emax_2
            drug1_model.Emax = E0_1
            drug2_model.Emax = E0_2

            tmp = E0
            E0 = Emax
            Emax = tmp


            # Should I do E = (E-E0)/(E0-Emax) + 1
            # This puts E on the range 1 to 0.
            # But it also kills info about drugs that don't reach 0% viability

        
        logh1 = np.log(h1)
        logh2 = np.log(h2)
        logC1 = np.log(C1)
        logC2 = np.log(C2)

        self._h_21 = [] # Hill slope of drug 1, after treated by drug 2
        self._h_12 = [] # Hill slope of drug 2, after treated by drug 1
        self._C_21 = [] # EC50 of drug 1, after treated by drug 2
        self._C_12 = [] # EC50 of drug 2, after treated by drug 1
        
        zip_model = hill.Hill_2P(Emax=Emax, h_bounds=self.h_bounds, C_bounds=self.C_bounds)

        for D1, D2 in zip(d1, d2):
            # Fix d2==D2, and fit hill for D1
            mask = np.where(d2==D2)
            y2 = drug2_model.E(D2)
            zip_model.E0 = y2
            zip_model.fit(d1[mask],E[mask], use_jacobian=use_jacobian, p0=[h1,C1])
            self._h_21.append(zip_model.h)
            self._C_21.append(zip_model.C)

            # Fix d1==D1, and fit hill for D2
            mask = np.where(d1==D1)
            y1 = drug1_model.E(D1)
            zip_model.E0 = y1
            zip_model.fit(d2[mask],E[mask], use_jacobian=use_jacobian, p0=[h2,C2])
            self._h_12.append(zip_model.h)
            self._C_12.append(zip_model.C)
        
        self._h_21 = np.asarray(self._h_21)
        self._h_12 = np.asarray(self._h_12)
        self._C_21 = np.asarray(self._C_21)
        self._C_12 = np.asarray(self._C_12)

        self.synergy = self._delta_score(d1, d2, E0, Emax, h1, h2, C1, C2, self._h_21, self._h_12, self._C_21, self._C_12)

        return self.synergy

    def _delta_score(self, d1, d2, E0, Emax, h1, h2, C1, C2, h_21, h_12, C_21, C_12):
        dCh1 = (d1/C1)**h1
        dCh2 = (d2/C2)**h2
        dCh1_prime = (d1/C_21)**h_21
        dCh2_prime = (d2/C_12)**h_12
        
        AA = ((E0 + Emax*dCh2)/(1.+dCh2) + Emax*dCh1_prime) / (1+dCh1_prime)
        BB = ((E0 + Emax*dCh1)/(1.+dCh1) + Emax*dCh2_prime) / (1+dCh2_prime)
        #CC = (E0 + Emax*dCh1)/(1+dCh1) + (E0 + Emax*dCh2)/(1+dCh2) - (E0 + Emax*dCh1)/(1+dCh1)*(E0 + Emax*dCh2)/(1+dCh2) # This form expects E0==0, Emax=1
        CC = (E0 + Emax*dCh1)/(1+dCh1)*(E0 + Emax*dCh2)/(1+dCh2)

        return CC - (AA+BB)/2.
