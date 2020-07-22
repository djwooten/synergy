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
import inspect
import warnings

from ..single import Hill, Hill_2P
from .nonparametric_base import DoseDependentModel
from .. import utils

class ZIP(DoseDependentModel):
    """The Zero Interaction Potency (ZIP) model (doi: 10.1016/j.csbj.2015.09.001). This model is based on the multiplicative survival principal (i.e., Bliss). All across the dose-response surface, it fits Hill-equations either holding d1==constant or d2==constant.
    
    ZIP quantifies changes in the EC50 (C) and Hill-slope (h) of each drug, as the other is increased. For instance, at a given d1==D1, d2==D2, ZIP fits two Hill equations: one for drug1 (fixing d2==D2), and one for drug2 (fixing d1==D1). These Hill equations are averaged to get a fit value of E (y_c in their paper) at these doses. This is then subtracted from the expected result (y_zip in their paper) obtained by assuming these Hill equations have equivalent h and C to their single-drug counterparts. This difference (delta in their paper) becomes the metric of synergy.

    ZIP models store these delta values as model._synergy, but also store the Hill equation fits for drug1 and drug2 across the whole surface, allowing investigation of how h and C change across the surface

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
    def __init__(self, E0_bounds=(0,1.5), E1_bounds=(0,1.5), E2_bounds=(0,1.5), h1_bounds=(0,np.inf), C1_bounds=(0,np.inf), h2_bounds=(0,np.inf), C2_bounds=(0,np.inf), synergyfinder=False):

        super().__init__(h1_bounds=h1_bounds, h2_bounds=h2_bounds, C1_bounds=C1_bounds, C2_bounds=C2_bounds, E0_bounds=E0_bounds, E1_bounds=E1_bounds, E2_bounds=E2_bounds)

        self.synergyfinder = synergyfinder

        self._h_21 = []
        self._h_12 = []
        self._C_21 = []
        self._C_12 = []
        self._Emax_21 = []
        self._Emax_12 = []
        
    def _get_single_drug_classes(self):
        return Hill, Hill

    def fit(self, d1, d2, E, drug1_model=None, drug2_model=None, use_jacobian=True, **kwargs):
    
        d1 = np.asarray(d1)
        d2 = np.asarray(d2)
        E = np.asarray(E)
        super().fit(d1,d2,E, drug1_model=drug1_model, drug2_model=drug2_model, ues_jacobian=use_jacobian, **kwargs)

        drug1_model = self.drug1_model
        drug2_model = self.drug2_model
 
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
        
        logh1 = np.log(h1)
        logh2 = np.log(h2)
        logC1 = np.log(C1)
        logC2 = np.log(C2)

        self._h_21 = [] # Hill slope of drug 1, after treated by drug 2
        self._h_12 = [] # Hill slope of drug 2, after treated by drug 1
        self._C_21 = [] # EC50 of drug 1, after treated by drug 2
        self._C_12 = [] # EC50 of drug 2, after treated by drug 1
        self._Emax_21 = []
        self._Emax_12 = []
        
        if self.synergyfinder:
            zip_model = _Hill_3P(Emax_bounds=(-1e-6,1e-6))
        else:
            zip_model = _Hill_3P(Emax_bounds=(0,1.5))
        

        for D1, D2 in zip(d1, d2):
            # Fix d2==D2, and fit hill for D1
            mask = np.where(d2==D2)
            y2 = drug2_model.E(D2)
            zip_model.E0 = y2
            zip_model.fit(d1[mask],E[mask], use_jacobian=use_jacobian, p0=[Emax_1,h1,C1])
            self._h_21.append(zip_model.h)
            self._C_21.append(zip_model.C)
            self._Emax_21.append(zip_model.Emax)

            # Fix d1==D1, and fit hill for D2
            mask = np.where(d1==D1)
            y1 = drug1_model.E(D1)
            zip_model.E0 = y1
            zip_model.fit(d2[mask],E[mask], use_jacobian=use_jacobian, p0=[Emax_2,h2,C2])
            self._h_12.append(zip_model.h)
            self._C_12.append(zip_model.C)
            self._Emax_12.append(zip_model.Emax)
        
        self._h_21 = np.asarray(self._h_21)
        self._h_12 = np.asarray(self._h_12)
        self._C_21 = np.asarray(self._C_21)
        self._C_12 = np.asarray(self._C_12)
        self._Emax_21 = np.asarray(self._Emax_21)
        self._Emax_12 = np.asarray(self._Emax_12)

        self.synergy = self._delta_score(d1, d2, E0, self.drug1_model.Emax, self.drug2_model.Emax, h1, h2, C1, C2, self._Emax_21, self._Emax_12, self._h_21, self._h_12, self._C_21, self._C_12)

        mask = np.where((d1==0) | (d2==0))
        self.synergy[mask]=0
        self.synergy

        return self.synergy

    def _delta_score(self, d1, d2, E0, E1, E2, h1, h2, C1, C2, Emax_21, Emax_12, h_21, h_12, C_21, C_12):

        single_drug_1 = E0 + (E1-E0) * np.power(d1,h1) / (np.power(C1,h1) + np.power(d1,h1))

        single_drug_2 = E0 + (E2-E0) * np.power(d2,h2) / (np.power(C2,h2) + np.power(d2,h2))

        zip_fit_1 = single_drug_2 + (Emax_21-single_drug_2) * np.power(d1,h_21) / (np.power(C_21,h_21) + np.power(d1,h_21))

        zip_fit_2 = single_drug_1 + (Emax_12-single_drug_1) * np.power(d2,h_12) / (np.power(C_12,h_12) + np.power(d2,h_12))
        
        zip_fit = (zip_fit_1+zip_fit_2)/2.
        zip_ind = single_drug_1*single_drug_2

        self.reference = zip_ind

        return zip_ind-zip_fit



class _Hill_3P(Hill):
    def __init__(self, E0=1, Emax=0, h=None, C=None, Emax_bounds=(-np.inf, np.inf), h_bounds=(0,np.inf), C_bounds=(0,np.inf)):
        super().__init__(h=h, C=C, E0=E0, Emax=Emax, Emax_bounds=Emax_bounds, h_bounds=h_bounds, C_bounds=C_bounds)

        self.fit_function = lambda d, Emax, logh, logC: self._model(d, self.E0, Emax, np.exp(logh), np.exp(logC))

        self.jacobian_function = lambda d, Emax, logh, logC: self._model_jacobian(d, Emax, logh, logC)

        self.bounds = tuple(zip(self.Emax_bounds, self.logh_bounds, self.logC_bounds))

    def _model_jacobian(self, d, Emax, logh, logC):
        dh = d**(np.exp(logh))
        Ch = (np.exp(logC))**(np.exp(logh))
        logd = np.log(d)
        E0 = self.E0

        jEmax = dh/(Ch+dh)

        jC = (E0-Emax)*dh*np.exp(logh+logC)*(np.exp(logC))**(np.exp(logh)-1) / ((Ch+dh)*(Ch+dh))

        jh = (Emax-E0)*dh*np.exp(logh) * ((Ch+dh)*logd - (logC*Ch + logd*dh)) / ((Ch+dh)*(Ch+dh))
        
        jac = np.hstack((jEmax.reshape(-1,1), jh.reshape(-1,1), jC.reshape(-1,1)))
        jac[np.isnan(jac)]=0
        return jac

    def _get_initial_guess(self, d, E, p0=None):

        if p0 is None:
            p0 = [np.nanmin(E), 1, np.median(d)]
            
        p0 = list(self._transform_params_to_fit(p0))
        utils.sanitize_initial_guess(p0, self.bounds)
        
        return p0

    def get_parameters(self):
        """Gets the model's parameters
        
        Returns
        ----------
        parameters : tuple
            (Emax, h, C)
        """
        return (self.Emax, self.h, self.C)

    def _set_parameters(self, popt):
        Emax, h, C = popt
        
        self.Emax = Emax
        self.h = h
        self.C = C

    def _transform_params_from_fit(self, params):
        return params[0], np.exp(params[1]), np.exp(params[2])

    def _transform_params_to_fit(self, params):
        return params[0], np.log(params[1]), np.log(params[2])

    def __repr__(self):
        if not self._is_parameterized(): return "Hill_3P()"
        
        return "Hill_3P(E0=%0.2f, Emax=%0.2f, h=%0.2f, C=%0.2e)"%(self.E0, self.Emax, self.h, self.C)