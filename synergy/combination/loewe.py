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

from ..single import Hill
from .. import utils
from .nonparametric_base import DoseDependentModel

# Used for delta variant of Loewe
from scipy.optimize import minimize_scalar

class Loewe(DoseDependentModel):
    """Loewe Additivity Synergy
    
    Loewe's model of drug combination additivity expects a linear tradeoff, such that withholding X parts of drug 1 can be compensated for by adding Y parts of drug 2. In Loewe's model, X and Y are constant (e.g., withholding 5X parts of drug 1 will be compensated for with 5Y parts of drug 2.)

    We support two methods to calculate Loewe synergy. For each method, synergy and antagonism are defined and interpreted differently (see "synergy" below).
    
    The vairant may be set as 
    
        model = Loewe(..., variant="CI") (default)
        or
        model = Loewe(..., variant="delta")

    Variant "CI" (default) calculates synergy using an equation equivalent to combination index, but unlike CI, Loewe allows arbitrary single-drug models.

    Variant "delta" calculates an expected dose response surface, and defines synergy as the difference between the measured values and the expected values. This variant REQUIRES single-drugs be fitted using a Hill model.

    synergy : array_like, float
        if variant=="CI":
            [0,1)=synergism, (1,inf)=antagonism
        if variant=="delta":
            (0,inf)=synergism, (-inf,0)=antagonism
    """

    def __init__(self, h1_bounds=(0,np.inf), h2_bounds=(0,np.inf),  \
            C1_bounds=(0,np.inf), C2_bounds=(0,np.inf),             \
            E0_bounds=(-np.inf,np.inf), E1_bounds=(-np.inf,np.inf), \
            E2_bounds=(-np.inf,np.inf), variant="CI"):
        
        super().__init__(h1_bounds=h1_bounds, h2_bounds=h2_bounds, C1_bounds=C1_bounds, C2_bounds=C2_bounds, E0_bounds=E0_bounds, E1_bounds=E1_bounds, E2_bounds=E2_bounds)
        
        self.variant = "CI"
        if isinstance(variant, str):
            self.variant = variant.lower()

    def fit(self, d1, d2, E, drug1_model=None, drug2_model=None, **kwargs):

        d1 = np.asarray(d1)
        d2 = np.asarray(d2)
        E = np.asarray(E)
        super().fit(d1,d2,E, drug1_model=drug1_model, drug2_model=drug2_model, **kwargs)

        drug1_model = self.drug1_model
        drug2_model = self.drug2_model


        if self.variant.startswith("delta"):
            self.reference = self._E_reference(d1, d2, drug1_model, drug2_model)
            self.synergy = self._get_synergy_delta(d1, d2, E, drug1_model, drug2_model)
        else:
            self.synergy = self._get_synergy_CI(d1, d2, E, drug1_model, drug2_model)

        return self.synergy
    
    def _get_synergy_delta(self, d1, d2, E, drug1_model, drug2_model):
        
        # Save reference and synergy
        if self.reference is None:
            self.reference = self._E_reference(d1, d2, drug1_model, drug2_model)
        synergy = self.reference - E

        if hasattr(synergy,"__iter__"): synergy[(d1==0) | (d2==0)] = 0
        elif d1==0 or d2==0: synergy=0
        return synergy

    def _get_synergy_CI(self, d1, d2, E, drug1_model, drug2_model):
        with np.errstate(divide='ignore', invalid='ignore'):
            d1_alone = drug1_model.E_inv(E)
            d2_alone = drug2_model.E_inv(E)
            synergy = d1/d1_alone + d2/d2_alone
        
        if hasattr(synergy,"__iter__"): synergy[(d1==0) | (d2==0)] = 1
        elif d1==0 or d2==0: synergy=1
        return synergy

    def _get_single_drug_classes(self):
        # The delta model ONLY works when single drugs are fit with Hill equation
        #if self.variant.startswith("delta"):
        #    return Hill, Hill

        # Otherwise, default to Hill, but allow anything
        return Hill, None

    def _fit_Loewe_reference(self, X_r, X_c, drug1_model, drug2_model):
        """Calculates a reference (null) model for Loewe for drug1 and drug2 at concentrations X_r and X_c

        drug1_model and drug2_model MUST be some form of Hill equation
        
        Contributed by Mark Russo at Bristol Myers Squibb

        Returns scipy.optimize.minimize_scalar object
        """
        # Unpack single drug parameters
        #pa_r = drug1_model.get_parameters()
        #pa_c = drug2_model.get_parameters()
        #[Emin_r, Emax_r, h_r, m_r] = pa_r
        #[Emin_c, Emax_c, h_c, m_c] = pa_c

        # Compute the bounds within which Y_Loewe is valid.
        # Any value outside these bounds causes algorithm to take a root of a negative.
        #bounds_r = [Emin_r, Emax_r]; bounds_r.sort()
        #bounds_c = [Emin_c, Emax_c]; bounds_c.sort()
        #bounds   = [max(bounds_r[0], bounds_c[0]), min(bounds_r[1], bounds_c[1])]


        
        bounds1 = sorted([drug1_model.E0, drug1_model.Emax])
        bounds2 = sorted([drug2_model.E0, drug2_model.Emax])

        bounds   = [max(bounds1[0], bounds2[0]), min(bounds1[1], bounds2[1])]
        


        # Quadratic function with a minimum at CI=1
        #f  = lambda Y: ((X_r/(m_r*(((Y-Emin_r)/(Emax_r-Y))**(1/h_r))) + X_c/(m_c*(((Y-Emin_c)/(Emax_c-Y))**(1/h_c)))) - 1.0)**2

        f = lambda Y : ((X_r / drug1_model.E_inv(Y)) + (X_c / drug2_model.E_inv(Y)) -1.0)**2

        # Perform constrained minimization to find Y as close as possible to CI == 1 and return result.
        return minimize_scalar(f, method='bounded', bounds=bounds)

    def _E_reference(self, d1, d2, drug1_model, drug2_model):
        """Calculates a reference (null) model for Loewe for drug1 and drug2.
        
        Contributed by Mark Russo at Bristol Myers Squibb
        Modified by David Wooten
        """

        # Compute the reference model
        with np.errstate(divide='ignore', invalid='ignore'):
            ref  = 0*d1

            #pa1 = drug1_model.get_parameters()
            #pa2 = drug2_model.get_parameters()

            Emax_1 = drug1_model.Emax
            Emax_2 = drug2_model.Emax

            #weakest_E = max(pa1[1], pa2[1])
            weakest_E = max(Emax_1, Emax_2)

            stronger_drug = drug1_model
            #if pa1[1] > pa2[1]: # pa[1] is Emax
            if Emax_1 > Emax_2: # pa[1] is Emax
                stronger_drug = drug2_model

            # Loewe becomes undefined for effects past the weaker drug's Emax
            # We implement several variants to handle this case:
            #  1) variant="delta" - this will set E_reference to weakest_E
            #  2) variant="delta_HSA" - this will set E_r to min(E1, E2)
            #  3) variant="delta_nan" - this will set E_r to nan
            #  4) variant="synergyfinder" - sets E_r to stronger.E(d1+d2)
            if self.variant=="delta": option=1
            elif self.variant=="delta_hsa": option=2
            elif self.variant=="delta_nan": option=3
            elif self.variant=="delta_synergyfinder": option=4
            else: option=1

            for i in range(len(ref)):
                X1, X2 = d1[i], d2[i]
                E1, E2 = drug1_model.E(X1), drug2_model.E(X2)
                
                # No drug
                if X1 == 0.0 and X2 == 0.0:
                    ref[i] = 0.5 * (E2 + E1)

                # Single drugs
                elif X2 == 0:
                    ref[i] = E1
                elif X1 == 0:
                    ref[i] = E2

                # If the combo E is stronger than the weaker drug is capable of, don't bother trying to fit
                elif E1 < weakest_E or E2 < weakest_E:
                    if option==1:
                        ref[i] = weakest_E
                    elif option==2:
                        ref[i] = min(E1,E2)
                    elif option==3:
                        ref[i] = np.nan
                    elif option==4:
                        ref[i] = stronger_drug.E(X1+X2)
                    else:
                        ref[i] = np.nan
                else:
                    # Numerically solve the value for Loewe
                    res = self._fit_Loewe_reference(X1, X2, drug1_model, drug2_model)
                    if res.success:
                        ref[i] = res.x
                    else:
                        ref[i] = min(E2, E1)
        return ref

    def plot_heatmap(self, cmap="PRGn", neglog=None, center_on_zero=True, **kwargs):
        if neglog is None:
            if self.variant=="delta":
                neglog=False
            else:
                neglog=True
        super().plot_heatmap(cmap=cmap, neglog=neglog, center_on_zero=center_on_zero, **kwargs)

    def plot_surface_plotly(self, cmap="PRGn", neglog=None, center_on_zero=True, **kwargs):
        if neglog is None:
            if self.variant=="delta":
                neglog=False
            else:
                neglog=True
        super().plot_surface_plotly(cmap=cmap, neglog=neglog, center_on_zero=center_on_zero, **kwargs)