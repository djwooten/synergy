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
try:
    import pandas as pd
    pandas_installed = True
except:
    pandas_installed = False

def to_synergyfinder(d1, d2, E, d1_name="drug1", d2_name="drug2", d1_unit="uM", d2_unit="uM"):
    if not pandas_installed:
        ret = [",".join(["block_id", "drug_col", "drug_row", "conc_c", "conc_r", "response", "conc_c_unit", "conc_r_unit"])]
        for D1, D2, EE in zip(d1, d2, E):
            ret.append(",".join(["1", d1_name, d2_name, "%f"%D1, "%f"%D2,"%f"%EE, d1_unit, d2_unit]))
        return "\n".join(ret)
    else:
        return pd.DataFrame(dict(block_id=1, drug_col=d1_name, drug_row=d2_name, conc_c=d1, conc_r=d2, response=E, conc_c_unit=d1_unit, conc_r_unit=d2_unit))

