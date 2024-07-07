try:
    import pandas as pd

    pandas_installed = True
except ImportError:
    pandas_installed = False


def to_synergyfinder(
    d1,
    d2,
    E,
    d1_name: str = "drug1",
    d2_name: str = "drug2",
    d1_unit: str = "uM",
    d2_unit: str = "uM",
):
    """Converts the data to the format used by synergyfinder.

    :param ArrayLike d1: The concentrations of the first drug.
    :param ArrayLike d2: The concentrations of the second drug.
    :param ArrayLike E: The observed responses.
    :param str d1_name: The name of the first drug.
    :param str d2_name: The name of the second drug.
    :param str d1_unit: The unit of the first drug.
    :param str d2_unit: The unit of the second drug.
    :return Union[str, pandas.DataFrame]: The data in the format used by synergyfinder. This will be a pandas DataFrame
        if pandas is installed, otherwise it will be a string.
    """
    if not pandas_installed:
        ret = [
            ",".join(["block_id", "drug_col", "drug_row", "conc_c", "conc_r", "response", "conc_c_unit", "conc_r_unit"])
        ]
        for D1, D2, EE in zip(d1, d2, E):
            ret.append(",".join(["1", d1_name, d2_name, "%f" % D1, "%f" % D2, "%f" % EE, d1_unit, d2_unit]))
        return "\n".join(ret)
    else:
        return pd.DataFrame(
            dict(
                block_id=1,
                drug_col=d1_name,
                drug_row=d2_name,
                conc_c=d1,
                conc_r=d2,
                response=E,
                conc_c_unit=d1_unit,
                conc_r_unit=d2_unit,
            )
        )
