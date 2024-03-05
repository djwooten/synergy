import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from synergy.combination import MuSyC
from synergy.single import Hill
from synergy.combination import BRAID

from synergy.utils import plots
from synergy.utils.dose_tools import make_dose_grid


def create_data(i):
    if i == 1:
        E0, E1, E2, E3 = 1.0, 0.0, 0.0, 0.0
        h1, h2 = 1.0, 1.0
        C1, C2 = 1e-2, 1e-1
        alpha12, alpha21 = 10.0, 1.0
        gamma12, gamma21 = 1, 1
    elif i == 2:
        E0, E1, E2, E3 = 1.0, 0.0, 0.0, 0.0
        h1, h2 = 1.0, 3.0
        C1, C2 = 1e-2, 1e-1
        alpha12, alpha21 = 0.1, 1.0
        gamma12, gamma21 = 1, 1
    elif i == 3:
        E0, E1, E2, E3 = 1.0, 0.4, 0.2, 0.0
        h1, h2 = 2.0, 1.0
        C1, C2 = 1e-2, 1e-1
        alpha12, alpha21 = 1.0, 1.0
        gamma12, gamma21 = 1, 1

    musyc = MuSyC(
        E0=E0,
        E1=E1,
        E2=E2,
        E3=E3,
        h1=h1,
        h2=h2,
        C1=C1,
        C2=C2,
        alpha12=alpha12,
        alpha21=alpha21,
        gamma12=gamma12,
        gamma21=gamma21,
    )

    npoints1 = 8
    npoints2 = 10
    D1, D2 = make_dose_grid(1e-4, 10, 1e-4, 10, npoints1, npoints2, include_zero=True)

    E = musyc.E(D1, D2)
    return D1, D2, E


def get_braidrm(variant, i):
    if variant == "both":
        variant = "full"
    return pd.read_csv("%d/%s_results.csv" % (i, variant), index_col=0)


def trans_log(v, variable):
    # if variable in ["h1","h2","C1","C2","delta"]:
    #   return np.log(v)
    return v


python_results = dict()
rm_results = dict()
pyrss = dict()
rmrss = dict()
pyr2 = dict()
rmr2 = dict()

for i in [1, 2, 3]:
    D1, D2, E = create_data(i)
    python_results[i] = dict()
    rm_results[i] = dict()

    pyrss[i] = dict()
    rmrss[i] = dict()
    pyr2[i] = dict()
    rmr2[i] = dict()

    for variant in ["kappa", "delta", "both"]:
        python_results[i][variant] = dict()
        rm_results[i][variant] = dict()

        # Build and fit BRAID model
        braid_model = BRAID(mode=variant)
        braid_model.fit(D1, D2, E, bootstrap_iterations=100)
        results = braid_model.get_parameters()
        braidrm_fits = get_braidrm(variant, i)

        kappa = 0
        delta = 1
        if "kappa" in braidrm_fits.index:
            kappa = braidrm_fits.loc["kappa", "best"]
        if "delta" in braidrm_fits.index:
            kappa = braidrm_fits.loc["delta", "best"]
        braidrm_model = BRAID(
            mode=variant,
            E0=braidrm_fits.loc["E0", "best"],
            E1=braidrm_fits.loc["E1", "best"],
            E2=braidrm_fits.loc["E2", "best"],
            E3=braidrm_fits.loc["E3", "best"],
            h1=braidrm_fits.loc["h1", "best"],
            h2=braidrm_fits.loc["h2", "best"],
            C1=braidrm_fits.loc["C1", "best"],
            C2=braidrm_fits.loc["C2", "best"],
            kappa=kappa,
            delta=delta,
        )
        braidrm_model._score(D1, D2, E)

        rmrss[i][variant] = braidrm_model.sum_of_squares_residuals
        pyrss[i][variant] = braid_model.sum_of_squares_residuals
        rmr2[i][variant] = braidrm_model.r_squared
        pyr2[i][variant] = braid_model.r_squared

        for variable in results.keys():
            rm_results[i][variant][variable] = list(braidrm_fits.loc[variable])
            l = []
            l.append(results[variable][1][0])  # lower
            l.append(results[variable][0])  # Best fit
            l.append(results[variable][1][1])  # Upper

            python_results[i][variant][variable] = l


variables = ["E0", "E1", "E2", "E3", "h1", "h2", "C1", "C2", "kappa", "delta"]

for _model in [1, 2, 3]:
    # for _model in [1,]:
    fig = plt.figure(figsize=(8, 7))
    nrows = 4
    ncols = 3
    delta_x = 2
    for _i, variable in enumerate(variables):
        ax = fig.add_subplot(nrows, ncols, _i + 1)
        ax.set_title(variable)
        rmx = []
        rmy = []
        rmyl = []
        rmyu = []

        pyx = []
        pyy = []
        pyyl = []
        pyyu = []

        xticks = []
        xticklabels = []

        for _x, variant in enumerate(["kappa", "delta", "both"]):
            xticks.append(_x * delta_x)
            xticklabels.append("variant=\n%s" % variant)
            if variable in python_results[_model][variant]:
                pyres = python_results[_model][variant][variable]
                rmres = rm_results[_model][variant][variable]

                pyx.append((_x - 1 / 10) * delta_x)
                rmx.append((_x + 1 / 10) * delta_x)

                pyyl.append(trans_log(pyres[0], variable))
                pyy.append(trans_log(pyres[1], variable))
                pyyu.append(trans_log(pyres[2], variable))

                rmyl.append(trans_log(rmres[0], variable))
                rmy.append(trans_log(rmres[1], variable))
                rmyu.append(trans_log(rmres[2], variable))
        rmy = np.asarray(rmy)
        rmyl = rmy - np.asarray(rmyl)
        rmyu = np.asarray(rmyu) - rmy

        pyy = np.asarray(pyy)
        pyyl = pyy - np.asarray(pyyl)
        pyyu = np.asarray(pyyu) - pyy

        pyyerr = np.vstack([pyyl, pyyu])
        rmyerr = np.vstack([rmyl, rmyu])
        ax.scatter(pyx, pyy)
        ax.errorbar(pyx, pyy, yerr=pyyerr, fmt="none")

        ax.scatter(rmx, rmy)
        ax.errorbar(rmx, rmy, yerr=rmyerr, fmt="none")
        ax.set_xlim(-0.4, 4.4)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)

    # Add sum of squares residuals
    ax = fig.add_subplot(nrows, ncols, 11)
    pyy = []
    rmy = []
    pyx = []
    rmx = []
    for _x, variant in enumerate(["kappa", "delta", "both"]):
        pyx.append((_x - 1 / 10) * delta_x)
        rmx.append((_x + 1 / 10) * delta_x)
        pyy.append(pyrss[_model][variant])
        rmy.append(rmrss[_model][variant])
    ax.bar(pyx, pyy, width=delta_x / 8, label="synergy")
    ax.bar(rmx, rmy, width=delta_x / 8, label="braidrm")
    ax.set_xlim(-0.4, 4.4)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_title("RSS")
    ax.legend()

    # Add R^2
    ax = fig.add_subplot(nrows, ncols, 12)
    pyy = []
    rmy = []
    pyx = []
    rmx = []
    for _x, variant in enumerate(["kappa", "delta", "both"]):
        pyx.append((_x - 1 / 10) * delta_x)
        rmx.append((_x + 1 / 10) * delta_x)
        pyy.append(pyr2[_model][variant])
        rmy.append(rmr2[_model][variant])
    ax.bar(pyx, pyy, width=delta_x / 8)
    ax.bar(rmx, rmy, width=delta_x / 8)
    ax.set_xlim(-0.4, 4.4)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_title("R^2")

    plt.tight_layout()
    # plt.show()
    plt.savefig("braid_comparison_%d.pdf" % _model)
    plt.close()
