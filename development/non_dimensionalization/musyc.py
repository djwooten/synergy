import numpy as np
from sympy import *

# MuSyC
d1, d2 = symbols("d1 d2")
logh1, logh2 = symbols("logh1 logh2")
E0, E1, E2, E3 = symbols("E0 E1 E2 E3")
logC1, logC2 = symbols("logC1 logC2")
r1, r2 = symbols("r1 r2")
logalpha12, logalpha21 = symbols("logalpha12 logalpha21")

C1 = exp(logC1)
C2 = exp(logC2)
h1 = exp(logh1)
h2 = exp(logh2)
alpha12 = exp(logalpha12)
alpha21 = exp(logalpha21)

r1r = r1 * C1**h1
r2r = r2 * C2**h2

U = (
    r1r
    * r2r
    * (r1 * (alpha21 * d1) ** h1 + r1r + r2 * (alpha12 * d2) ** h2 + r2r)
    / (
        d1**h1 * r1**2 * r2 * (alpha12 * d2) ** h2 * (alpha21 * d1) ** h1
        + d1**h1 * r1**2 * r2r * (alpha21 * d1) ** h1
        + d1**h1 * r1 * r1r * r2 * (alpha12 * d2) ** h2
        + d1**h1 * r1 * r1r * r2r
        + d1**h1 * r1 * r2 * r2r * (alpha12 * d2) ** h2
        + d1**h1 * r1 * r2r**2
        + d2**h2 * r1 * r1r * r2 * (alpha21 * d1) ** h1
        + d2**h2 * r1 * r2**2 * (alpha12 * d2) ** h2 * (alpha21 * d1) ** h1
        + d2**h2 * r1 * r2 * r2r * (alpha21 * d1) ** h1
        + d2**h2 * r1r**2 * r2
        + d2**h2 * r1r * r2**2 * (alpha12 * d2) ** h2
        + d2**h2 * r1r * r2 * r2r
        + r1 * r1r * r2r * (alpha21 * d1) ** h1
        + r1r**2 * r2r
        + r1r * r2 * r2r * (alpha12 * d2) ** h2
        + r1r * r2r**2
    )
)

A1 = (
    r1
    * r2r
    * (d1**h1 * r1 * (alpha21 * d1) ** h1 + d1**h1 * r1r + d1**h1 * r2r + d2**h2 * r2 * (alpha21 * d1) ** h1)
    / (
        d1**h1 * r1**2 * r2 * (alpha12 * d2) ** h2 * (alpha21 * d1) ** h1
        + d1**h1 * r1**2 * r2r * (alpha21 * d1) ** h1
        + d1**h1 * r1 * r1r * r2 * (alpha12 * d2) ** h2
        + d1**h1 * r1 * r1r * r2r
        + d1**h1 * r1 * r2 * r2r * (alpha12 * d2) ** h2
        + d1**h1 * r1 * r2r**2
        + d2**h2 * r1 * r1r * r2 * (alpha21 * d1) ** h1
        + d2**h2 * r1 * r2**2 * (alpha12 * d2) ** h2 * (alpha21 * d1) ** h1
        + d2**h2 * r1 * r2 * r2r * (alpha21 * d1) ** h1
        + d2**h2 * r1r**2 * r2
        + d2**h2 * r1r * r2**2 * (alpha12 * d2) ** h2
        + d2**h2 * r1r * r2 * r2r
        + r1 * r1r * r2r * (alpha21 * d1) ** h1
        + r1r**2 * r2r
        + r1r * r2 * r2r * (alpha12 * d2) ** h2
        + r1r * r2r**2
    )
)

A2 = (
    r1r
    * r2
    * (d1**h1 * r1 * (alpha12 * d2) ** h2 + d2**h2 * r1r + d2**h2 * r2 * (alpha12 * d2) ** h2 + d2**h2 * r2r)
    / (
        d1**h1 * r1**2 * r2 * (alpha12 * d2) ** h2 * (alpha21 * d1) ** h1
        + d1**h1 * r1**2 * r2r * (alpha21 * d1) ** h1
        + d1**h1 * r1 * r1r * r2 * (alpha12 * d2) ** h2
        + d1**h1 * r1 * r1r * r2r
        + d1**h1 * r1 * r2 * r2r * (alpha12 * d2) ** h2
        + d1**h1 * r1 * r2r**2
        + d2**h2 * r1 * r1r * r2 * (alpha21 * d1) ** h1
        + d2**h2 * r1 * r2**2 * (alpha12 * d2) ** h2 * (alpha21 * d1) ** h1
        + d2**h2 * r1 * r2 * r2r * (alpha21 * d1) ** h1
        + d2**h2 * r1r**2 * r2
        + d2**h2 * r1r * r2**2 * (alpha12 * d2) ** h2
        + d2**h2 * r1r * r2 * r2r
        + r1 * r1r * r2r * (alpha21 * d1) ** h1
        + r1r**2 * r2r
        + r1r * r2 * r2r * (alpha12 * d2) ** h2
        + r1r * r2r**2
    )
)

A12 = 1 - (U + A1 + A2)

f = U * E0 + A1 * E1 + A2 * E2 + A12 * E3

# I am not fitting r1 or r2
# fr1 = diff(f,r1)
# fr2 = diff(f,r2)

frlogC1 = diff(f, logC1)
frlogC2 = diff(f, logC2)
flogh1 = diff(f, logh1)
flogh2 = diff(f, logh2)
fE0 = U
fE1 = A1
fE2 = A2
fE3 = A12
flogalpha12 = diff(f, logalpha12)
flogalpha21 = diff(f, logalpha21)

outfile = open("musyc_jacobian_correct.txt", "w")
for name, eq in zip(
    ["j_logh1", "j_logh2", "j_logC1", "j_logC2", "j_logalpha12", "j_logalpha21"],
    [flogh1, flogh2, frlogC1, frlogC2, flogalpha12, flogalpha21],
):
    outfile.write("# ********** %s ********\n\n" % name[2:])
    outfile.write("%s = %s\n\n" % (name, repr(eq)))
outfile.close()
