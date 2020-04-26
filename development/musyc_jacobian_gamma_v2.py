from sympy import *
import numpy as np


# MuSyC
d1, d2 = symbols('d1 d2')
logh1, logh2 = symbols('logh1 logh2')
E0, E1, E2, E3 = symbols('E0 E1 E2 E3')
logC1, logC2 = symbols('logC1 logC2')
r1, r2 = symbols('r1 r2')
logalpha12, logalpha21 = symbols('logalpha12 logalpha21')
loggamma12, loggamma21 = symbols('loggamma12 loggamma21')

C1 = exp(logC1)
C2 = exp(logC2)
h1 = exp(logh1)
h2 = exp(logh2)
alpha12 = exp(logalpha12)
alpha21 = exp(logalpha21)
gamma12 = exp(loggamma12)
gamma21 = exp(loggamma21)


r1r = r1*C1**h1
r2r = r2*C2**h2

row1 = [-(r1*d1**h1+r2*d2**h2), r1r, r2r, 0]
row2 = [r1*d1**h1, -(r1r+r2**gamma12*(alpha12*d2)**(gamma12*h2)), 0, r2r**gamma12]
row3 = [r2*d2**h2, 0, -(r1**gamma21*(alpha21*d1)**(gamma21*h1)+r2r), r1r**gamma21]
row4 = [1,1,1,1]

M = Matrix([row1, row2, row3, row4])

UA = M.inv()*Matrix([0,0,0,1])

U = simplify(UA[0,0])
A1 = simplify(UA[1,0])
A2 = simplify(UA[2,0])
A12 = simplify(UA[3,0])
    
f = U*E0 + A1*E1 + A2*E2 + A12*E3

# I am not fitting r1 or r2
#fr1 = diff(f,r1)
#fr2 = diff(f,r2)

frlogC1 = diff(f,logC1)
frlogC2 = diff(f,logC2)
flogh1 = diff(f,logh1)
flogh2 = diff(f,logh2)
fE0 = U
fE1 = A1
fE2 = A2
fE3 = A12
flogalpha12 = diff(f, logalpha12)
flogalpha21 = diff(f, logalpha21)
floggamma12 = diff(f, loggamma12)
floggamma21 = diff(f, loggamma21)

#frlogC1_2 = simplify(frlogC1)
#frlogC2_2 = simplify(frlogC2)
#flogh1_2 = simplify(flogh1) # flogh1 simplfy takes FOREVER!
#flogh1_2 = flogh12
#flogh2_2 = simplify(flogh2)
#flogalpha12_2 = simplify(flogalpha12)
#flogalpha21_2 = simplify(flogalpha21)
#floggamma12_2 = simplify(floggamma12)
#floggamma21_2 = simplify(floggamma21)

outfile = open("musyc_jacobian_gamma_v2.txt","w")
for name, eq in zip(["j_logh1", "j_logh2", "j_logC1", "j_logC2", "j_logalpha12", "j_logalpha21", "j_loggamma12", "j_loggamma21", "j_E0", "j_E1", "j_E2", "j_E3"], [flogh1, flogh2, frlogC1, frlogC2, flogalpha12, flogalpha21, floggamma12, floggamma21, fE0, fE1, fE2, fE3]):
    outfile.write("# ********** %s ********\n\n"%name[2:])
    outfile.write("%s = %s\n\n"%(name, repr(eq)))
outfile.close()


outfile = open("musyc_jacobian_gamma_v2.txt","w")
for name, eq in zip(["j_logh1", "j_logh2", "j_logC1", "j_logC2", "j_logalpha12", "j_logalpha21", "j_loggamma12", "j_loggamma21", "j_E0", "j_E1", "j_E2", "j_E3"], [flogh1_2, flogh21_2, frlogC11_2, frlogC21_2, flogalpha121_2, flogalpha211_2, floggamma121_2, floggamma211_2, fE0, fE1, fE2, fE3]):
    outfile.write("# ********** %s ********\n\n"%name[2:])
    outfile.write("%s = %s\n\n"%(name, repr(eq)))
outfile.close()