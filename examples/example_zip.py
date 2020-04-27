import numpy as np
from matplotlib import pyplot as plt

from synergy.combination import ZIP
from synergy.combination import MuSyC
from synergy.single import Hill

from synergy.utils import plots
from synergy.utils.dose_tools import grid
from synergy.utils.data_exchange import to_synergyfinder


E0, E1, E2, E3 = 1., 0., 0., 0.
h1, h2 = 1., 1.
C1, C2 = 1e-2, 1e-1
alpha12, alpha21 = 10., 1.
gamma12, gamma21 = 1, 1

musyc = MuSyC(E0=E0, E1=E1, E2=E2, E3=E3, h1=h1, h2=h2, C1=C1, C2=C2, alpha12=alpha12, alpha21=alpha21, gamma12=gamma12, gamma21=gamma21)

npoints1 = 8
npoints2 = 10
D1, D2 = grid(1e-4, 10, 1e-4, 10,npoints1, npoints2, include_zero=True)

E = musyc.E(D1, D2)


# Build ZIP model
model = ZIP()

#Efit = E*(1+(np.random.rand(len(D1))-0.5)/10.)
Efit = E

# Output data to test in synergyfinder R package
df = to_synergyfinder(D1, D2, Efit*100)
df.to_csv("synergyfinder_comparison/zip_test_data.csv", index=None)

synergy = model.fit(D1, D2, Efit, use_jacobian=True)

print(model.drug1_model, model.drug2_model)

fig = plt.figure(figsize=(7,3))

ax=fig.add_subplot(121)
musyc.plot_colormap(D1, D2, ax=ax, title="Data")

ax=fig.add_subplot(122)
model.plot_colormap(ax=ax, title="ZIP")

plt.tight_layout()
plt.show()


if False:
    sfdf = pd.read_csv("synergyfinder_comparison/synergyfinder_output.csv", index_col=0)
    sfdf.sort_values(by=["d2","d1"], inplace=True)

    print ("Correlation between this package and synergyfinder = %f"%np.corrcoef(model.synergy, sfdf['synergy'])[0,1])
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    ax.scatter(model.synergy*100, sfdf['synergy'])
    ax.set_xlabel("Python ZIP synergy (x100)")
    ax.set_ylabel("R synergyfinder ZIP synergy")
    ax.set_title("Comparison with synergyfinder")
    plt.tight_layout()
    #plt.savefig("synergyfinder_comparison/python_vs_synergyfinder.pdf")
    #plt.close()
    #plt.show()