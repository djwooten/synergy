# Parametric Models
from .musyc import MuSyC

# Nonparametric models
from .bliss import Bliss
from .hsa import HSA
from .loewe import Loewe
from .combination_index import CombinationIndex
from .schindler import Schindler

######################

# No higher order synergy models described

# Parametric Models
#from .zimmer import Zimmer # They do pairwise calculations of 2-drug combos
#from .braid import BRAID

# Nonparametric models
#from .zero_interaction_potency import ZIP # This could almost certainly be done by averaging over 3+ Hill equation slices, but isn't explicitly described