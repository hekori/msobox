# -*- coding: utf-8 -*-

"""
===============================================================================

bimolkat example for integration

===============================================================================
"""

# system imports
import numpy as np
import matplotlib.pyplot as pl

# local imports
from msobox.ind.explicit_euler import ExplicitEuler
from msobox.ind.implicit_euler import ImplicitEuler
from msobox.ind.rk4classic import RK4Classic

from msobox.mf.tapenade import Differentiator
from msobox.mf.fortran import BackendFortran

# setting print options to print all array elements
np.set_printoptions(threshold=np.nan)

def get_dir_path():
    """return script directory"""
    import inspect, os
    return os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

"""
===============================================================================
"""

# differentiate model functions
# Differentiator(get_dir_path() + "/fortran/bimolkat/ffcn.f")
backend_fortran = BackendFortran(get_dir_path() + "/fortran/bimolkat/gen/libproblem.so")

# choose an integrator
# integrator = RK4Classic(backend_fortran)
integrator = ExplicitEuler(backend_fortran)
# integrator = ImplicitEuler(backend_fortran)

"""
===============================================================================
"""

# set parameters
Nts         = 100
ts          = np.linspace(0, 2, Nts)
x           = np.ones(5)
p           = np.ones(5)
q           = np.zeros(4)
q[0]        = 90
q[1:]       = 1


# integrate
xs = np.zeros((Nts, 5))
for j in range(Nts-1):
    xs[j, :] = x
    x[:] = integrator.zo_forward(ts[j:j+2], x, p, q)

j = Nts-1
xs[j, :] = x


# plot results
pl.plot(ts, xs)
pl.xlabel("t")
pl.savefig(get_dir_path() + "/out/ind_zo_forward_bimolkat.png")
pl.show()

"""
===============================================================================
"""
