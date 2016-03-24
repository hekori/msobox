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

"""
===============================================================================
"""

# differentiate model functions
Differentiator("./examples/fortran/bimolkat/ffcn.f")
backend_fortran = BackendFortran("./examples/fortran/bimolkat/gen/libproblem.so")

# choose an integrator
integrator = RK4Classic(backend_fortran)
# integrator = ExplicitEuler(backend_fortran)
# integrator = ImplicitEuler(backend_fortran)

"""
===============================================================================
"""

# set parameters
ts          = np.linspace(0, 2, 100)
x0          = np.ones(5)
p           = np.ones(5)
q           = np.zeros((4, ts.size, 2))
q[0, :, 0]  = 90
q[1:, :, 0] = 1

# integrate
integrator.zo_forward(ts, x0, p, q)

# plot results
pl.plot(integrator.ts, integrator.xs)
pl.xlabel("t")
pl.show()

"""
===============================================================================
"""