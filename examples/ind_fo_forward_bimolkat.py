# -*- coding: utf-8 -*-

"""
===============================================================================

bimolkat example for first order forward derivative generation via IND

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
ts          = np.linspace(0, 2, 10)
x0          = np.ones(5)
p           = np.ones(5)
q           = np.zeros((4, ts.size, 2))
q[0, :, 0]  = 90.
q[1:, :, 0] = 1.

# set directions for first order forward differentiation w.r.t. p
P           = p.size
x0_dot      = np.zeros(x0.shape + (P,))
p_dot       = np.eye(P)
q_dot       = np.zeros(q.shape + (P,))

# integrate
integrator.fo_forward_xpq(ts, x0, x0_dot, p, p_dot, q, q_dot)

# plot d/dp x2(t)
pl.title("d/dp x2(t)")
pl.plot(integrator.ts, integrator.xs_dot[:, 1, :])
pl.xlabel("t")
pl.show()

"""
===============================================================================
"""

# set parameters
ts          = np.linspace(0, 2, 10)
x0          = np.ones(5)
p           = np.ones(5)
q           = np.zeros((4, ts.size, 1))
q[0, :]     = 90.
q[1:, :]    = 1.

# set directions for first order forward differentiation w.r.t. q
P           				= q.size
x0_dot      				= np.zeros(x0.shape + (P,))
p_dot       				= np.zeros(p.shape + (P,))
q_dot       			    = np.zeros(q.shape + (P,))
q_dot.reshape((P, P))[:, :] = np.eye(P)

# integrate
integrator.fo_forward_xpq(ts, x0, x0_dot, p, p_dot, q, q_dot)

# plot d/dq1 x2(t)
pl.title("d/dq1 x2(t)")
pl.plot(integrator.ts, integrator.xs_dot[:, 1, :])
pl.xlabel("t")
pl.show()

"""
===============================================================================
"""