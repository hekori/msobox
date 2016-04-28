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


def get_dir_path():
    """return script directory"""
    import inspect, os
    return os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


"""
===============================================================================
"""
import json
with open(get_dir_path() + "/fortran/bimolkat/json.txt", "r") as f:
    ds = json.load(f)

# differentiate model functions
# Differentiator(get_dir_path() + "/fortran/bimolkat", ds=ds)
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
q[0]        = 90.
q[1:]       = 1.

# set directions for first order forward differentiation w.r.t. p
P           = p.size
x_dot       = np.zeros(x.shape + (P,))
p_dot       = np.eye(P)
q_dot       = np.zeros(q.shape + (P,))

xs_dot = np.zeros((ts.size,) + x_dot.shape)

# integrate
for j in range(Nts-1):
    xs_dot[j, ...] = x_dot
    x[...], x_dot[...] = integrator.fo_forward_xpq(np.linspace(ts[j], ts[j+1], 2),
                                              x, x_dot,
                                              p, p_dot,
                                              q, q_dot)
j = Nts-1
xs_dot[j, ...] = x_dot

# plot d/dp x2(t)
pl.figure()
pl.title("d/dp x2(t)")
pl.plot(ts, xs_dot[:, 1, :])
pl.xlabel("t")
pl.savefig(get_dir_path() + "/out/ind_fo_forward_bimolkat_p.png")
# pl.show()

"""
===============================================================================
"""

# set parameters
Nts         = 10
ts          = np.linspace(0, 2, Nts)
x           = np.ones(5)
p           = np.ones(5)
q           = np.zeros(4)
q[0]        = 90.
q[1:]       = 1.

# set directions for first order forward differentiation w.r.t. p
P           = Nts * q.size
x_dot       = np.zeros(x.shape + (P,))
p_dot       = np.zeros(p.shape + (P,))
q_dot       = np.zeros(q.shape + (P,))

xs_dot = np.zeros((ts.size,) + x_dot.shape)

# integrate
for j in range(Nts-1):
    xs_dot[j, ...] = x_dot
    q_dot[...] = 0
    q_dot[:, j*q.size:(j+1)*q.size] = np.eye(q.size)
    x[...], x_dot[...] = integrator.fo_forward_xpq(ts[j:j+2],
                                              x, x_dot,
                                              p, p_dot,
                                              q, q_dot)
j = Nts-1
xs_dot[j, ...] = x_dot

# plot d/dp x2(t)
pl.figure()
pl.title("d/dq x2(t)")
pl.plot(ts, xs_dot[:, 1, :])
pl.xlabel("t")
pl.savefig(get_dir_path() + "/out/ind_fo_forward_bimolkat_q.png")
pl.show()


"""
===============================================================================
"""
