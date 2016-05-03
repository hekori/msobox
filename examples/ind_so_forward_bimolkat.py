# -*- coding: utf-8 -*-

"""
===============================================================================

bimolkat example for second order forward derivative generation via IND

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
integrator = RK4Classic(backend_fortran)
# integrator = ExplicitEuler(backend_fortran)
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
q[1]        = 90
q[1:]       = 1

# set directions for second order forward differentiation w.r.t. p
P           = p.size
x_dot1      = np.zeros(x.shape + (P,))
x_dot2      = np.zeros(x.shape + (P,))
x_ddot   	= np.zeros(x_dot1.shape + (P,))
p_dot1      = np.eye(P)
p_dot2      = np.eye(P)
p_ddot      = np.zeros(p_dot1.shape + (P,))
q_dot1      = np.zeros(q.shape + (P,))
q_dot2      = np.zeros(q.shape + (P,))
q_ddot 		= np.zeros(q_dot1.shape + (P,))

xs_ddot = np.zeros((ts.size,) + x_ddot.shape)

# integrate
for j in range(Nts-1):
    xs_ddot[j, ...] = x_ddot
    x[...], x_dot2[...], x_dot1[...], x_ddot[...] = integrator.so_forward(ts[j:j+2],
                                                      x, x_dot2, x_dot1, x_ddot,
                                                      p, p_dot2, p_dot1, p_ddot,
                                                      q, q_dot2, q_dot1, q_ddot)
j = Nts-1
xs_ddot[j, ...] = x_ddot

# print d^2/dp^2 x(t=2)
print integrator.xs_ddot[-1, :, :, :]

# plot results
pl.plot(ts, xs_ddot[:, :, 0, 0])
pl.xlabel("t")
pl.savefig(get_dir_path() + "/out/ind_so_forward_bimolkat.png")
pl.show()



"""
===============================================================================
"""
