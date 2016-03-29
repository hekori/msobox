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
Differentiator(get_dir_path() + "/fortran/bimolkat", ds=ds)
backend_fortran = BackendFortran(get_dir_path() + "/fortran/bimolkat/gen/libproblem.so")

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
q[0, :, 0]  = 90
q[1:, :, 0] = 1

# set directions for second order forward differentiation w.r.t. p
P           = p.size
x0_dot1     = np.zeros(x0.shape + (P,))
x0_dot2     = np.zeros(x0.shape + (P,))
x0_ddot 	= np.zeros(x0_dot1.shape + (P,))
p_dot1      = np.eye(P)
p_dot2      = np.eye(P)
p_ddot      = np.zeros(p_dot1.shape + (P,))
q_dot1      = np.zeros(q.shape + (P,))
q_dot2      = np.zeros(q.shape + (P,))
q_ddot 		= np.zeros(q_dot1.shape + (P,))

# integrate
integrator.so_forward_xpq_xpq(ts,
							  x0, x0_dot2, x0_dot1, x0_ddot,
							  p, p_dot2, p_dot1, p_ddot,
							  q, q_dot2, q_dot1, q_ddot)

# print d^2/dp^2 x(t=2)
print integrator.xs_ddot[-1, :, :, :]

"""
===============================================================================
"""
