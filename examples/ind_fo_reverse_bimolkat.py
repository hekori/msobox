# -*- coding: utf-8 -*-

"""
===============================================================================

bimolkat example for first order reverse derivative generation via IND

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

"""
===============================================================================
"""

# set parameters
ts          = np.linspace(0, 2, 10)
x0          = np.ones(5)
p           = np.ones(5)
q           = np.zeros(4)
q[0]        = 90
q[1:]       = 1

# necessary for reverse mode: integrate zeroth order
integrator.zo_forward(ts, x0, p, q)

# set directions for reverse differentation
xs_bar 		  = np.zeros(integrator.xs.shape)
xs_bar[-1, 1] = 1

# integrate
integrator.fo_reverse(xs_bar)

# print results
print("gradient of x(t=2; x0, p, q) w.r.t. p  = \n", integrator.p_bar, "\n")
print("gradient of x(t=2; x0, p, q) w.r.t. q  = \n", integrator.q_bar, "\n")
print("gradient of x(t=2; x0, p, q) w.r.t. x0 = \n", integrator.x0_bar)

"""
===============================================================================
"""
