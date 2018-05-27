#!/usr/bin/env python
# -*- coding: utf-8 -*-

# system imports
import numpy as np
import pprint

# local imports
from msobox.oc.ss import SS

# setting print options to print all array elements
np.set_printoptions(threshold=np.nan)

def get_dir_path():
    """return script directory"""
    import inspect, os
    return os.path.dirname(os.path.abspath(inspect.getfile(
        inspect.currentframe())))

# =============================================================================

# setup the ocp
ocp = SS()
ocp.path = get_dir_path() + "/fortran/chemistry/"
ocp.NX = 2
ocp.NP = 1
ocp.NU = 1
ocp.NG = 0
ocp.NH = 0
ocp.ts = np.linspace(0, 1, 25)
ocp.NTS = ocp.ts.size
ocp.NTSI = 2
ocp.x0 = [1, 0]
ocp.xend = [None, None]
ocp.bnds = np.array([0, 1], ndmin=2)
ocp.p = np.array([3.5])
ocp.q = 0.5 * np.ones((ocp.NTS * ocp.NU,))
ocp.approximate_s()
ocp.minormax = "max"
ocp.integrator = "rk4classic"
ocp.prepare()

# derivative check
# ocp.check_fo_derivatives()
# ocp.check_so_derivatives()

# solve the ocp
ocp.solver = "snopt"
ocp.solve()
ocp.plot("nominal ocp")

# sensitivity analyis
ocp.calc_multipliers()
ocp.calc_sensitivities()
ocp.calc_sensitivity_domain()

# approximate solution for new parameters
ocp.p_new = np.array([3.0])
ocp.calc_approximations()
ocp.q = ocp.q_approx
ocp.s = ocp.s_approx
ocp.p = ocp.p_new
ocp.integrate()
ocp.plot("approximated perturbed ocp")

# restore feasibility
ocp.restore_feasibility()
ocp.integrate()
ocp.plot("approximated perturbed ocp with feasibility")

# solve again with new parameters
ocp.q = 0.5 * np.ones((ocp.NTS * ocp.NU,))
ocp.approximate_s()
ocp.solve()
ocp.plot("perturbed ocp")

# wait for input to close figures
input("Finished. Press any key to exit.")

# =============================================================================