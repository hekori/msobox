# -*- coding: utf-8 -*-

"""
===============================================================================

spline example for optimal control and sensitivity analysis with single shooting
http://www.math.uni-bremen.de/zetem/alt/optimmedia/webcontrol/spline2.html

x1_dot = x2
x2_dot = u
x3_dot = u ** 2

x1 < 0.1

===============================================================================
"""

# system imports
import os as os
import datetime as datetime
import numpy as np
import matplotlib.pyplot as pl

# local imports
from msobox.oc.ocss_snopt import OCSS_snopt
from msobox.oc.ocss_indegrator import OCSS_indegrator
from msobox.oc.ocss_sensitivity import OCSS_sensitivity

# setting print options to print all array elements
np.set_printoptions(threshold=np.nan)

def get_dir_path():
    """return script directory"""
    import inspect, os
    return os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

"""
===============================================================================
"""

# initialize an optimal control problem
name    = "oc-ss-spline"
path    = get_dir_path() + "/fortran/spline/"
ts      = np.linspace(0, 1, 20)
bcq     = np.array([-1e6, 1e6], ndmin=2)
bcg     = np.array([-1e6, 0], ndmin=2)
problem = OCSS_indegrator(name=name, path=path, minormax="min", NX=3, NG=1, NP=1, NU=1, bcq=bcq, bcg=bcg, ts=ts)
x0      = [0, 1, 0]
xend    = [0, -1, None]
p       = np.array([1.])
q0      = 0 * np.ones((problem.NQ,))
s0      = np.array([0, 1, 0, 0, -1, 0])

# choose an integrator
problem.set_integrator("rk4")

# solve the problem
solver  = OCSS_snopt(problem)
results = solver.solve(x0=x0, xend=xend, p=p, q0=q0, s0=s0)

# print results
print "\n" + "optimal controls:", 		  results[0]
print "shooting variables:",      	      results[1]
print "objective:",               		  results[2]
print "constraints:",             		  results[3]
print "multipliers:",               	  results[4]
print "matching conditions:", 			  results[5]
print "multipliers matching conditions:", results[6]

q_opt   = results[0]
s_opt   = results[1]
mul_opt = results[4]

# plot controls and states
x_opt, x_opt_dp, x_opt_dq, x_opt_dpdq = problem.integrate_dpdq(p, q_opt, s_opt)
x_opt_dpdp 							  = problem.integrate_dpdp(p, q_opt, s_opt)[3]
x_opt_dqdq 							  = problem.integrate_dqdq(p, q_opt, s_opt)[3]

colors = ["blue", "red", "green", "yellow"]
for i in xrange(0, problem.NU):
    pl.plot(ts, problem.q_array2ind(q_opt)[i, :, 0], color=colors[i], linewidth=2, linestyle="solid", label="u_opt_" + str(i))
# for i in xrange(0, problem.NX):
#     pl.plot(np.linspace(0, 1, x_opt[:, i].size), x_opt[:, i], color=colors[i], linewidth=2, linestyle="solid", label="x_opt_" + str(i))

"""
===============================================================================
"""

# sensivity analysis
sensitivity = OCSS_sensitivity(problem)

# evaluate the active constraints
sensitivity.determine_active_constraints(x_opt, p, q_opt, s_opt)
print "active constraints:", sensitivity.ca, "\n"

# print sensitivites
sensitivity.calculate_sensitivities(x_opt, x_opt_dp, x_opt_dq,
								    x_opt_dpdp, x_opt_dqdq, x_opt_dpdq,
								    p, q_opt, s_opt, mul_opt)
print "sensitivities for controls:",    sensitivity.q_dp
print "sensitivities for multipliers:", sensitivity.mul_dp
print "sensitivities for objective:",   sensitivity.F_dp

# give approximations for new parameter values
p_new = np.array([1.05])
sensitivity.calculate_approximations(x_opt, p, q_opt, s_opt, mul_opt, p_new)
print "first-order approximation for controls:",       sensitivity.q_approx
print "first-order approximation for multipliers:",    sensitivity.mul_approx
print "second-order approximation for optimal value:", sensitivity.F_approx

# plot approximated controls
for i in xrange(0, problem.NU):
    pl.plot(ts, problem.q_array2ind(sensitivity.q_approx)[i, :, 0], color = colors[i], linewidth = 2, linestyle = "dashed", label = "u_approx_" + str(i))

# calculate real solution
results_new = solver.solve(x0=x0, xend=xend, p=p_new, q0=q0, s0=s0)

# print results
print "\n" + "optimal controls:", 		  results_new[0]
print "shooting variables:",      	      results_new[1]
print "objective:",               		  results_new[2]
print "constraints:",             		  results_new[3]
print "multipliers:",               	  results_new[4]
print "matching conditions:", 			  results_new[5]
print "multipliers matching conditions:", results_new[6]

q_new   = results_new[0]
s_new   = results_new[1]
mul_new = results_new[4]
x_new   = problem.integrate(p_new, q_new, s_new)

# evaluate the active constraints
sensitivity.determine_active_constraints(x_new, p_new, q_new, s_new)
print "active constraints:", sensitivity.ca, "\n"

# plot new controls
for i in xrange(0, problem.NU):
    pl.plot(ts, problem.q_array2ind(q_new)[i, :, 0], color = colors[i], linewidth = 2, linestyle = "dotted", label = "u_new_" + str(i))

# print differences
print "\n" + "diff optimal controls:", abs((results_new[0] - sensitivity.q_approx))
print "diff objective:",               abs((results_new[2] - sensitivity.F_approx))
print "diff multipliers:",             abs((results_new[4] - sensitivity.mul_approx))

# set plotting preferences, save and show plot
pl.xlabel("t")
pl.ylabel("")
pl.title("solution of ocp")
pl.grid(True)
pl.legend(loc="lower right")
# pl.savefig(problem.path + "output/" + problem.name + "-" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "-plot.png", bbox_inches="tight")
# pl.savefig(problem.path + "output/" + problem.name + "-" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "-plot.pdf", bbox_inches="tight")
pl.show()

"""
===============================================================================
"""