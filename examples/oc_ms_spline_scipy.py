# -*- coding: utf-8 -*-

"""
===============================================================================

spline example for optimal control with multiple shooting
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
from msobox.oc.ocms_scipy import OCMS_scipy
from msobox.oc.ocms_indegrator import OCMS_indegrator

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
name    = "oc-ms-spline"
path    = get_dir_path() + "/fortran/spline/"
ts      = np.linspace(0, 1, 20)
bcq     = np.array([-1e6, 1e6], ndmin=2)
problem = OCMS_indegrator(name=name, path=path, minormax="min", NX=3, NG=1, NH=0, NP=2, NU=1, bcq=bcq, ts=ts, NTSI=10)
x0      = [0, 1, 0]
xend    = [0, -1, None]
p       = 1 * np.ones((problem.NP,))
q0      = -3 * np.ones((problem.NQ,))
s0      = np.array([0] * problem.NS)
s0      = problem.initial_s0(x0, xend)

# choose an integrator
problem.set_integrator("rk4")

###########################


###########################

# solve the problem
solver  = OCMS_scipy(problem)
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
x_opt = problem.integrate(p, q_opt, s_opt)
x_opt = problem.x_intervals2plot(x_opt)
q_opt = problem.q_array2ind(q_opt)[:, :, 0]

colors = ["blue", "red", "green", "yellow"]
for i in xrange(0, problem.NU):
    pl.plot(ts, q_opt[i], color=colors[i], linewidth=2, linestyle="dashed", label="u_" + str(i))
for i in xrange(0, problem.NX):
    pl.plot(np.linspace(0, 1, x_opt[:, i].size), x_opt[:, i], color=colors[i], linewidth=2, linestyle="solid", label="x_" + str(i))

# set plotting preferences, save and show plot
pl.xlabel("t")
pl.ylabel("")
pl.title("solution of ocp")
pl.grid(True)
pl.legend(loc="upper right")
pl.savefig(problem.path + "/output/" + problem.name + "-" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "-plot.png", bbox_inches="tight")
pl.savefig(problem.path + "/output/" + problem.name + "-" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "-plot.pdf", bbox_inches="tight")
pl.show()

"""
===============================================================================
"""
