# -*- coding: utf-8 -*-

"""
===============================================================================

chemical reaction example for optimal control with multiple shooting taken
from http://www.math.uni-bremen.de/zetem/alt/optimmedia/webcontrol/chemie.html

x1_dot = - u * x1 + u ** 2 * x2
x2_dot = u * x1 - p * u ** 2 * x2

0 < u < 1

===============================================================================
"""

# system imports
import os as os
import datetime as datetime
import numpy as np
import matplotlib.pyplot as pl

# local imports
from msobox.oc.ocms_snopt import OCMS_snopt
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
name    = "oc-ms-chemistry"
path    = get_dir_path() + "/fortran/chemistry/"
ts      = np.linspace(0, 1, 20)
bc      = np.array([0, 1], ndmin=2)
problem = OCMS_indegrator(name=name, path=path, minormax="max", NX=2, NG=0, NP=1, NU=1, bc=bc, ts=ts, NTSI=10)
x0      = [1, 0]
xend    = [None, None]
p       = np.array([3.5])
q0      = 0 * np.ones((problem.NQ,))
s0      = np.array([1, 0] * problem.NS)

# choose an integrator
problem.set_integrator("rk4")
# problem.set_integrator("explict_euler")
# problem.set_integrator("implicit_euler")

# solve the problem
solver  = OCMS_snopt(problem)
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
