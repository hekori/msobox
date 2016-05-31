# -*- coding: utf-8 -*-

"""
===============================================================================

spline example for optimal control with single shooting
http://www.math.uni-bremen.de/zetem/alt/optimmedia/webcontrol/spline2.html

x1_dot = x2
x2_dot = u
x3_dot = u ** 2

x1 < 0.1

===============================================================================
"""

# system imports
import numpy as np
import pprint

# local imports
from msobox.oc.ss import Problem
from msobox.oc.solver import Solver

# setting print options to print all array elements
np.set_printoptions(threshold=np.nan)

def get_dir_path():
    """return script directory"""
    import inspect, os
    return os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

"""
===============================================================================
"""

ocp 	 = Problem()
ocp.path = get_dir_path() + "/fortran/spline/"   # folder containing the fortran model files
ocp.NX   = 3 					  		   		 # number of states >= 1
ocp.NP   = 2  					  		   		 # number of parameters >= 1
ocp.NU   = 1  									 # number of control functions >= 1
ocp.NG   = 1                             		 # number of inequality constraints
ocp.NH   = 1  							 		 # number of equality constraints
ocp.ts   = np.linspace(0, 1, 20)    		     # control grid
ocp.NTS  = ocp.ts.size   						 # number of controls
ocp.NTSI = 2              		  		 		 # number of time steps per control interval >= 2

ocp.x0   	   = [0, 1, 0] 			  		     # initial values for states
ocp.xend 	   = [0, -1, None]          		 # boundary values for states
ocp.bnds 	   = np.array([-1e6, 1e6], ndmin=2)  # box constraints for control functions
ocp.p  		   = 1 * np.ones((ocp.NP,))       	 # parameter values
ocp.q 		   = -3 * np.ones((ocp.NTS,))        # initial guess for controls
ocp.s          = np.array([0, 1, 0, 0, -1, 0]) 	 # initial guess for shooting variables
ocp.minormax   = "min"						     # choose minimization or maximization
ocp.integrator = "rk4classic"    	     		 # integrator to be used
ocp.prepare() 			  						 # check input and prepare subsequent solution

ocs        = Solver()
ocs.ocp    = ocp      # ocp to be solved
ocs.solver = "snopt"  # nlp solver to be used
ocs.solve()			  # solve the ocp

print "\n"
pprint.pprint(ocs.results)  # print results

ocp.q = ocs.results["q"]
ocp.s = ocs.results["s"]
ocp.plot()  # plot results

"""
===============================================================================
"""

# # sensivity analysis
# sensitivity = OCSS_sensitivity(problem)

# # evaluate the active constraints
# sensitivity.determine_active_constraints(x_opt, p, q_opt, s_opt)
# print "active constraints:", sensitivity.ca, "\n"

# # print sensitivites
# sensitivity.calculate_sensitivities(x_opt, x_opt_dp, x_opt_dq,
# 								    x_opt_dpdp, x_opt_dqdq, x_opt_dpdq,
# 								    p, q_opt, s_opt, mul_opt)
# print "sensitivities for controls:",    sensitivity.q_dp
# print "sensitivities for multipliers:", sensitivity.mul_dp
# print "sensitivities for objective:",   sensitivity.F_dp

# # give approximations for new parameter values
# p_new = np.array([1.05])
# sensitivity.calculate_approximations(x_opt, p, q_opt, s_opt, mul_opt, p_new)
# print "first-order approximation for controls:",       sensitivity.q_approx
# print "first-order approximation for multipliers:",    sensitivity.mul_approx
# print "second-order approximation for optimal value:", sensitivity.F_approx

# # plot approximated controls
# for i in xrange(0, problem.NU):
#     pl.plot(ts, problem.q_array2ind(sensitivity.q_approx)[i, :, 0], color = colors[i], linewidth = 2, linestyle = "dashed", label = "u_approx_" + str(i))

# # calculate real solution
# results_new = solver.solve(x0=x0, xend=xend, p=p_new, q0=q0, s0=s0)

# # print results
# print "\n" + "optimal controls:", 		  results_new[0]
# print "shooting variables:",      	      results_new[1]
# print "objective:",               		  results_new[2]
# print "constraints:",             		  results_new[3]
# print "multipliers:",               	  results_new[4]
# print "matching conditions:", 			  results_new[5]
# print "multipliers matching conditions:", results_new[6]

# q_new   = results_new[0]
# s_new   = results_new[1]
# mul_new = results_new[4]
# x_new   = problem.integrate(p_new, q_new, s_new)

# # evaluate the active constraints
# sensitivity.determine_active_constraints(x_new, p_new, q_new, s_new)
# print "active constraints:", sensitivity.ca, "\n"

# # plot new controls
# for i in xrange(0, problem.NU):
#     pl.plot(ts, problem.q_array2ind(q_new)[i, :, 0], color = colors[i], linewidth = 2, linestyle = "dotted", label = "u_new_" + str(i))

# # print differences
# print "\n" + "diff optimal controls:", abs((results_new[0] - sensitivity.q_approx))
# print "diff objective:",               abs((results_new[2] - sensitivity.F_approx))
# print "diff multipliers:",             abs((results_new[4] - sensitivity.mul_approx))

# # set plotting preferences, save and show plot
# pl.xlabel("t")
# pl.ylabel("")
# pl.title("solution of ocp")
# pl.grid(True)
# pl.legend(loc="lower right")
# # pl.savefig(problem.path + "output/" + problem.name + "-" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "-plot.png", bbox_inches="tight")
# # pl.savefig(problem.path + "output/" + problem.name + "-" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "-plot.pdf", bbox_inches="tight")
# pl.show()

"""
===============================================================================
"""