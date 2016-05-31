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
import numpy as np
import pprint

# project imports
from msobox.oc.ms import Problem
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
ocp.path = get_dir_path() + "/fortran/chemistry/"   # folder containing the fortran model files
ocp.NX   = 2 					  		   		 	# number of states >= 1
ocp.NP   = 1  					  		   		 	# number of parameters >= 1
ocp.NU   = 1  									 	# number of control functions >= 1
ocp.NG   = 0                             		 	# number of inequality constraints
ocp.NH   = 0  							 		 	# number of equality constraints
ocp.ts   = np.linspace(0, 1, 20)    		     	# control and shooting grid
ocp.NTS  = ocp.ts.size   						 	# number of controls and shooting nodes
ocp.NTSI = 2              		  		 		 	# number of time steps per shooting interval >= 2

ocp.x0   	   = [1, 0] 			  		        # initial values for states
ocp.xend 	   = [None, None]	          		    # boundary values for states
ocp.bnds 	   = np.array([0, 1], ndmin=2)  	    # box constraints for control functions
ocp.p  		   = np.array([3.5])	         	    # parameter values
ocp.q 		   = 0 * np.ones((ocp.NTS,))            # initial guess for controls
ocp.approximate_s()  				  	     	    # calculate initial guess for shooting variables
ocp.minormax   = "max"						        # choose minimization or maximization
ocp.integrator = "rk4classic"    	     		    # integrator to be used
ocp.prepare() 			  						    # check input and prepare subsequent solution

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