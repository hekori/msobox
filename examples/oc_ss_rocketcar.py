# -*- coding: utf-8 -*-

"""
===============================================================================

rocketcar example for optimal control with single shooting taken from
http://www.math.uni-bremen.de/zetem/alt/optimmedia/webcontrol/rakete.html

x1_dot = x2
x2_dot = u

-1 < u < 1

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
ocp.path = get_dir_path() + "/fortran/rocketcar/"   # folder containing the fortran model files
ocp.NX   = 3 					  		   		 	# number of states >= 1
ocp.NP   = 1  					  		   		 	# number of parameters >= 1
ocp.NU   = 1  									 	# number of control functions >= 1
ocp.NG   = 0                             		 	# number of inequality constraints
ocp.NH   = 0  							 		 	# number of equality constraints
ocp.ts   = np.linspace(0, 1, 50)    		     	# control grid
ocp.NTS  = ocp.ts.size   						 	# number of controls
ocp.NTSI = 2              		  		 		 	# number of time steps per control interval >= 2

ocp.x0   	   = [4, -1, None] 			  		    # initial values for states
ocp.xend 	   = [0, 0, None]          			    # boundary values for states
ocp.bnds 	   = np.array([-1, 1], ndmin=2)  	    # box constraints for control functions
ocp.p  		   = 1 * np.ones((ocp.NP,))       	    # parameter values
ocp.q 		   = 0 * np.ones((ocp.NTS,))            # initial guess for controls
ocp.approximate_s()  				  	     	    # calculate initial guess for shooting variables
ocp.minormax   = "min"						        # choose minimization or maximization
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