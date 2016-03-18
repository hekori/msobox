# -*- coding: utf-8 -*-

"""
===============================================================================

testing enviroment for optimal control with multiple shooting...

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

"""
===============================================================================
"""

if __name__ == "__main__":

    # measure execution time
    start = datetime.datetime.now()
    print "\n" + "... starting script at " + start.strftime("%Y-%m-%d %H:%M:%S") + "..." + "\n"

    """
    ===============================================================================
    """

    # initialize an optimal control problem
    ts          = np.linspace(0, 1, 10)
    bc          = np.array([-1e6, 1e6], ndmin=2)
    problem     = OCMS_indegrator(path=os.path.dirname(os.path.abspath(__file__)), minormax="max", NX=3, NG=1, NP=1, NU=1, bc=bc, ts=ts, NTSI=5)
    x0          = [0, 1, 0]
    xend        = [0, -1, None]
    p           = np.array([3])
    q0          = 0 * np.ones((problem.NQ,))
    s0          = np.array([0, 1, 0] * problem.NS)

    # solve the problem
    solver      = OCMS_snopt(problem)
    # sensitivity = OCSS_sensitivity(problem)
    results     = solver.solve(x0=x0, xend=xend, p=p, q0=q0, s0=s0)

    # print results
    print "\n" + "optimal controls:",   results[0]
    print "shooting variables:",        results[1]
    print "objective:",                 results[2]
    print "constraints:",               results[3]
    print "multipliers:",               results[4]

    q_opt   = results[0]
    s_opt   = results[1]
    mul     = results[4]

    # plot controls and states
    x_opt  = problem.integrate(p, q_opt, s_opt)
    q_opt  = problem.convert_q(q_opt)[:, :, 0]

    colors  = ["blue", "red", "green", "yellow"]
    for i in xrange(0, problem.NU):
        pl.plot(ts, q_opt[i], color=colors[i], linewidth=2, linestyle="dashed", label="u_" + str(i))
    for i in xrange(0, problem.NX):
        pl.plot(np.linspace(0, 1, x_opt[:, i].size), x_opt[:, i], color=colors[i], linewidth=2, linestyle="solid", label="x_" + str(i))

    """
    ===============================================================================
    """

    # # evaluate the active constraints
    # NCA, ca = sensitivity.active(x0, p, q_opt)
    # print "active constraints:", ca, "\n"

    # # print sensitivites
    # sensitivites = sensitivity.dp(x0, p, q_opt, mul)
    # print "sensitivities for controls:",    sensitivites[0]
    # print "sensitivities for multipliers:", sensitivites[1]
    # print "sensitivities for objective:",   sensitivites[2:]
    # # print "sensitivity domain:", sensitivity.domain(x0, p, optimal_controls, mul)

    # # give taylor approximations based on the sensitivities
    # # p_new = np.array([1.1, 1.0])
    # p_new = np.array([3.2])
    # approximations = sensitivity.taylor(x0, p, q_opt, mul, p_new)
    # print "first-order approximation for controls:",        approximations[0]
    # print "first-order approximation for multipliers:",     approximations[1]
    # print "second-order approximation for optimal value:",  approximations[2]

    # # plot approximated controls
    # q_plot  = problem.convert_q(approximations[0])[:, :, 0]
    # colors  = ["blue", "red", "green", "yellow"]
    # for i in xrange(0, problem.NU):
    #     pl.plot(ts, q_plot[i], color = colors[i], linewidth = 2, linestyle = "dashed", label = "approximated #" + str(i))

    # # calculate real solution
    # results_new = solver.solve(x0, p_new, q0)
    # print "\n" + "optimal controls:", results_new[0]
    # print "objective:",     results_new[1]
    # print "constraints:",   results_new[2]
    # print "multipliers:",   results_new[3]

    # # plot new controls
    # q_plot  = problem.convert_q(results_new[0])[:, :, 0]
    # colors  = ["blue", "red", "green", "yellow"]
    # for i in xrange(0, problem.NU):
    #     pl.plot(ts, q_plot[i], color = colors[i], linewidth = 2, linestyle = "dotted", label = "new #" + str(i))

    # # compare differences
    # print "\n" + "diff optimal controls in %:",     abs((results_new[0] - approximations[0]) / results_new[0] * 100)
    # print "diff objective in %:",                   abs((results_new[1] - approximations[2]) / results_new[1] * 100)
    # print "diff multipliers in %:",                 abs((results_new[3] - approximations[1]) / results_new[3] * 100)

    """
    ===============================================================================
    """

    # print execution time
    end             = datetime.datetime.now()
    execution_time  = end - start

    print "\n" + "... this script ended at " + end.strftime("%Y-%m-%d %H:%M:%S") + " and took", execution_time.days, \
          "days,", execution_time.seconds, "seconds and", execution_time.microseconds, "microseconds to execute ..." + "\n"

    # set plotting preferences, save and show plot
    pl.xlabel("t")
    pl.ylabel("")
    pl.title("solution of ocp")
    pl.grid(True)
    pl.legend(loc="upper right")
    pl.savefig(problem.path + "/output/" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "-plot.png", bbox_inches="tight")
    pl.savefig(problem.path + "/output/" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "-plot.pdf", bbox_inches="tight")
    pl.show()

"""
===============================================================================
"""