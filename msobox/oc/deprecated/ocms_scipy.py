# -*- coding: utf-8 -*-

"""
===============================================================================

single shooting solution of discretized optimal control problems with scipy ...

===============================================================================
"""

# system imports
import numpy as np
import datetime as datetime
import scipy as sp

"""
===============================================================================
"""

class OCMS_scipy(object):

    """

    provides functionalities for ...

    """

    """
    ===============================================================================
    """

    def __init__(self, ocp):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        # set attributes
        self.ocp = ocp

    """
    ===============================================================================
    """

    def solve(self, x0, xend, p, q0, s0):

        """

        solve an ocp ...

        input:

        output:

        TODO:

        """

        NQ = self.ocp.NQ + self.ocp.NS   # add the shooting variables as controls
        NC = self.ocp.NC + self.ocp.NMC  # add matching conditions for shooting nodes

        # set bnds
        bnds = []

        # set the upper and lower bnds for the controls q
        for j in xrange(0, self.ocp.NU):
            for k in xrange(0, self.ocp.NTS):
                bnds.append((self.ocp.bnds[j, 0], self.ocp.bnds[j, 1]))

        # set the upper and lower bnds for the shooting variables s
        for i in xrange(0, self.ocp.NX):
            if x0[i] is not None:
                bnds.append((x0[i], x0[i]))

            else:
                bnds.append((-1e6, 1e6))

        for i in xrange(0, self.ocp.NS - 2 * self.ocp.NX):
            bnds.append((-1e6, 1e6))

        for i in xrange(0, self.ocp.NX):
            if xend[i] is not None:
                bnds.append((xend[i], xend[i]))

            else:
                bnds.append((-1e6, 1e6))

        def obj(x):

            # separate controls and shooting variables for readability
            q = x[:self.ocp.NQ]
            s = x[self.ocp.NQ:]

            # integrate
            xs, xs_dot_q = self.ocp.integrate_dq(p, q, s)
            xs_dot_s     = self.ocp.integrate_ds(p, q, s)[1]

            # evaluate objective
            obj, obj_dq = self.ocp.obj_dq(xs, xs_dot_q, None, None, p, q, s)

            # allocate memory
            jac = np.zeros((NQ,))

            # build derviatives
            jac[0:self.ocp.NQ]                         = obj_dq
            jac[self.ocp.NQ:self.ocp.NQ + self.ocp.NS] = self.ocp.obj_ds(xs, xs_dot_s, None, None, p, q, s)[1]

            return obj, jac

        def eqc(x):

            # separate controls and shooting variables for readability
            q = x[:self.ocp.NQ]
            s = x[self.ocp.NQ:]

            # integrate and evaluate constraints
            xs  = self.ocp.integrate(p, q, s)

            return self.ocp.eqc(xs, None, None, None, p, q, s)

        def eqc_jac(x):

            # separate controls and shooting variables for readability
            q = x[:self.ocp.NQ]
            s = x[self.ocp.NQ:]

            # integrate
            xs, xs_dot_q = self.ocp.integrate_dq(p, q, s)
            xs_dot_s     = self.ocp.integrate_ds(p, q, s)[1]

            # allocate memory
            jac = np.zeros((self.ocp.NCH, NQ))

            # build derviatives
            jac[:, 0:self.ocp.NQ]                         = self.ocp.eqc_dq(xs, xs_dot_q, None, None, p, q, s)[1]
            jac[:, self.ocp.NQ:self.ocp.NQ + self.ocp.NS] = self.ocp.eqc_ds(xs, xs_dot_s, None, None, p, q, s)[1]

            return jac

        def ineqc(x):

            # separate controls and shooting variables for readability
            q = x[:self.ocp.NQ]
            s = x[self.ocp.NQ:]

            # integrate and evaluate constraints
            xs    = self.ocp.integrate(p, q, s)

            return -self.ocp.ineqc(xs, None, None, None, p, q, s)

        def ineqc_jac(x):

            # separate controls and shooting variables for readability
            q = x[:self.ocp.NQ]
            s = x[self.ocp.NQ:]

            # integrate
            xs, xs_dot_q = self.ocp.integrate_dq(p, q, s)
            xs_dot_s     = self.ocp.integrate_ds(p, q, s)[1]

            # allocate memory
            jac = np.zeros((self.ocp.NCG, NQ))

            # build derviatives
            jac[:, 0:self.ocp.NQ]                         = self.ocp.ineqc_dq(xs, xs_dot_q, None, None, p, q, s)[1]
            jac[:, self.ocp.NQ:self.ocp.NQ + self.ocp.NS] = self.ocp.ineqc_ds(xs, xs_dot_s, None, None, p, q, s)[1]

            return -jac

        def mc(x):

            # separate controls and shooting variables for readability
            q = x[:self.ocp.NQ]
            s = x[self.ocp.NQ:]

            # integrate and evaluate constraints
            xs = self.ocp.integrate(p, q, s)

            return self.ocp.mc(xs, None, None, None, p, q, s)

        def mc_jac(x):

            # separate controls and shooting variables for readability
            q = x[:self.ocp.NQ]
            s = x[self.ocp.NQ:]

            # integrate
            xs, xs_dot_q = self.ocp.integrate_dq(p, q, s)
            xs_dot_s     = self.ocp.integrate_ds(p, q, s)[1]

            # allocate memory
            jac = np.zeros((self.ocp.NMC, NQ))

            # build derviatives
            jac[:, 0:self.ocp.NQ] = self.ocp.mc_dq(xs, xs_dot_q, None, None, p, q, s)[1]
            jac[:, self.ocp.NQ:]  = self.ocp.mc_ds(xs, xs_dot_s, None, None, p, q, s)[1]

            return jac

        # set initial guess
        x = np.append(q0, s0)

        # solve with equality constraints only
        if self.ocp.NG == 0 and self.ocp.NH > 0:
            constraints = ({"type":"eq", "fun":eqc, "jac":eqc_jac},
                           {"type":"eq", "fun":mc, "jac":mc_jac})

            # call solver
            results = sp.optimize.minimize(obj, x, args=(), method="SLSQP", jac=True, bnds=bnds,
                                           constraints=constraints,
                                           options={"disp":True, "iprint":2, "ftol":1e-9})

            # print detailed output
            print results

            return results.x[:self.ocp.NQ], results.x[self.ocp.NQ:], results.fun, eqc(results.x), [], mc(results.x), []

        # solve with inequality constraints only
        if self.ocp.NH == 0 and self.ocp.NG > 0:
            constraints = ({"type":"ineq", "fun":ineqc, "jac":ineqc_jac},
                           {"type":"eq", "fun":mc, "jac":mc_jac})

            # call solver
            results = sp.optimize.minimize(obj, x, args=(), method="SLSQP", jac=True, bnds=bnds,
                                           constraints=constraints,
                                           options={"disp":True, "iprint":2, "ftol":1e-9})

            # print detailed output
            print results

            return results.x[:self.ocp.NQ], results.x[self.ocp.NQ:], results.fun, -ineqc(results.x), [], mc(results.x), []

        # solve with equality and inequality constraints
        if self.ocp.NH > 0 and self.ocp.NG > 0:
            constraints = ({"type":"eq", "fun":eqc, "jac":eqc_jac},
                           {"type":"ineq", "fun":ineqc, "jac":ineqc_jac},
                           {"type":"eq", "fun":mc, "jac":mc_jac})

            # call solver
            results = sp.optimize.minimize(obj, x, args=(), method="SLSQP", jac=True, bnds=bnds,
                                           constraints=constraints,
                                           options={"disp":True, "iprint":2, "ftol":1e-9})

            # print detailed output
            print results

            return results.x[:self.ocp.NQ], results.x[self.ocp.NQ:], results.fun, np.append(-ineqc(results.x), eqc(results.x)), [], mc(results.x), []

        # solve without constraints
        if self.ocp.NH == 0 and self.ocp.NG == 0:
            constraints = ({"type":"eq", "fun":mc, "jac":mc_jac})

            # call solver
            results = sp.optimize.minimize(obj, x, args=(), method="SLSQP", jac=True, bnds=bnds,
                                           constraints=constraints,
                                           options={"disp":True, "iprint":2, "ftol":1e-9})

            # print detailed output
            print results

            return results.x[:self.ocp.NQ], results.x[self.ocp.NQ:], results.fun, [], [], mc(results.x), []


"""
===============================================================================
"""