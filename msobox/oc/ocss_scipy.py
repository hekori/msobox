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

class OCSS_scipy(object):

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

        # set nu
        NQ = self.ocp.NQ + 2 * self.ocp.NX  # add the shooting variables as controls
        NC = self.ocp.NC + 1 * self.ocp.NX  # add matching conditions for boundary

        # set bounds
        b = []

        # set the upper and lower bounds for the controls q
        for j in xrange(0, self.ocp.NU):
            for k in xrange(0, self.ocp.NTS):
                b.append((self.ocp.bcq[j, 0], self.ocp.bcq[j, 1]))

        # fix the shooting variables s at the boundaries if necessary
        for i in xrange(0, self.ocp.NX):
            if x0[i] is not None:
                b.append((x0[i], x0[i]))

            else:
                b.append((-1e6, 1e6))

        for i in xrange(0, self.ocp.NX):
            if xend[i] is not None:
                b.append((xend[i], xend[i]))

            else:
                b.append((-1e6, 1e6))

        def obj(x):

            # separate controls and shooting variables for readability
            q = x[:self.ocp.NQ]
            s = x[self.ocp.NQ:]

            # integrate
            xs, xs_dot_q = self.ocp.integrate_dq(p, q, s)
            xs_dot_x0    = self.ocp.integrate_dx0(p, q, s)[1]

            # evaluate gradient of objective
            obj, obj_dq = self.ocp.obj_dq(xs, xs_dot_q, None, None, p, q, s)
            obj_dx0     = self.ocp.obj_dx0(xs, xs_dot_x0, None, None, p, q, s)[1]

            # allocate memory
            jac = np.zeros((NQ,))

            # build jacobian
            jac[0:self.ocp.NQ]                         = obj_dq
            jac[self.ocp.NQ:self.ocp.NQ + self.ocp.NX] = obj_dx0
            jac[self.ocp.NQ + self.ocp.NX:]            = 0

            return obj, jac

        def cons(x):

            # separate controls and shooting variables for readability
            q = x[:self.ocp.NQ]
            s = x[self.ocp.NQ:]

            # integrate and evaluate constraints
            xs = self.ocp.integrate(p, q, s)
            c  = -self.ocp.c(xs, None, None, None, p, q, s)

            return c

        def cons_jac(x):

            # separate controls and shooting variables for readability
            q = x[:self.ocp.NQ]
            s = x[self.ocp.NQ:]

            # integrate
            xs, xs_dot_q = self.ocp.integrate_dq(p, q, s)
            xs_dot_x0    = self.ocp.integrate_dx0(p, q, s)[1]

            # evaluate jacobian of constraints
            c_dq  = -self.ocp.c_dq(xs, xs_dot_q, None, None, p, q, s)[1]
            c_dx0 = -self.ocp.c_dx0(xs, xs_dot_x0, None, None, p, q, s)[1]

            # allocate memory
            jac = np.zeros((self.ocp.NC, NQ))

            # build jacobian
            jac[:, 0:self.ocp.NQ]                         = c_dq
            jac[:, self.ocp.NQ:self.ocp.NQ + self.ocp.NX] = c_dx0
            jac[:, self.ocp.NQ + self.ocp.NX:]            = 0

            return jac

        def matching_conditions(x):

            # separate controls and shooting variables for readability
            q = x[:self.ocp.NQ]
            s = x[self.ocp.NQ:]

            # integrate and evaluate constraints
            xs = self.ocp.integrate(p, q, s)
            mc = xs[-1, :] - self.ocp.s_array2ind(s)[-1, :]

            return mc

        def matching_conditions_jac(x):

            # separate controls and shooting variables for readability
            q = x[:self.ocp.NQ]
            s = x[self.ocp.NQ:]

            # integrate
            xs, xs_dot_q = self.ocp.integrate_dq(p, q, s)
            xs_dot_x0    = self.ocp.integrate_dx0(p, q, s)[1]

            # evaluate jacobian of constraints
            c_dq  = self.ocp.c_dq(xs, xs_dot_q, None, None, p, q, s)[1]
            c_dx0 = self.ocp.c_dx0(xs, xs_dot_x0, None, None, p, q, s)[1]

            # allocate memory
            jac = np.zeros((self.ocp.NX, NQ))

            # build jacobian
            jac[:, 0:self.ocp.NQ]                         = xs_dot_q[-1, :, :]
            jac[:, self.ocp.NQ:self.ocp.NQ + self.ocp.NX] = xs_dot_x0[-1, :, :]
            jac[:, self.ocp.NQ + self.ocp.NX:]            = -np.eye(self.ocp.NX)

            return jac

        # set initial guess
        x = np.append(q0, s0)

        if self.ocp.NC > 0:

            # call solver with constraints
            results = sp.optimize.minimize(obj, x, args=(), method="SLSQP", jac=True, bounds=b,
                                           constraints=({"type":"ineq", "fun":cons, "jac":cons_jac},
                                                        {"type":"eq", "fun":matching_conditions, "jac":matching_conditions_jac}),
                                           tol=1e-6, options={'disp':True, 'iprint':2})

            # detailed output
            print results

            return results.x[:self.ocp.NQ], results.x[self.ocp.NQ:], results.fun, cons(results.x), [], matching_conditions(results.x), []

        else:

            # call solver without constraints
            results = sp.optimize.minimize(obj, x, args=(), method="SLSQP", jac=True, bounds=b,
                                           constraints=({"type":"eq", "fun":matching_conditions, "jac":matching_conditions_jac}),
                                           tol=1e-6, options={'disp':True, 'iprint':2})

            # detailed output
            print results

            return results.x[:self.ocp.NQ], results.x[self.ocp.NQ:], results.fun, [], [], matching_conditions(results.x), []



"""
===============================================================================
"""