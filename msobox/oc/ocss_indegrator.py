# -*- coding: utf-8 -*-

"""
===============================================================================

optimal control problem discretized by INDegrator for single shooting ...

===============================================================================
"""

# system imports
import numpy as np
import json

# local imports
from msobox.mf.tapenade import Differentiator
from msobox.mf.fortran import BackendFortran

from msobox.ind.explicit_euler import ExplicitEuler
from msobox.ind.implicit_euler import ImplicitEuler
from msobox.ind.rk4classic import RK4Classic

"""
===============================================================================
"""

class OCSS_indegrator(object):

    """

    provides functionalities for ...

    """

    """
    ===============================================================================
    """

    def __init__(self, name, path, minormax, NX, NG, NP, NU, bc, ts):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        # decide whether to minimize or maximize
        if minormax == "max":
            self.sign = -1  # sign is negative for maximization
        else:
            self.sign = 1   # sign is positive for minimization

        # set attributes
        self.name = name
        self.path = path
        self.ts   = ts
        self.NTS  = ts.size
        self.NX   = NX
        self.NP   = NP
        self.NG   = NG
        self.NC   = NG * self.NTS
        self.NU   = NU
        self.NQ   = NU * self.NTS
        self.bc   = bc

        # load json containing data structure for differentiator
        with open(path + "ds.json", "r") as f:
            ds = json.load(f)

        # differentiate model functions
        Differentiator(path, ds=ds)
        self.backend_fortran = BackendFortran(path + "gen/libproblem.so")

    """
    ===============================================================================
    """

    def set_integrator(self, integrator):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        if integrator == "rk4":
            self.integrator = RK4Classic(self.backend_fortran)

        elif integrator == "explicit_euler":
            self.integrator = ExplicitEuler(self.backend_fortran)

        elif integrator == "implicit_euler":
            self.integrator = ImplicitEuler(self.backend_fortran)

        else:
            raise NotImplementedError

    """
    ===============================================================================
    """

    def q_array2ind(self, q):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            implement for non-constant controls

        """

        # set up array
        q_ind = np.zeros((self.NU, self.NTS, 1))

        # convert controls from one-dimensional array to INDegrator specific format
        for i in xrange(0, self.NU):
            q_ind[i, :, 0] = q[self.NTS * i:self.NTS * (i + 1)]

        return q_ind

    """
    ===============================================================================
    """

    def s_array2ind(self, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        # set up array
        s_ind = np.zeros((2, self.NX))

        # convert shooting variables from one-dimensional array to INDegrator specific format
        for i in xrange(0, 2):
            s_ind[i, :] = s[i * self.NX:(i + 1) * self.NX]

        return s_ind

    """
    ===============================================================================
    """

    def integrate(self, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        # convert controls and shooting variables to INDegrator specific format
        q = self.q_array2ind(q)
        s = self.s_array2ind(s)

        # set initial conditions
        x0 = s[0, :]

        # integrate
        self.integrator.zo_forward(self.ts, x0, p, q)

        return self.integrator.xs

    """
    ===============================================================================
    """

    def integrate_dx0(self, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        # convert controls and shooting variables to INDegrator specific format
        q = self.q_array2ind(q)
        s = self.s_array2ind(s)

        # set initial conditions
        x0 = s[0, :]

        # set up directions for differentation
        x0_dot = np.eye(self.NX)
        p_dot  = np.zeros((self.NP, self.NX))
        q_dot  = np.zeros(q.shape + (self.NX,))

        # integrate
        self.integrator.fo_forward_xpq(self.ts,
                                       x0, x0_dot,
                                       p, p_dot,
                                       q, q_dot)

        return self.integrator.xs, self.integrator.xs_dot

    """
    ===============================================================================
    """

    def integrate_dp(self, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        # convert controls and shooting variables to INDegrator specific format
        q = self.q_array2ind(q)
        s = self.s_array2ind(s)

        # set initial conditions
        x0 = s[0, :]

        # set up directions for differentation
        x0_dot = np.zeros((self.NX, self.NP))
        p_dot  = np.eye(self.NP)
        q_dot  = np.zeros(q.shape + (self.NP,))

        # integrate
        self.integrator.fo_forward_xpq(self.ts,
                                       x0, x0_dot,
                                       p, p_dot,
                                       q, q_dot)

        return self.integrator.xs, self.integrator.xs_dot

    """
    ===============================================================================
    """

    def integrate_dq(self, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        # convert controls and shooting variables to INDegrator specific format
        q = self.q_array2ind(q)
        s = self.s_array2ind(s)

        # set initial conditions
        x0 = s[0, :]

        # set up directions for differentation
        x0_dot                                  = np.zeros((self.NX, self.NQ))
        p_dot                                   = np.zeros((self.NP, self.NQ))
        q_dot                                   = np.zeros(q.shape + (self.NQ,))
        q_dot.reshape((self.NQ, self.NQ))[:, :] = np.eye(self.NQ)

        # integrate
        self.integrator.fo_forward_xpq(self.ts,
                                       x0, x0_dot,
                                       p, p_dot,
                                       q, q_dot)

        return self.integrator.xs, self.integrator.xs_dot

    """
    ===============================================================================
    """

    def integrate_dx0dx0(self, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        # convert controls and shooting variables to INDegrator specific format
        q = self.q_array2ind(q)
        s = self.s_array2ind(s)

        # set initial conditions
        x0 = s[0, :]

        # set up directions for differentation
        x0_dot  = np.eye(self.NX)
        x0_ddot = np.zeros(x0_dot.shape + (self.NX,))
        p_dot   = np.zeros((self.NP, self.NX))
        p_ddot  = np.zeros(p_dot.shape + (self.NX,))
        q_dot   = np.zeros(q.shape + (self.NX,))
        q_ddot  = np.zeros(q_dot.shape + (self.NX,))

        # integrate
        self.integrator.so_forward_xpq_xpq(self.ts,
                                           x0, x0_dot, x0_dot, x0_ddot,
                                           p, p_dot, p_dot, p_ddot,
                                           q, q_dot, q_dot, q_ddot)

        return self.integrator.xs, self.integrator.xs_dot1, self.integrator.xs_dot2, self.integrator.xs_ddot

    """
    ===============================================================================
    """

    def integrate_dpdp(self, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        # convert controls and shooting variables to INDegrator specific format
        q = self.q_array2ind(q)
        s = self.s_array2ind(s)

        # set initial conditions
        x0 = s[0, :]

        # set up directions for differentation
        x0_dot  = np.zeros((self.NX, self.NP))
        x0_ddot = np.zeros(x0_dot.shape + (self.NP,))
        p_dot   = np.eye(self.NP)
        p_ddot  = np.zeros(p_dot.shape + (self.NP,))
        q_dot   = np.zeros(q.shape + (self.NP,))
        q_ddot  = np.zeros(q_dot.shape + (self.NP,))

        # integrate
        self.integrator.so_forward_xpq_xpq(self.ts,
                                           x0, x0_dot, x0_dot, x0_ddot,
                                           p, p_dot, p_dot, p_ddot,
                                           q, q_dot, q_dot, q_ddot)

        return self.integrator.xs, self.integrator.xs_dot1, self.integrator.xs_dot2, self.integrator.xs_ddot

    """
    ===============================================================================
    """

    def integrate_dqdq(self, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        # convert controls and shooting variables to INDegrator specific format
        q = self.q_array2ind(q)
        s = self.s_array2ind(s)

        # set initial conditions
        x0 = s[0, :]

        # set up directions for differentation
        x0_dot                                  = np.zeros((self.NX, self.NQ))
        x0_ddot                                 = np.zeros(x0_dot.shape + (self.NQ,))
        p_dot                                   = np.zeros((p.size, self.NQ))
        p_ddot                                  = np.zeros(p_dot.shape + (self.NQ,))
        q_dot                                   = np.zeros(q.shape + (self.NQ,))
        q_dot.reshape((self.NQ, self.NQ))[:, :] = np.eye(self.NQ)
        q_ddot                                  = np.zeros(q_dot.shape + (self.NQ,))

        # integrate
        self.integrator.so_forward_xpq_xpq(self.ts,
                                           x0, x0_dot, x0_dot, x0_ddot,
                                           p, p_dot, p_dot, p_ddot,
                                           q, q_dot, q_dot, q_ddot)

        return self.integrator.xs, self.integrator.xs_dot1, self.integrator.xs_dot2, self.integrator.xs_ddot

    """
    ===============================================================================
    """

    def integrate_dx0dp(self, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        # convert controls and shooting variables to INDegrator specific format
        q = self.q_array2ind(q)
        s = self.s_array2ind(s)

        # set initial conditions
        x0 = s[0, :]

        # set up directions for differentation
        x0_dot1 = np.eye(self.NX)
        x0_dot2 = np.zeros((self.NX, self.NP))
        x0_ddot = np.zeros(x0_dot1.shape + (self.NP,))
        p_dot1  = np.zeros((self.NP, self.NX))
        p_dot2  = np.eye(self.NP)
        p_ddot  = np.zeros(p_dot1.shape + (self.NP,))
        q_dot1  = np.zeros(q.shape + (self.NX,))
        q_dot2  = np.zeros(q.shape + (self.NP,))
        q_ddot  = np.zeros(q_dot1.shape + (self.NP,))

        # integrate
        self.integrator.so_forward_xpq_xpq(self.ts,
                                           x0, x0_dot2, x0_dot1, x0_ddot,
                                           p, p_dot2, p_dot1, p_ddot,
                                           q, q_dot2, q_dot1, q_ddot)

        return self.integrator.xs, self.integrator.xs_dot1, self.integrator.xs_dot2, self.integrator.xs_ddot

    """
    ===============================================================================
    """

    def integrate_dx0dq(self, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        # convert controls and shooting variables to INDegrator specific format
        q = self.q_array2ind(q)
        s = self.s_array2ind(s)

        # set initial conditions
        x0 = s[0, :]

        # set up directions for differentation
        x0_dot1                                  = np.eye(self.NX)
        x0_dot2                                  = np.zeros((self.NX, self.NQ))
        x0_ddot                                  = np.zeros(x0_dot1.shape + (self.NQ,))
        p_dot1                                   = np.zeros((self.NP, self.NX))
        p_dot2                                   = np.zeros((self.NP, self.NQ))
        p_ddot                                   = np.zeros(p_dot1.shape + (self.NQ,))
        q_dot1                                   = np.zeros(q.shape + (self.NX,))
        q_dot2                                   = np.zeros(q.shape + (self.NQ,))
        q_dot2.reshape((self.NQ, self.NQ))[:, :] = np.eye(self.NQ)
        q_ddot                                   = np.zeros(q_dot1.shape + (self.NQ,))

        # integrate
        self.integrator.so_forward_xpq_xpq(self.ts,
                                           x0, x0_dot2, x0_dot1, x0_ddot,
                                           p, p_dot2, p_dot1, p_ddot,
                                           q, q_dot2, q_dot1, q_ddot)

        return self.integrator.xs, self.integrator.xs_dot1, self.integrator.xs_dot2, self.integrator.xs_ddot

    """
    ===============================================================================
    """

    def integrate_dpdq(self, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        # convert controls and shooting variables to INDegrator specific format
        q = self.q_array2ind(q)
        s = self.s_array2ind(s)

        # set initial conditions
        x0 = s[0, :]

        # set up directions for differentation
        x0_dot1                                  = np.zeros((self.NX, self.NP))
        x0_dot2                                  = np.zeros((self.NX, self.NQ))
        x0_ddot                                  = np.zeros(x0_dot1.shape + (self.NQ,))
        p_dot1                                   = np.eye(self.NP)
        p_dot2                                   = np.zeros((self.NP, self.NQ))
        p_ddot                                   = np.zeros(p_dot1.shape + (self.NQ,))
        q_dot1                                   = np.zeros(q.shape + (self.NP,))
        q_dot2                                   = np.zeros(q.shape + (self.NQ,))
        q_dot2.reshape((self.NQ, self.NQ))[:, :] = np.eye(self.NQ)
        q_ddot                                   = np.zeros(q_dot1.shape + (self.NQ,))

        # integrate
        self.integrator.so_forward_xpq_xpq(self.ts,
                                           x0, x0_dot2, x0_dot1, x0_ddot,
                                           p, p_dot2, p_dot1, p_ddot,
                                           q, q_dot2, q_dot1, q_ddot)

        return self.integrator.xs, self.integrator.xs_dot1, self.integrator.xs_dot2, self.integrator.xs_ddot

    """
    ===============================================================================
    """

    def c(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        c = None

        if self.NG > 0:

            # allocate memory
            c = np.zeros((self.NC,))

            t = np.zeros((1,))
            x = np.zeros((self.NX,))
            g = np.zeros((self.NG,))
            u = np.zeros((self.NU,))

            # loop through all time steps
            for i in xrange(0, self.NTS):

                # set time, state and controls for this time step
                t[0] = self.ts[i]
                x    = xs[i, :]

                for k in xrange(0, self.NU):
                    u[k] = q[i + k * self.NTS]

                # call fortran backend to calculate constraint functions for every control
                self.backend_fortran.gfcn(g, t, x, p, u)

                # build constraints
                for k in xrange(0, self.NG):
                    c[i + k * self.NTS] = g[k]

        return c

    """
    ===============================================================================
    """

    def c_dx0(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        dx0 = None

        if self.NG > 0:

            # allocate memory
            c     = np.zeros((self.NC,))
            dx0   = np.zeros((self.NC, self.NX))

            t     = np.zeros((1,))
            x     = np.zeros((self.NX,))
            x_dot = np.zeros((self.NX, self.NX))
            g     = np.zeros((self.NG,))
            g_dot = np.zeros((self.NG, self.NX))
            p_dot = np.zeros((self.NP, self.NX))
            u     = np.zeros((self.NU,))
            u_dot = np.zeros((self.NU, self.NX))

            # loop through all time step
            for i in xrange(0, self.NTS):

                # set time, state and controls for this time step
                t[0]  = self.ts[i]
                x     = xs[i, :]
                x_dot = np.reshape(xs_dot1[i, :, :], x_dot.shape)

                for l in xrange(0, self.NU):
                    u[l] = q[i + l * self.NTS]

                # call fortran backend to calculate derivatives of constraint functions
                self.backend_fortran.gfcn_dot(g, g_dot, t, x, x_dot, p, p_dot, u, u_dot)

                # store gradient
                for k in xrange(0, self.NG):
                    dx0[i + k * self.NTS, :] = g_dot[k, :]
                    c[i + k * self.NTS]      = g[k]

        return c, dx0

    """
    ===============================================================================
    """

    def c_dp(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        dp = None

        if self.NG > 0:

            # allocate memory
            c     = np.zeros((self.NC,))
            dp    = np.zeros((self.NC, self.NP))

            t     = np.zeros((1,))
            x     = np.zeros((self.NX,))
            x_dot = np.zeros((self.NX, self.NP))
            g     = np.zeros((self.NG,))
            g_dot = np.zeros((self.NG, self.NP))
            p_dot = np.eye(self.NP)
            u     = np.zeros((self.NU,))
            u_dot = np.zeros((self.NU, self.NP))

            # loop through all time steps
            for i in xrange(0, self.NTS):

                # set time, state and controls for this time step
                t[0]  = self.ts[i]
                x     = xs[i, :]
                x_dot = np.reshape(xs_dot1[i, :, :], x_dot.shape)

                for l in xrange(0, self.NU):
                    u[l] = q[i + l * self.NTS]

                # call fortran backend to calculate derivatives of constraint functions
                self.backend_fortran.gfcn_dot(g, g_dot, t, x, x_dot, p, p_dot, u, u_dot)

                # store gradient
                for k in xrange(0, self.NG):
                    dp[i + k * self.NTS, :] = g_dot[k, :]
                    c[i + k * self.NTS]     = g[k]

        return c, dp

    """
    ===============================================================================
    """

    def c_dq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        dq = None

        if self.NG > 0:

            # allocate memory
            c     = np.zeros((self.NC,))
            dq    = np.zeros((self.NC, self.NQ))

            t     = np.zeros((1,))
            x     = np.zeros((self.NX,))
            x_dot = np.zeros((self.NX, self.NU))
            g     = np.zeros((self.NG,))
            g_dot = np.zeros((self.NG, self.NU))
            p_dot = np.zeros((self.NP, self.NU))
            u     = np.zeros((self.NU,))
            u_dot = np.zeros((self.NU, self.NU))

            # loop through all time steps and controls
            for i in xrange(0, self.NTS):
                for j in xrange(0, self.NTS):

                    # set time, state and controls for this time step
                    t[0]  = self.ts[i]
                    x     = xs[i, :]
                    x_dot = np.reshape(xs_dot1[i, :, j], x_dot.shape)

                    for k in xrange(0, self.NU):
                        u[k] = q[i + k * self.NTS]

                    if i == j:
                        u_dot = np.eye(self.NU)
                    else:
                        u_dot = np.zeros((self.NU, self.NU))

                    # call fortran backend to calculate derivatives of constraint functions
                    self.backend_fortran.gfcn_dot(g, g_dot, t, x, x_dot, p, p_dot, u, u_dot)

                    # store gradient
                    for k in xrange(0, self.NG):
                        c[i + k * self.NTS] = g[k]

                        for l in xrange(0, self.NU):
                            dq[i + k * self.NTS, j + l * self.NTS] = g_dot[k, l]

        return c, dq

    """
    ===============================================================================
    """

    def c_dx0dx0(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        dx0dx0 = None

        if self.NG > 0:

            # allocate memory
            c      = np.zeros((self.NC,))
            dx01   = np.zeros((self.NC, self.NX))
            dx02   = np.zeros((self.NC, self.NX))
            dx0dx0 = np.zeros((self.NC, self.NX, self.NX))

            t      = np.zeros((1,))
            x      = np.zeros((self.NX,))
            x_dot1 = np.zeros((self.NX, self.NX))
            x_dot2 = np.zeros((self.NX, self.NX))
            x_ddot = np.zeros(x_dot1.shape + (self.NX,))
            g      = np.zeros((self.NG,))
            g_dot1 = np.zeros((self.NG, self.NX))
            g_dot2 = np.zeros((self.NG, self.NX))
            g_ddot = np.zeros(f_dot1.shape + (self.NX,))
            p_dot  = np.zeros((self.NP, self.NX))
            p_ddot = np.zeros(p_dot.shape + (self.NX,))
            u      = np.zeros((self.NU,))
            u_dot  = np.zeros((self.NU, self.NX))
            u_ddot = np.zeros(u_dot.shape + (self.NX,))

            # loop through all time step
            for i in xrange(0, self.NTS):

                # set time, state and controls for this time step
                t[0]   = self.ts[i]
                x      = xs[i, :]
                x_dot1 = np.reshape(xs_dot1[i, :, :], x_dot1.shape)
                x_dot2 = np.reshape(xs_dot2[i, :, :], x_dot2.shape)
                x_ddot = np.reshape(xs_ddot[i, :, :, :], x_ddot.shape)

                for l in xrange(0, self.NU):
                    u[l] = q[i + l * self.NTS]

                # call fortran backend to calculate derivatives of constraint functions
                self.backend_fortran.gfcn_ddot(g, g_dot2, g_dot1, g_ddot,
                                               t,
                                               x, x_dot2, x_dot1, x_ddot,
                                               p, p_dot, p_dot, p_ddot,
                                               u, u_dot, u_dot, u_ddot)

                # store gradient
                for k in xrange(0, self.NG):
                    dx0dx0[i + k * self.NTS, :, :] = g_ddot[k, :, :]
                    dx01[i + k * self.NTS, :]      = g_dot1[k, :]
                    dx02[i + k * self.NTS, :]      = g_dot2[k, :]
                    c[i + k * self.NTS]            = g[k]

        return c, dx01, dx02, dx0dx0

    """
    ===============================================================================
    """

    def c_dpdp(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        dpdp = None

        if self.NG > 0:

            # allocate memory
            c      = np.zeros((self.NC,))
            dp1    = np.zeros((self.NC, self.NP))
            dp2    = np.zeros((self.NC, self.NP))
            dpdp   = np.zeros((self.NC, self.NP, self.NP))

            t      = np.zeros((1,))
            x      = np.zeros((self.NX,))
            x_dot1 = np.zeros((self.NX, self.NP))
            x_dot2 = np.zeros((self.NX, self.NP))
            x_ddot = np.zeros(x_dot1.shape + (self.NP,))
            g      = np.zeros((self.NG,))
            g_dot1 = np.zeros((self.NG, self.NP))
            g_dot2 = np.zeros((self.NG, self.NP))
            g_ddot = np.zeros(f_dot1.shape + (self.NP,))
            p_dot  = np.eye(self.NP)
            p_ddot = np.zeros(p_dot.shape + (self.NP,))
            u      = np.zeros((self.NU,))
            u_dot  = np.zeros((self.NU, self.NP))
            u_ddot = np.zeros(u_dot.shape + (self.NP,))

            # loop through all time step
            for i in xrange(0, self.NTS):

                # set time, state and controls for this time step
                t[0]   = self.ts[i]
                x      = xs[i, :]
                x_dot1 = np.reshape(xs_dot1[i, :, :], x_dot1.shape)
                x_dot2 = np.reshape(xs_dot2[i, :, :], x_dot2.shape)
                x_ddot = np.reshape(xs_ddot[i, :, :, :], x_ddot.shape)

                for l in xrange(0, self.NU):
                    u[l] = q[i + l * self.NTS]

                # call fortran backend to calculate derivatives of constraint functions
                self.backend_fortran.gfcn_ddot(g, g_dot2, g_dot1, g_ddot,
                                               t,
                                               x, x_dot2, x_dot1, x_ddot,
                                               p, p_dot, p_dot, p_ddot,
                                               u, u_dot, u_dot, u_ddot)

                # store gradient
                for k in xrange(0, self.NG):
                    dpdp[i + k * self.NTS, :, :] = g_ddot[k, :, :]
                    dp1[i + k * self.NTS, :]     = g_dot1[k, :]
                    dp2[i + k * self.NTS, :]     = g_dot2[k, :]
                    c[i + k * self.NTS]          = g[k]

        return c, dp1, dp2, dpdp

    """
    ===============================================================================
    """

    def c_dqdq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        dqdq = None

        if self.NG > 0:

            # allocate memory
            c      = np.zeros((self.NC,))
            dq1    = np.zeros((self.NC, self.NQ))
            dq2    = np.zeros((self.NC, self.NQ))
            dqdq   = np.zeros((self.NC, self.NQ, self.NQ))

            t      = np.zeros((1,))
            x      = np.zeros((self.NX,))
            x_dot1 = np.zeros((self.NX, self.NU))
            x_dot2 = np.zeros((self.NX, self.NU))
            x_ddot = np.zeros(x_dot1.shape + (self.NU,))
            g      = np.zeros((self.NG,))
            g_dot1 = np.zeros((self.NG, self.NU))
            g_dot2 = np.zeros((self.NG, self.NU))
            g_ddot = np.zeros(f_dot1.shape + (self.NU,))
            p_dot  = np.zeros((self.NP, self.NU))
            p_ddot = np.zeros(p_dot.shape + (self.NU,))
            u      = np.zeros((self.NU,))
            u_dot  = np.zeros((self.NU, self.NU))
            u_ddot = np.zeros(u_dot.shape + (self.NU,))

            # loop through all time steps three times
            for i in xrange(0, self.NTS):
                for j in xrange(0, self.NTS):
                    for m in xrange(0, self.NTS):

                        # set time, state and controls for this time step
                        t[0]   = self.ts[i]
                        x      = xs[i, :]
                        x_dot1 = np.reshape(xs_dot1[i, :, j], x_dot1.shape)
                        x_dot2 = np.reshape(xs_dot2[i, :, j], x_dot2.shape)
                        x_ddot = np.reshape(xs_ddot[i, :, j, m], x_ddot.shape)

                        for k in xrange(0, self.NU):
                            u[k] = q[i + k * self.NTS]

                        if i == j and i == m:
                            u_dot = np.eye(self.NU)
                        else:
                            u_dot = np.zeros((self.NU, self.NU))

                        # call fortran backend to calculate derivatives of constraint functions
                        self.backend_fortran.gfcn_ddot(g, g_dot2, g_dot1, g_ddot,
                                                       t,
                                                       x, x_dot2, x_dot1, x_ddot,
                                                       p, p_dot, p_dot, p_ddot,
                                                       u, u_dot, u_dot, u_ddot)

                        # store gradient
                        for k in xrange(0, self.NG):
                            c[i + k * self.NTS] = g[k]

                            for l in xrange(0, self.NU):
                                for b in xrange(0, self.NU):
                                    dq1[i + k * self.NTS, j + l * self.NTS]                    = g_dot1[k, l]
                                    dq2[i + k * self.NTS, j + l * self.NTS]                    = g_dot2[k, l]
                                    dqdq[i + k * self.NTS, j + l * self.NTS, m + b * self.NTS] = g_ddot[k, l, b]

        return c, dq1, dq2, dqdq

    """
    ===============================================================================
    """

    def c_dx0dp(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        dx0dp = None

        if self.NG > 0:

            # allocate memory
            c      = np.zeros((self.NC,))
            dx0    = np.zeros((self.NC, self.NX))
            dp     = np.zeros((self.NC, self.NP))
            dx0dp  = np.zeros((self.NC, self.NX, self.NP))

            t      = np.zeros((1,))
            x      = np.zeros((self.NX,))
            x_dot1 = np.zeros(self.NX, self.NX)
            x_dot2 = np.zeros(self.NX, self.NP)
            x_ddot = np.zeros(x_dot.shape + (self.NP,))
            g      = np.zeros((self.NG,))
            g_dot1 = np.zeros((self.NG, self.NX))
            g_dot2 = np.zeros((self.NG, self.NP))
            g_ddot = np.zeros(f_dot1.shape + (self.NP,))
            p_dot1 = np.zeros((self.NP, self.NX))
            p_dot2 = np.eye(self.NP)
            p_ddot = np.zeros(p_dot1.shape + (self.NP,))
            u      = np.zeros((self.NU,))
            u_dot  = np.zeros((self.NU, self.NX))
            u_ddot = np.zeros(u_dot.shape + (self.NP,))

            # loop through all time step
            for i in xrange(0, self.NTS):

                # set time, state and controls for this time step
                t[0]   = self.ts[i]
                x      = xs[i, :]
                x_dot1 = np.reshape(xs_dot1[i, :, :], x_dot1.shape)
                x_dot2 = np.reshape(xs_dot2[i, :, :], x_dot2.shape)
                x_ddot = np.reshape(xs_ddot[i, :, :, :], x_ddot.shape)

                for l in xrange(0, self.NU):
                    u[l] = q[i + l * self.NTS]

                # call fortran backend to calculate derivatives of constraint functions
                self.backend_fortran.gfcn_ddot(g, g_dot2, g_dot1, g_ddot,
                                               t,
                                               x, x_dot2, x_dot1, x_ddot,
                                               p, p_dot2, p_dot1, p_ddot,
                                               u, u_dot, u_dot, u_ddot)

                # store gradient
                for k in xrange(0, self.NG):
                    dx0dp[i + k * self.NTS, :, :] = g_ddot[k, :, :]
                    dx0[i + k * self.NTS, :]      = g_dot1[k, :]
                    dxp[i + k * self.NTS, :]      = g_dot2[k, :]
                    c[i + k * self.NTS]           = g[k]

        return c, dx0, dp, dx0dp

    """
    ===============================================================================
    """

    def c_dx0dq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        dx0dq = None

        if self.NG > 0:

            # allocate memory
            c      = np.zeros((self.NC,))
            dx0    = np.zeros((self.NC, self.NX))
            dq     = np.zeros((self.NC, self.NP))
            dx0dq  = np.zeros((self.NC, self.NX, self.NP))

            t      = np.zeros((1,))
            x      = np.zeros((self.NX,))
            x_dot1 = np.zeros(self.NX, self.NX)
            x_dot2 = np.zeros(self.NX, self.NP)
            x_ddot = np.zeros(x_dot.shape + (self.NP,))
            g      = np.zeros((self.NG,))
            g_dot1 = np.zeros((self.NG, self.NX))
            g_dot2 = np.zeros((self.NG, self.NP))
            g_ddot = np.zeros(f_dot1.shape + (self.NP,))
            p_dot  = np.zeros((self.NP, self.NX))
            p_ddot = np.zeros(p_dot.shape + (self.NP,))
            u      = np.zeros((self.NU,))
            u_dot1 = np.zeros((self.NU, self.NX))
            u_dot2 = np.zeros((self.NU, self.NU))
            u_ddot = np.zeros(u_dot1.shape + (self.NP,))

            # loop through all time step
            for i in xrange(0, self.NTS):
                for j in xrange(0, self.NTS):

                    # set time, state and controls for this time step
                    t[0]   = self.ts[i]
                    x      = xs[i, :]
                    x_dot1 = np.reshape(xs_dot1[i, :, :], x_dot1.shape)
                    x_dot2 = np.reshape(xs_dot2[i, :, j], x_dot2.shape)
                    x_ddot = np.reshape(xs_ddot[i, :, :, j], x_ddot.shape)

                    for l in xrange(0, self.NU):
                        u[l] = q[i + l * self.NTS]

                    if i == j:
                        u_dot2 = np.eye(self.NU)
                    else:
                        u_dot2 = np.zeros((self.NU, self.NU))

                    # call fortran backend to calculate derivatives of constraint functions
                    self.backend_fortran.gfcn_ddot(g, g_dot2, g_dot1, g_ddot,
                                                   t,
                                                   x, x_dot2, x_dot1, x_ddot,
                                                   p, p_dot, p_dot, p_ddot,
                                                   u, u_dot2, u_dot1, u_ddot)

                    # store gradient
                    for k in xrange(0, self.NG):
                        c[i + k * self.NTS]      = g[k]
                        dx0[i + k * self.NTS, :] = g_dot1[k, :]

                        for l in xrange(0, self.NU):
                            dx0dq[i + k * self.NTS, :, j + l * self.NTS] = g_ddot[k, :, l]
                            dxq[i + k * self.NTS, j + l * self.NTS]      = g_dot2[k, l]


        return c, dx0, dq, dx0dq

    """
    ===============================================================================
    """

    def c_dpdq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        dpdq = None

        if self.NG > 0:

            # allocate memory
            c    = np.zeros((self.NC,))
            dp   = np.zeros((self.NC, self.NP))
            dq   = np.zeros((self.NC, self.NQ))
            dpdq = np.zeros((self.NC, self.NX, self.NQ))

            t      = np.zeros((1,))
            x      = np.zeros((self.NX,))
            x_dot1 = np.zeros(self.NX, self.NP)
            x_dot2 = np.zeros(self.NX, self.NU)
            x_ddot = np.zeros(x_dot.shape + (self.NU,))
            g      = np.zeros((self.NG,))
            g_dot1 = np.zeros((self.NG, self.NP))
            g_dot2 = np.zeros((self.NG, self.NU))
            g_ddot = np.zeros(f_dot1.shape + (self.NU,))
            p_dot1 = np.eye(self.NP)
            p_dot2 = np.zeros((self.NP, self.NU))
            p_ddot = np.zeros(p_dot1.shape + (self.NU,))
            u      = np.zeros((self.NU,))
            u_dot1 = np.zeros((self.NU, self.NP))
            u_dot2 = np.zeros((self.NU, self.NU))
            u_ddot = np.zeros(u_dot1.shape + (self.NU,))

            # loop through all time step
            for i in xrange(0, self.NTS):
                for j in xrange(0, self.NTS):

                    # set time, state and controls for this time step
                    t[0]   = self.ts[i]
                    x      = xs[i, :]
                    x_dot1 = np.reshape(xs_dot1[i, :, :], x_dot1.shape)
                    x_dot2 = np.reshape(xs_dot2[i, :, j], x_dot2.shape)
                    x_ddot = np.reshape(xs_ddot[i, :, :, j], x_ddot.shape)

                    for l in xrange(0, self.NU):
                        u[l] = q[i + l * self.NTS]

                    if i == j:
                        u_dot2 = np.eye(self.NU)
                    else:
                        u_dot2 = np.zeros((self.NU, self.NU))

                    # call fortran backend to calculate derivatives of constraint functions
                    self.backend_fortran.gfcn_ddot(g, g_dot2, g_dot1, g_ddot,
                                                   t,
                                                   x, x_dot2, x_dot1, x_ddot,
                                                   p, p_dot, p_dot, p_ddot,
                                                   u, u_dot2, u_dot1, u_ddot)

                    # store gradient
                    for k in xrange(0, self.NG):
                        c[i + k * self.NTS]     = g[k]
                        dp[i + k * self.NTS, :] = g_dot1[k, :]

                        for l in xrange(0, self.NU):
                            dpdq[i + k * self.NTS, :, j + l * self.NTS] = g_ddot[k, :, l]
                            dxq[i + k * self.NTS, j + l * self.NTS]     = g_dot2[k, l]


        return c, dp, dq, dpdq

    """
    ===============================================================================
    """

    def obj(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        return self.sign * xs[-1, -1]

    """
    ===============================================================================
    """

    def obj_dx0(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        return self.sign * xs_dot1[-1, -1, :]

    """
    ===============================================================================
    """

    def obj_dp(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        return self.sign * xs_dot1[-1, -1, :]

    """
    ===============================================================================
    """

    def obj_dq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        return self.sign * xs_dot1[-1, -1, :]

    """
    ===============================================================================
    """

    def obj_dx0dx0(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        return self.sign * xs_ddot[-1, -1, :, :]

    """
    ===============================================================================
    """

    def obj_dpdp(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        return self.sign * xs_ddot[-1, -1, :, :]

    """
    ===============================================================================
    """

    def obj_dqdq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        return self.sign * xs_ddot[-1, -1, :, :]

    """
    ===============================================================================
    """

    def obj_dx0dp(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        return self.sign * xs_ddot[-1, -1, :, :]

    """
    ===============================================================================
    """

    def obj_dx0dq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        return self.sign * xs_ddot[-1, -1, :, :]

    """
    ===============================================================================
    """

    def obj_dpdq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        return self.sign * xs_ddot[-1, -1, :, :]

"""
===============================================================================
"""
