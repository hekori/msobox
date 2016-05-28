# -*- coding: utf-8 -*-

"""
===============================================================================

optimal control problem discretized by INDegrator for multiple shooting ...

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

class OCMS_indegrator(object):

    """

    provides functionalities for ...

    """

    """
    ===============================================================================
    """

    def __init__(self, name, path, minormax, NX, NG, NP, NU, bcq, bcs, bcg, ts, NTSI):

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
        self.name   = name
        self.path   = path
        self.ts     = ts
        self.NTS    = ts.size
        self.NX     = NX
        self.NP     = NP
        self.NG     = NG
        self.NC     = NG * self.NTS
        self.NU     = NU
        self.NQ     = NU * self.NTS
        self.bcq    = bcq
        self.bcs    = bcs
        self.bcg    = bcg
        self.NTSI   = NTSI
        self.NS     = self.NTS * self.NX

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

    def initial_s0(self, x0, xend):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        # allocate memory
        s0 = np.zeros((self.NTS * self.NX,))

        # approximate shooting variables by linear interpolation if possible
        for j in xrange(0, self.NTS):
            for i in xrange(0, self.NX):

                # set all shooting variables to x0
                s0[j * self.NX + i] = x0[i]

                # interpolate from x0 to xend if possible
                if xend[i] is not None:
                    s0[j * self.NX + i] = x0[i] + float(j) / (self.NTS - 1) * (xend[i] - x0[i]) / (self.ts[-1] - self.ts[0])

        return s0

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
            q_ind[i, :, 0] = q[i * self.NTS:(i + 1) * self.NTS]

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
        s_ind = np.zeros((self.NTS, self.NX))

        # convert shooting variables from one-dimensional array to INDegrator specific format
        for i in xrange(0, self.NTS):
            s_ind[i, :] = s[i * self.NX:(i + 1) * self.NX]

        return s_ind

    """
    ===============================================================================
    """

    def x_intervals2plot(self, x):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        # differentiate between states and derivatives
        if len(x.shape) == 3:

            # set up array
            x_plot = np.zeros((self.NTS + (self.NTSI - 2) * (self.NTS - 1), x.shape[2]))

            # copy data
            for i in xrange(0, x.shape[0]):
                x_plot[i * (self.NTSI - 1):(i + 1) * (self.NTSI - 1), :] = x[i, :-1, :]

            # set last time step
            x_plot[-1, :] = x[-1, -1, :]

        elif len(x.shape) == 4:

            # set up array
            x_plot = np.zeros((self.NTS + (self.NTSI - 2) * (self.NTS - 1), x.shape[2], x.shape[3]))

            # copy data
            for i in xrange(0, x.shape[0]):
                x_plot[i * (self.NTSI - 1):(i + 1) * (self.NTSI - 1), :, :] = x[i, :-1, :, :]

            # set last time step
            x_plot[-1, :] = x[-1, -1, :]

        else:

            # print error message
            print "Error: Format of the array is unknown."
            x_plot = None

        return x_plot

    """
    ===============================================================================
    """

    def integrate_interval(self, interval, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...integr

        TODO:
            ...

        """

        # convert controls and shooting variables to INDegrator specific format
        q = self.q_array2ind(q)
        s = self.s_array2ind(s)

        # set initial conditions
        x0 = s[interval, :]

        # set time steps for this interval
        tsi = np.linspace(self.ts[interval], self.ts[interval + 1], self.NTSI)

        # set constant controls for this interval
        q_interval = q[:, interval, 0]

        # integrate
        self.integrator.zo_forward(tsi,
                                   x0,
                                   p,
                                   q_interval)

        return self.integrator.xs

    """
    ===============================================================================
    """

    def integrate_interval_dp(self, interval, p, q, s):

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
        x0 = s[interval, :]

        # set time steps for this interval
        tsi = np.linspace(self.ts[interval], self.ts[interval + 1], self.NTSI)

        # set constant controls for this interval
        q_interval[:, :, :] = q[:, interval, 0]

        # set up directions for differentation
        x0_dot = np.zeros((self.NX, self.NP))
        p_dot  = np.eye(self.NP)
        q_dot  = np.zeros(q_interval.shape + (self.NP,))

        # integrate
        self.integrator.fo_forward(tsi,
                                   x0, x0_dot,
                                   p, p_dot,
                                   q_interval, q_dot)

        return self.integrator.xs, self.integrator.xs_dot

    """
    ===============================================================================
    """

    def integrate_interval_dq(self, interval, p, q, s):

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
        x0 = s[interval, :]

        # set time steps for this interval
        tsi = np.linspace(self.ts[interval], self.ts[interval + 1], self.NTSI)

        # set constant controls for this interval
        q_interval = q[:, interval, 0]

        # set up directions for differentation
        x0_dot = np.zeros((self.NX, self.NU))
        p_dot  = np.zeros((self.NP, self.NU))
        q_dot  = np.eye(self.NU)

        # allocate memory
        xs_dot = np.zeros((self.NTSI, self.NX, self.NQ))

        # integrate
        self.integrator.fo_forward(tsi,
                                   x0, x0_dot,
                                   p, p_dot,
                                   q_interval, q_dot)

        xs_dot[:, :, interval * self.NU:(interval + 1) * self.NU] = self.integrator.xs_dot

        return self.integrator.xs, xs_dot

    """
    ===============================================================================
    """

    def integrate_interval_dx0(self, interval, p, q, s):

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
        x0 = s[interval, :]

        # set time steps for this interval
        tsi = np.linspace(self.ts[interval], self.ts[interval + 1], self.NTSI)

        # set constant controls for this interval
        q_interval = q[:, interval, 0]

        # set up directions for differentation
        x0_dot = np.eye(self.NX)
        p_dot  = np.zeros((self.NP, self.NX))
        q_dot  = np.zeros(q_interval.shape + (self.NX,))

        # integrate
        self.integrator.fo_forward(tsi,
                                   x0, x0_dot,
                                   p, p_dot,
                                   q_interval, q_dot)

        return self.integrator.xs, self.integrator.xs_dot

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

        # allocate memory
        xs = np.zeros((self.NTS - 1, self.NTSI, self.NX))

        # integrate on shooting intervals
        for i in xrange(0, self.NTS - 1):
            xs[i, :, :] = self.integrate_interval(i, p, q, s)

        return xs

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

        # allocate memory
        xs_    = np.zeros((self.NTS - 1, self.NTSI, self.NX))
        xs_dot = np.zeros((self.NTS - 1, self.NTSI, self.NX, self.NP))

        # integrate on shooting intervals
        for i in xrange(0, self.NTS - 1):
            xs[i, :, :], xs_dot[i, :, :, :] = self.integrate_interval_dp(i, p, q, s)

        return xs, xs_dot

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

        # allocate memory
        xs     = np.zeros((self.NTS - 1, self.NTSI, self.NX))
        xs_dot = np.zeros((self.NTS - 1, self.NTSI, self.NX, self.NQ))

        # integrate on shooting intervals
        for i in xrange(0, self.NTS - 1):
            xs[i, :, :], xs_dot[i, :, :, :] = self.integrate_interval_dq(i, p, q, s)

        return xs, xs_dot

    """
    ===============================================================================
    """

    def integrate_ds(self, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        # allocate memory
        xs     = np.zeros((self.NTS - 1, self.NTSI, self.NX))
        xs_dot = np.zeros((self.NTS - 1, self.NTSI, self.NX, self.NS))

        # integrate on shooting intervals
        for i in xrange(0, self.NTS - 1):
            xs[i, :, :], xs_dot[i, :, :, i * self.NX:(i + 1) * self.NX] = self.integrate_interval_dx0(i, p, q, s)

        return xs, xs_dot

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

        pass

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

        pass

    """
    ===============================================================================
    """

    def integrate_dqdp(self, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        pass

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
                if i == self.NTS - 1:
                    x = xs[i - 1, -1, :]

                else:
                    x = xs[i, 0, :]

                for k in xrange(0, self.NU):
                    u[k] = q[i + k * self.NTS]

                # call fortran backend to calculate constraint functions for every control
                self.backend_fortran.gfcn(g, self.ts[i:i + 1], x, p, u)

                # build constraints
                for k in xrange(0, self.NG):
                    c[i + k * self.NTS] = g[k]

        return c

    """
    ===============================================================================
    """

    def c_ds(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        c  = None
        ds = None

        if self.NG > 0:

            # allocate memory
            c     = np.zeros((self.NC,))
            ds    = np.zeros((self.NC, self.NS))

            x     = np.zeros((self.NX,))
            x_dot = np.zeros((self.NX, self.NX))
            g     = np.zeros((self.NG,))
            g_dot =  np.zeros((self.NG, self.NX))
            p_dot = np.zeros((self.NP, self.NX))
            u     = np.zeros((self.NU,))
            u_dot = np.zeros((self.NU, self.NX))

            # # loop through all time steps and shooting nodes
            # for i in xrange(0, self.NTS):
            #     for j in xrange(0, self.NTS):

            #         # set state and controls for this time step
            #         # x       = xs[i * (self.NTSI - 1), :]
            #         # x_dot   = np.reshape(xs_dot1[i * (self.NTSI - 1), :, j * self.NX:(j + 1) * self.NX], x_dot.shape)

            #         # set state and controls for this time step
            #         if i == self.NTS - 1:
            #             x     = xs[i - 1, -1, :]
            #             x_dot = np.reshape(xs_dot1[i - 1, -1, :, j * self.NX:(j + 1) * self.NX], x_dot.shape)

            #         else:
            #             x     = xs[i, 0, :]
            #             x_dot = np.reshape(xs_dot1[i, 0, :, j * self.NX:(j + 1) * self.NX], x_dot.shape)

            #         for l in xrange(0, self.NU):
            #             u[l] = q[i + l * self.NTS]

            #         # call fortran backend to calculate derivatives of constraint functions
            #         self.backend_fortran.gfcn_dot(g, g_dot, self.ts[i:i + 1], x, x_dot, p, p_dot, u, u_dot)

            #         # store gradient
            #         for l in xrange(0, self.NX):
            #             for m in xrange(0, self.NG):
            #                 ds[i + m * self.NTS, l + j * self.NX] = g_dot[m, l]

            # loop through all time steps but the last one
            for i in xrange(0, self.NTS - 1):

                x     = xs[i, 0, :]
                x_dot = np.reshape(xs_dot1[i, 0, :, i * self.NX:(i + 1) * self.NX], x_dot.shape)

                for l in xrange(0, self.NU):
                    u[l] = q[i + l * self.NTS]

                # call fortran backend to calculate derivatives of constraint functions
                self.backend_fortran.gfcn_dot(g, g_dot, self.ts[i:i + 1], x, x_dot, p, p_dot, u, u_dot)

                # store gradient
                for l in xrange(0, self.NX):
                    for m in xrange(0, self.NG):
                        ds[i + m * self.NTS, l + i * self.NX] = g_dot[m, l]

            # set last time step separately
            x     = xs[self.NTS - 2, -1, :]
            x_dot = np.reshape(xs_dot1[self.NTS - 2, -1, :, (self.NTS - 2) * self.NX:(self.NTS - 1) * self.NX], x_dot.shape)

            for l in xrange(0, self.NU):
                u[l] = q[self.NTS - 1 + l * self.NTS]

            # call fortran backend to calculate derivatives of constraint functions
            self.backend_fortran.gfcn_dot(g, g_dot, self.ts[self.NTS - 1:self.NTS], x, x_dot, p, p_dot, u, u_dot)

            for l in xrange(0, self.NX):
                for m in xrange(0, self.NG):
                    ds[self.NTS - 1 + m * self.NTS, l + (self.NTS - 2) * self.NX] = g_dot[m, l]

        return c, ds

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

        c  = None
        dp = None

        if self.NG > 0:

            # allocate memory
            c     = np.zeros((self.NC,))
            dp    = np.zeros((self.NC, self.NP))

            x     = np.zeros((self.NX,))
            x_dot = np.zeros((self.NX, self.NP))
            g     = np.zeros((self.NG,))
            g_dot = np.zeros((self.NG, self.NP))
            p_dot = np.zeros((self.NP, self.NP))
            u     = np.zeros((self.NU,))
            u_dot = np.zeros((self.NU, self.NP))

            # loop through all time steps
            for i in xrange(0, self.NTS):

                # set state and controls for this time step
                if i == self.NTS - 1:
                    x     = xs[i - 1, -1, :]
                    x_dot = np.reshape(xs_dot1[i - 1, -1, :, :], x_dot.shape)

                else:
                    x     = xs[i, 0, :]
                    x_dot = np.reshape(xs_dot1[i, 0, :, :], x_dot.shape)

                for l in xrange(0, self.NU):
                    u[l] = q[i + l * self.NTS]

                # call fortran backend to calculate derivatives of constraint functions
                self.backend_fortran.gfcn_dot(g, g_dot, self.ts[i:i + 1], x, x_dot, p, p_dot, u, u_dot)

                # store gradient
                for j in xrange(0, self.NP):
                    for k in xrange(0, self.NG):
                        dp[i + k * self.NP, j] = g_dot[k, j]

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

        c  = None
        dq = None

        if self.NG > 0:

            # allocate memory
            c     = np.zeros((self.NC,))
            dq    = np.zeros((self.NC, self.NQ))

            x     = np.zeros((self.NX,))
            x_dot = np.zeros((self.NX, self.NU))
            g     = np.zeros((self.NG,))
            g_dot = np.zeros((self.NG, self.NU))
            p_dot = np.zeros((self.NP, self.NU))
            u     = np.zeros((self.NU,))
            u_dot = np.zeros((self.NU, self.NU))

            # # loop through all time steps and controls
            # for i in xrange(0, self.NTS):
            #     for j in xrange(0, self.NTS):

            #         # set state and controls for this time step
            #         x     = xs[i * (self.NTSI - 1), :]
            #         x_dot = np.reshape(xs_dot1[i * (self.NTSI - 1), :, j], x_dot.shape)

            #         for k in xrange(0, self.NU):
            #             u[k] = q[i + k * self.NTS]

            #         if i == j:
            #             u_dot = np.eye(self.NU)
            #         else:
            #             u_dot = np.zeros((self.NU, self.NU))

            #         # call fortran backend to calculate derivatives of constraint functions
            #         self.backend_fortran.gfcn_dot(g, g_dot, self.ts[i:i + 1], x, x_dot, p, p_dot, u, u_dot)

            #         # store gradient
            #         for k in xrange(0, self.NG):
            #             for l in xrange(0, self.NU):
            #                 dq[i + k * self.NTS, j + l * self.NTS] = g_dot[k, l]

            # set gradient for constraints of last time step
            x     = xs[self.NTS - 2, -1, :]
            x_dot = np.reshape(xs_dot1[self.NTS - 2, -1, :, self.NTS - 2], x_dot.shape)

            for k in xrange(0, self.NU):
                u[k] = q[self.NTS - 1 + k * self.NTS]

            u_dot = np.zeros((self.NU, self.NU))

            # call fortran backend to calculate derivatives of constraint functions
            self.backend_fortran.gfcn_dot(g, g_dot, self.ts[self.NTS - 1:self.NTS], x, x_dot, p, p_dot, u, u_dot)

            # store gradient
            for l in xrange(0, self.NU):
                for m in xrange(0, self.NG):
                    dq[self.NTS - 1 + m * self.NTS, self.NTS - 2 + l * self.NTS] = g_dot[m, l]

        return c, dq

    """
    ===============================================================================
    """

    def c_dsds(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

        """

        ...

        input:

        output:

        TODO:

        """

        c    = None
        ds1  = None
        ds2  = None
        dsds = None

        return c, ds1, ds2, dsds

    """
    ===============================================================================
    """

    def c_dpdp(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

        """

        ...

        input:

        output:

        TODO:

        """

        c    = None
        dp1  = None
        dp2  = None
        dpdp = None

        return c, dp1, dp2, dpdp

    """
    ===============================================================================
    """

    def c_dqdq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

        """

        ...

        input:

        output:

        TODO:

        """

        c    = None
        dq1  = None
        dq2  = None
        dqdq = None

        return c, dq1, dq2, dqdq

    """
    ===============================================================================
    """

    def c_dsdp(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

        """

        ...

        input:

        output:

        TODO:

        """
        c    = None
        ds   = None
        dp   = None
        dsdp = None

        return c, ds, dp, dsdp

    """
    ===============================================================================
    """

    def c_dsdq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

        """

        ...

        input:

        output:

        TODO:

        """

        c    = None
        ds   = None
        dq   = None
        dsdq = None

        return c, ds, dq, dsdq

    """
    ===============================================================================
    """

    def c_dpdq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

        """

        ...

        input:

        output:

        TODO:

        """

        c    = None
        dp   = None
        dq   = None
        dpdq = None

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

        return self.sign * xs[-1, -1, -1]

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

        return self.sign * xs[-1, -1, -1], self.sign * xs_dot1[-1, -1, -1, :]

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

        return self.sign * xs[-1, -1, -1], self.sign * xs_dot1[-1, -1, -1, :]

    """
    ===============================================================================
    """

    def obj_ds(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        return self.sign * xs[-1, -1, -1], self.sign * xs_dot1[-1, -1, -1, :]

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

        return self.sign * xs[-1, -1, -1], self.sign * xs_dot1[-1, -1, -1, :], self.sign * xs_dot2[-1, -1, -1, :], self.sign * xs_ddot[-1, -1, -1, :, :]

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

        return self.sign * xs[-1, -1, -1], self.sign * xs_dot1[-1, -1, -1, :], self.sign * xs_dot2[-1, -1, -1, :], self.sign * xs_ddot[-1, -1, -1, :, :]

    """
    ===============================================================================
    """

    def obj_dqdp(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        return self.sign * xs[-1, -1, -1], self.sign * xs_dot1[-1, -1, -1, :], self.sign * xs_dot2[-1, -1, -1, :], self.sign * xs_ddot[-1, -1, -1, :, :]


"""
===============================================================================
"""
