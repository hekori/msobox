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

    def __init__(self, name, path, minormax, NX, NG, NP, NU, bc, ts, NTSI):

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
        self.bc     = bc
        self.NTSI   = NTSI
        self.NS     = self.NTS

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
        s_ind = np.zeros((self.NS, self.NX))

        # convert shooting variables from one-dimensional array to INDegrator specific format
        for i in xrange(0, self.NS):
            s_ind[i, :] = s[i * self.NX:(i + 1) * self.NX]

        return s_ind

    """
    ===============================================================================
    """

    def integrate_interval(self, interval, p, q, s):

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

        # allocate memory
        q_interval = np.zeros((self.NU, self.NTSI, 1))

        # set time steps for this interval
        tsi = np.linspace(self.ts[interval], self.ts[interval + 1], self.NTSI)

        # set constant controls for this interval
        q_interval[:, :, :] = q[:, interval, :]

        # integrate
        self.integrator.zo_forward(tsi, s[interval], p, q_interval)

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

        # allocate memory
        q_interval = np.zeros((self.NU, self.NTSI, 1))

        # set time steps for this interval
        tsi = np.linspace(self.ts[interval], self.ts[interval + 1], self.NTSI)

        # set constant controls for this interval
        q_interval[:, :, :] = q[:, interval, :]

        # set up directions for differentation
        x0_dot  = np.zeros((self.NX, self.NP))
        p_dot   = np.eye(self.NP)
        q_dot   = np.zeros(q_interval.shape + (self.NP,))

        # integrate
        self.integrator.fo_forward_xpq(tsi,
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

        # allocate memory
        q_interval  = np.zeros((self.NU, self.NTSI, 1))
        xs_dot      = np.zeros((self.NTSI, self.NX))

        # set time steps for this interval
        tsi = np.linspace(self.ts[interval], self.ts[interval + 1], self.NTSI)

        # set constant controls for this interval
        q_interval[:, :, :] = q[:, interval, :]

        # set up directions for differentation
        x0_dot                                      = np.zeros((self.NX, self.NTSI))
        p_dot                                       = np.zeros((self.NP, self.NTSI))
        q_dot                                       = np.zeros(q_interval.shape + (self.NTSI,))
        q_dot.reshape((self.NTSI, self.NTSI))[:, :] = np.eye(self.NTSI)

        # integrate
        self.integrator.fo_forward_xpq(tsi,
                                       x0, x0_dot,
                                       p, p_dot,
                                       q_interval, q_dot)

        # build derivatives for controls by summing up in shooting interval
        for j in xrange(0, self.NTSI):
            for k in xrange(0, self.NTSI):
                xs_dot[j, :] = xs_dot[j, :] + self.integrator.xs_dot[j, :, k]

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

        # allocate memory
        q_interval = np.zeros((self.NU, self.NTSI, 1))

        # set time steps for this interval
        tsi = np.linspace(self.ts[interval], self.ts[interval + 1], self.NTSI)

        # set constant controls for this interval
        q_interval[:, :, :] = q[:, interval, :]

        # set up directions for differentation
        x0_dot  = np.eye(self.NX)
        p_dot   = np.zeros((self.NP, self.NX))
        q_dot   = np.zeros(q_interval.shape + (self.NX,))

        # integrate
        self.integrator.fo_forward_xpq(tsi,
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
        xs = np.zeros((self.NTS + (self.NTSI - 2) * (self.NTS - 1), self.NX))

        # integrate on shooting intervals
        for i in xrange(0, self.NS - 1):

            # integrate and save data
            xs_interval                                             = self.integrate_interval(i, p, q, s)
            xs[i * (self.NTSI - 1):(i + 1) * (self.NTSI - 1), :]    = xs_interval[:-1, :]

        # set last time step
        xs[-1, :] = xs_interval[-1, :]

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
        xs      = np.zeros((self.NTS + (self.NTSI - 2) * (self.NS - 1), self.NX))
        xs_dot  = np.zeros((self.NTS + (self.NTSI - 2) * (self.NS - 1), self.NX, self.NP))

        # integrate on shooting intervals
        for i in xrange(0, self.NS - 1):

            # integrate and save data
            xs_interval, xs_dot_interval                                    = self.integrate_interval_dp(i, p, q, s)
            xs[i * (self.NTSI - 1):(i + 1) * (self.NTSI - 1), :]            = xs_interval[:-1, :]
            xs_dot[i * (self.NTSI - 1):(i + 1) * (self.NTSI - 1), :, :]     = xs_dot_interval[:-1, :, :]

        # set last time step
        xs[-1, :]           = xs_interval[-1, :]
        xs_dot[-1, :, :]    = xs_dot_interval[-1, :, :]

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
        xs      	= np.zeros((self.NTS + (self.NTSI - 2) * (self.NS - 1), self.NX))
        xs_dot  	= np.zeros((self.NTS + (self.NTSI - 2) * (self.NS - 1), self.NX, self.NQ))

        # integrate on shooting intervals
        for i in xrange(0, self.NS - 1):

        	# integrate and save data
            xs_interval, xs_dot_interval                                = self.integrate_interval_dq(i, p, q, s)
            xs[i * (self.NTSI - 1):(i + 1) * (self.NTSI - 1), :]        = xs_interval[:-1, :]
            xs_dot[i * (self.NTSI - 1):(i + 1) * (self.NTSI - 1), :, i] = xs_dot_interval[:-1, :]

        # set last time step
        xs[-1, :]            = xs_interval[-1, :]
        xs_dot[-1, :, -2]    = xs_dot_interval[-1, :]

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
        xs      = np.zeros((self.NTS + (self.NTSI - 2) * (self.NS - 1), self.NX))
        xs_dot  = np.zeros((self.NTS + (self.NTSI - 2) * (self.NS - 1), self.NX, self.NS * self.NX))

        # integrate on shooting intervals
        for i in xrange(0, self.NS - 1):

            # integrate and save data
            xs_interval, xs_dot_interval                                                            = self.integrate_interval_dx0(i, p, q, s)
            xs[i * (self.NTSI - 1):(i + 1) * (self.NTSI - 1), :]                                    = xs_interval[:-1, :]
            xs_dot[i * (self.NTSI - 1):(i + 1) * (self.NTSI - 1), :, i * self.NX:(i + 1) * self.NX] = xs_dot_interval[:-1, :, :]

        # set last time step
        xs[-1, :]                                   = xs_interval[-1, :]
        xs_dot[-1, :, -(2 * self.NX):-(self.NX)]    = xs_dot_interval[-1, :, :]

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
                t[0]    = self.ts[i]
                x       = xs[i * (self.NTSI - 1), :]

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
            dp      = np.zeros((self.NC, self.NP))
            t       = np.zeros((1,))
            x       = np.zeros((self.NX,))
            x_dot   = np.zeros((self.NX, self.NP))
            g       = np.zeros((self.NG,))
            g_dot   = np.zeros((self.NG, self.NP))
            p_dot   = np.zeros((self.NP, self.NP))
            u       = np.zeros((self.NU,))
            u_dot   = np.zeros((self.NU, self.NP))

            # loop through all time steps
            for i in xrange(0, self.NTS):

                # set time, state and controls for this time step
                t[0]    = self.ts[i]
                x       = xs[i * (self.NTSI - 1), :]
                x_dot   = np.reshape(xs_dot[i * (self.NTSI - 1), :, :], x_dot.shape)

                for l in xrange(0, self.NU):
                    u[l] = q[i + l * self.NTS]

                # call fortran backend to calculate derivatives of constraint functions
                self.backend_fortran.gfcn_dot(g, g_dot, t, x, x_dot, p, p_dot, u, u_dot)

                # store gradient
                for j in xrange(0, self.NP):
                    for k in xrange(0, self.NG):
                        dp[i + k * self.NP, j] = g_dot[k, j]

        return dp

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
            dq      = np.zeros((self.NC, self.NQ))
            t       = np.zeros((1,))
            x       = np.zeros((self.NX,))
            x_dot   = np.zeros((self.NX, self.NU))
            g       = np.zeros((self.NG,))
            g_dot   = np.zeros((self.NG, self.NU))
            p_dot   = np.zeros((self.NP, self.NU))
            u       = np.zeros((self.NU,))
            u_dot   = np.zeros((self.NU, self.NU))

            # loop through all time steps and controls
            for i in xrange(0, self.NTS):
                for j in xrange(0, self.NTS):

                    # set time, state and controls for this time step
                    t[0]    = self.ts[i]
                    x       = xs[i * (self.NTSI - 1), :]
                    x_dot   = np.reshape(xs_dot[i * (self.NTSI - 1), :, j], x_dot.shape)

                    for k in xrange(0, self.NU):
                        u[k] = q[i + k * self.NTS]

                    if i == j:
                        u_dot = np.eye(self.NU)
                    else:
                        u_dot = np.zeros((self.NU, self.NU))

                    # call fortran backend to calculate derivatives of constraint functions
                    self.backend_fortran.gfcn_dot(g, g_dot, t, x, x_dot, p, p_dot, u, u_dot)

                    # store gradient
                    for l in xrange(0, self.NU):
                        for m in xrange(0, self.NG):
                            dq[i + m * self.NTS, j + l * self.NTS] = g_dot[m, l]

        return dq

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

        ds = None

        if self.NG > 0:

            # allocate memory
            ds      = np.zeros((self.NC, self.NX * self.NS))
            t       = np.zeros((1,))
            x       = np.zeros((self.NX,))
            x_dot   = np.zeros((self.NX, self.NX))
            g       = np.zeros((self.NG,))
            g_dot   = np.zeros((self.NG, self.NX))
            p_dot   = np.zeros((self.NP, self.NX))
            u       = np.zeros((self.NU,))
            u_dot   = np.zeros((self.NU, self.NX))

            # loop through all time steps and shooting nodes
            for i in xrange(0, self.NTS):
                for j in xrange(0, self.NS):

                    # set time, state and controls for this time step
                    t[0]    = self.ts[i]
                    x       = xs[i * (self.NTSI - 1), :]
                    x_dot   = np.reshape(xs_dot[i * (self.NTSI - 1), :, j * self.NX:(j + 1) * self.NX], x_dot.shape)

                    for l in xrange(0, self.NU):
                        u[l] = q[i + l * self.NTS]

                    # call fortran backend to calculate derivatives of constraint functions
                    self.backend_fortran.gfcn_dot(g, g_dot, t, x, x_dot, p, p_dot, u, u_dot)

                    # store gradient
                    for l in xrange(0, self.NX):
                        for m in xrange(0, self.NG):
                            ds[i + m * self.NTS, l + j * self.NX] = g_dot[m, l]

        return ds

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

        dqdq = None

        if self.NG > 0:
            dqdq = np.zeros((self.NC, self.NQ, self.NQ))

        return dqdq

    """
    ===============================================================================
    """

    def c_dqdp(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

        """

        ...

        input:

        output:

        TODO:

        """

        dqdp = None

        if self.NG > 0:
            dqdp = np.zeros((self.NC, self.NQ, self.NP))

        return dqdp

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

        dpdp = None

        if self.NG > 0:
            dpdp = np.zeros((self.NC, self.NP, self.NP))

        return dpdp

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

        return self.sign * xs[-1, -1], self.sign * xs_dot1[-1, -1, :]

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

        return self.sign * xs[-1, -1], self.sign * xs_dot1[-1, -1, :]

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

        return self.sign * xs[-1, -1], self.sign * xs_dot1[-1, -1, :]

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

        return self.sign * xs[-1, -1], self.sign * xs_dot1[-1, -1, :], self.sign * xs_dot2[-1, -1, :], self.sign * xs_ddot[-1, -1, :, :]

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

        return self.sign * xs[-1, -1], self.sign * xs_dot1[-1, -1, :], self.sign * xs_dot2[-1, -1, :], self.sign * xs_ddot[-1, -1, :, :]

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

        return self.sign * xs[-1, -1], self.sign * xs_dot1[-1, -1, :], self.sign * xs_dot2[-1, -1, :], self.sign * xs_ddot[-1, -1, :, :]


"""
===============================================================================
"""
