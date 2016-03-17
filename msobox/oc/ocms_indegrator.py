# -*- coding: utf-8 -*-

"""
===============================================================================

optimal control problem discretized by INDegrator for multiple shooting ...

===============================================================================
"""

# system imports
import numpy as np

# local imports
from indegrator.rk4 import RK4 as RK4Classic
from indegrator.tapenade import Differentiator
from indegrator.backend_fortran import BackendFortran as BackendTapenade
# from msobox.ind.rk4classic import RK4Classic
# from msobox.ad.tapenade import Differentiator
# from msobox.mf.tapenade import BackendTapenade

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

    def __init__(self, path, minormax, NX, NG, NP, NU, bc, ts, NTSI):

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

        # build model functions and derivatives from fortran files and initialize INDegrator
        Differentiator(path + "/ffcn/ffcn.f")
        self.backend_ffcn   = BackendTapenade(path + "/ffcn/libproblem.so")
        self.integrator     = RK4Classic(self.backend_ffcn)

        # if necessary build constraint functions and derivatives from fortran files
        self.backend_gfcn   = None
        if NG > 0:
            Differentiator(path + "/gfcn/ffcn.f")
            self.backend_gfcn = BackendTapenade(path + "/gfcn/libproblem.so")

    """
    ===============================================================================
    """

    def convert_q(self, q):

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
        q_indegrator = np.zeros((self.NU, self.NTS, 1))

        # convert controls from one-dimensional array to INDegrator specific format
        for i in xrange(0, self.NU):
            q_indegrator[i, :, 0] = q[i * self.NTS:(i + 1) * self.NTS]

        return q_indegrator

    """
    ===============================================================================
    """

    def convert_s(self, s):

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
        s_indegrator = np.zeros((self.NS, self.NX))

        # convert shooting variables from one-dimensional array to INDegrator specific format
        for i in xrange(0, self.NS):
            s_indegrator[i, :] = s[i * self.NX:(i + 1) * self.NX]

        return s_indegrator

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
        q = self.convert_q(q)
        s = self.convert_s(s)

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
        q = self.convert_q(q)
        s = self.convert_s(s)

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
        self.integrator.fo_forward_xpu(self.ts,
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
        q = self.convert_q(q)
        s = self.convert_s(s)

        # set initial conditions
        x0 = s[interval, :]

        # allocate memory
        q_interval = np.zeros((self.NU, self.NTSI, 1))

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
        self.integrator.fo_forward_xpu(self.ts,
                                       x0, x0_dot,
                                       p, p_dot,
                                       q_interval, q_dot)

        return self.integrator.xs, self.integrator.xs_dot

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
        q = self.convert_q(q)
        s = self.convert_s(s)

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
        self.integrator.fo_forward_xpu(self.ts,
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
        xs      = np.zeros((self.NTS + (self.NTSI - 2) * (self.NTS - 1), self.NX))
        xs_dot  = np.zeros((self.NTS + (self.NTSI - 2) * (self.NTS - 1), self.NX, self.NP))

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
        xs      = np.zeros((self.NTS + (self.NTSI - 2) * (self.NTS - 1), self.NX))
        xs_dot  = np.zeros((self.NTS + (self.NTSI - 2) * (self.NTS - 1), self.NX, self.NQ))

        # integrate on shooting intervals
        for i in xrange(0, self.NS - 1):

            # integrate and save data
            xs_interval, xs_dot_interval                                    = self.integrate_interval_dq(i, p, q, s)
            xs[i * (self.NTSI - 1):(i + 1) * (self.NTSI - 1), :]            = xs_interval[:-1, :]
            xs_dot[i * (self.NTSI - 1):(i + 1) * (self.NTSI - 1), :, :]     = xs_dot_interval[:-1, :, :]

        # set last time step
        xs[-1, :]           = xs_interval[-1, :]
        xs_dot[-1, :, :]    = xs_dot_interval[-1, :, :]

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
        xs      = np.zeros((self.NTS + (self.NTSI - 2) * (self.NTS - 1), self.NX))
        xs_dot  = np.zeros((self.NTS + (self.NTSI - 2) * (self.NTS - 1), self.NX, self.NX))

        # integrate on shooting intervals
        for i in xrange(0, self.NS - 1):

            # integrate and save data
            xs_interval, xs_dot_interval                                    = self.integrate_interval_dx0(i, p, q, s)
            xs[i * (self.NTSI - 1):(i + 1) * (self.NTSI - 1), :]            = xs_interval[:-1, :]
            xs_dot[i * (self.NTSI - 1):(i + 1) * (self.NTSI - 1), :, :]     = xs_dot_interval[:-1, :, :]

        # set last time step
        xs[-1, :]           = xs_interval[-1, :]
        xs_dot[-1, :, :]    = xs_dot_interval[-1, :, :]

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

    def c(self, xs, xs_dot, xs_ddot, p, q, s):

        """

        constraints for the discretized ocp

        input:

        output:

        TODO: implement for multiple controls

        """

        c = None

        if self.NG > 0:
            # allocate memory
            c = np.zeros((self.NC,))
            t = np.zeros((1,))
            x = np.zeros((self.NX,))
            f = np.zeros((self.NG,))
            u = np.zeros((self.NU,))

            # loop through all time steps
            for i in xrange(0, self.NTS):

                # choose current controls of this time step
                for k in xrange(0, self.NU):
                    u[k] = q[i + k * self.NTS]

                # call fortran backend to calculate constraint functions for every control
                self.backend_gfcn.ffcn(t, x, f, p, u)

                # build constraints
                for k in xrange(0, self.NG):
                    c[i + k * self.NTS] = f[k]

        return c

    """
    ===============================================================================
    """

    def c_dp(self, xs, xs_dot, xs_ddot, p, q, s):

        """

        ...

        input:

        output:

        TODO:

        """

        dp = None

        if self.NG > 0:
            # allocate memory
            dp      = np.zeros((self.NC, self.NP))
            t       = np.zeros((1,))
            x       = np.zeros((self.NX,))
            x_dot   = np.zeros((self.NX, self.NP))
            f       = np.zeros((self.NG,))
            f_dot   = np.zeros((self.NG, self.NP))
            p_dot   = np.eye(self.NP)
            u       = np.zeros((self.NU,))
            u_dot   = np.zeros((self.NU, self.NP))

            # loop through all time steps
            for i in xrange(0, self.NTS):

                # choose current controls of this time step
                for k in xrange(0, self.NU):
                    u[k] = q[i + k * self.NTS]

                # call fortran backend to calculate derivatives of constraint functions
                self.backend_gfcn.ffcn_dot(t, x, x_dot, f, f_dot, p, p_dot, u, u_dot)

                # store gradient
                for k in xrange(0, self.NU):
                    dp[i + k * self.NTS] = f[k]

                # store gradient
                for k in xrange(0, self.NG):
                    dp[i + k * self.NTS, :] = f_dot[k, :]

        return dp

    """
    ===============================================================================
    """

    def c_dq(self, xs, xs_dot, xs_ddot, p, q, s):

        """

        ...

        input:

        output:

        TODO:

        """

        dq = None

        if self.NG > 0:
            # allocate memory
            dq      = np.zeros((self.NC, self.NQ))
            t       = np.zeros((1,))
            x       = np.zeros((self.NX,))
            x_dot   = np.zeros((self.NX, self.NU))
            f       = np.zeros((self.NG,))
            f_dot   = np.zeros((self.NG, self.NU))
            p_dot   = np.zeros((self.NP, self.NU))
            u       = np.zeros((self.NU,))
            u_dot   = np.eye(self.NU)

            # loop through all time step
            for i in xrange(0, self.NTS):

                # choose current controls of this time step
                for k in xrange(0, self.NU):
                    u[k] = q[i + k * self.NTS]

                # call fortran backend to calculate derivatives of constraint functions
                self.backend_gfcn.ffcn_dot(t, x, x_dot, f, f_dot, p, p_dot, u, u_dot)

                # store gradient
                for j in xrange(0, self.NU):
                    for k in xrange(0, self.NG):
                        dq[i + k * self.NTS, i + j * self.NTS] = f_dot[k, j]

        return dq

    """
    ===============================================================================
    """

    def c_dqdq(self, xs, xs_dot, xs_ddot, p, q, s):

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

    def c_dqdp(self, xs, xs_dot, xs_ddot, p, q, s):

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

    def c_dpdp(self, xs, xs_dot, xs_ddot, p, q, s):

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

    def obj(self, xs, xs_dot, xs_ddot, p, q, s):

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

    def obj_dp(self, xs, xs_dot, xs_ddot, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        return self.sign * xs_dot[-1, -1, :]

    """
    ===============================================================================
    """

    def obj_dq(self, xs, xs_dot, xs_ddot, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        return self.sign * xs_dot[-1, -1, :]

    """
    ===============================================================================
    """

    def obj_dx0(self, xs, xs_dot, xs_ddot, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        return self.sign * xs_dot[-1, -1, :]

    """
    ===============================================================================
    """

    def obj_dqdq(self, xs, xs_dot, xs_ddot, p, q, s):

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

    def obj_dpdp(self, xs, xs_dot, xs_ddot, p, q, s):

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

    def obj_dqdp(self, xs, xs_dot, xs_ddot, p, q, s):

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