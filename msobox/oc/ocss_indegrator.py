# -*- coding: utf-8 -*-

"""
===============================================================================

optimal control problem discretized by INDegrator for single shooting ...

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

class OCSS_indegrator(object):

    """

    provides functionalities for ...

    """

    """
    ===============================================================================
    """

    def __init__(self, path, minormax, NX, NG, NP, NU, bc, ts):

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
            q_indegrator[i, :, 0] = q[self.NTS * i:self.NTS * (i + 1)]

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
        s_indegrator = np.zeros((2, self.NX))

        # convert shooting variables from one-dimensional array to INDegrator specific format
        for i in xrange(0, 2):
            s_indegrator[i, :] = s[i * self.NX:(i + 1) * self.NX]

        return s_indegrator

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
        q = self.convert_q(q)
        s = self.convert_s(s)

        # set initial conditions
        x0 = s[0, :]

        # integrate
        self.integrator.zo_forward(self.ts, x0, p, q)

        return self.integrator.xs

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
        q = self.convert_q(q)
        s = self.convert_s(s)

        # set initial conditions
        x0 = s[0, :]

        # set up directions for differentation
        x0_dot  = np.zeros((self.NX, self.NP))
        p_dot   = np.eye(self.NP)
        q_dot   = np.zeros(q.shape + (self.NP,))

        # integrate
        self.integrator.fo_forward_xpu(self.ts,
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
        q = self.convert_q(q)
        s = self.convert_s(s)

        # set initial conditions
        x0 = s[0, :]

        # set up directions for differentation
        x0_dot                                  = np.zeros((self.NX, self.NQ))
        p_dot                                   = np.zeros((self.NP, self.NQ))
        q_dot                                   = np.zeros(q.shape + (self.NQ,))
        q_dot.reshape((self.NQ, self.NQ))[:, :] = np.eye(self.NQ)

        # integrate
        self.integrator.fo_forward_xpu(self.ts,
                                       x0, x0_dot,
                                       p, p_dot,
                                       q, q_dot)

        return self.integrator.xs, self.integrator.xs_dot

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

        # convert controls and shooting variables to INDegrator specific format
        q = self.convert_q(q)
        s = self.convert_s(s)

        # set initial conditions
        x0 = s[0, :]

        # set up directions for differentation
        x0_dot  = np.eye(self.NX)
        p_dot   = np.zeros((self.NP, self.NX))
        q_dot   = np.zeros(q.shape + (self.NX,))

        # integrate
        self.integrator.fo_forward_xpu(self.ts,
                                       x0, x0_dot,
                                       p, p_dot,
                                       q, q_dot)

        return self.integrator.xs, self.integrator.xs_dot

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
        q = self.convert_q(q)
        s = self.convert_s(s)

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
        self.integrator.so_forward_xpu_xpu(self.ts,
                                           x0, x0_dot, x0_dot, x0_ddot,
                                           p, p_dot, p_dot, p_ddot,
                                           q, q_dot, q_dot, q_ddot)

        return self.integrator.xs, self.integrator.xs_dot, self.integrator.xs_ddot

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
        q = self.convert_q(q)
        s = self.convert_s(s)

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
        self.integrator.so_forward_xpu_xpu(self.ts,
                                           x0, x0_dot, x0_dot, x0_ddot,
                                           p, p_dot, p_dot, p_ddot,
                                           q, q_dot, q_dot, q_ddot)

        return self.integrator.xs, self.integrator.xs_dot, self.integrator.xs_ddot

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

        # convert controls and shooting variables to INDegrator specific format
        q = self.convert_q(q)
        s = self.convert_s(s)

        # set initial conditions
        x0 = s[0, :]

        # set up directions for differentation
        x0_dot1                                     = np.zeros((self.NX, self.NQ))
        x0_dot2                                     = np.zeros((self.NX, self.NP))
        x0_ddot                                     = np.zeros(x0_dot1.shape + (self.NP,))
        p_dot1                                      = np.zeros((self.NP, self.NQ))
        p_dot2                                      = np.eye(self.NP)
        p_ddot                                      = np.zeros(p_dot1.shape + (self.NP,))
        q_dot1                                      = np.zeros(q.shape + (self.NQ,))
        q_dot1.reshape((self.NQ, self.NQ))[:, :]    = np.eye(self.NQ)
        q_dot2                                      = np.zeros(q.shape + (self.NP,))
        q_ddot                                      = np.zeros(q_dot1.shape + (self.NP,))

        # integrate
        self.integrator.so_forward_xpu_xpu(self.ts,
                                           x0, x0_dot2, x0_dot1, x0_ddot,
                                           p, p_dot2, p_dot1, p_ddot,
                                           q, q_dot2, q_dot1, q_ddot)

        return self.integrator.xs, self.integrator.xs_dot, self.integrator.xs_ddot

    """
    ===============================================================================
    """

    def c(self, xs, xs_dot, xs_ddot, p, q, s):

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
            f = np.zeros((self.NG,))
            u = np.zeros((self.NU,))

            # loop through all time steps
            for i in xrange(0, self.NTS):

                # set time, state and controls for this time step
                t[0]    = self.ts[i]
                x       = xs[i, :]

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
            f       = np.zeros((self.NG,))
            f_dot   = np.zeros((self.NG, self.NP))
            p_dot   = np.zeros((self.NP, self.NP))
            u       = np.zeros((self.NU,))
            u_dot   = np.zeros((self.NU, self.NP))

            # loop through all time steps
            for i in xrange(0, self.NTS):

                # set time, state and controls for this time step
                t[0]    = self.ts[i]
                x       = xs[i, :]
                x_dot   = np.reshape(xs_dot[i, :, :], x_dot.shape)

                for l in xrange(0, self.NU):
                    u[l] = q[i + l * self.NTS]

                # call fortran backend to calculate derivatives of constraint functions
                self.backend_gfcn.ffcn_dot(t, x, x_dot, f, f_dot, p, p_dot, u, u_dot)

                # store gradient
                for j in xrange(0, self.NP):
                    for k in xrange(0, self.NG):
                        dp[i + k * self.NP, j] = f_dot[k, j]

        return dp

    """
    ===============================================================================
    """

    def c_dq(self, xs, xs_dot, xs_ddot, p, q, s):

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
            f       = np.zeros((self.NG,))
            f_dot   = np.zeros((self.NG, self.NU))
            p_dot   = np.zeros((self.NP, self.NU))
            u       = np.zeros((self.NU,))
            u_dot   = np.zeros((self.NU, self.NU))

            # loop through all time steps and controls
            for i in xrange(0, self.NTS):
                for j in xrange(0, self.NTS):

                    # set time, state and controls for this time step
                    t[0]    = self.ts[i]
                    x       = xs[i, :]
                    x_dot   = np.reshape(xs_dot[i, :, j], x_dot.shape)

                    for k in xrange(0, self.NU):
                        u[k] = q[i + k * self.NTS]

                    if i == j:
                        u_dot = np.eye(self.NU)
                    else:
                        u_dot = np.zeros((self.NU, self.NU))

                    # call fortran backend to calculate derivatives of constraint functions
                    self.backend_gfcn.ffcn_dot(t, x, x_dot, f, f_dot, p, p_dot, u, u_dot)

                    # store gradient
                    for l in xrange(0, self.NU):
                        for m in xrange(0, self.NG):
                            dq[i + m * self.NTS, j + l * self.NTS] = f_dot[m, l]

        return dq

    """
    ===============================================================================
    """

    def c_dx0(self, xs, xs_dot, xs_ddot, p, q, s):

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
            dx0     = np.zeros((self.NC, self.NX))
            t       = np.zeros((1,))
            x       = np.zeros((self.NX,))
            x_dot   = np.eye(self.NX)
            f       = np.zeros((self.NG,))
            f_dot   = np.zeros((self.NG, self.NX))
            p_dot   = np.zeros((self.NP, self.NX))
            u       = np.zeros((self.NU,))
            u_dot   = np.zeros((self.NU, self.NX))

            # loop through all time step
            for i in xrange(0, self.NTS):

                    # set time, state and controls for this time step
                    t[0]    = self.ts[i]
                    x       = xs[i, :]
                    x_dot   = np.reshape(xs_dot[i, :, :], x_dot.shape)

                    for l in xrange(0, self.NU):
                        u[l] = q[i + l * self.NTS]

                    # call fortran backend to calculate derivatives of constraint functions
                    self.backend_gfcn.ffcn_dot(t, x, x_dot, f, f_dot, p, p_dot, u, u_dot)

                    # store gradient
                    for j in xrange(0, self.NX):
                        for k in xrange(0, self.NG):
                            dx0[i + k * self.NX, j] = f_dot[k, j]

        return dx0

    """
    ===============================================================================
    """

    def c_dqdq(self, xs, xs_dot, xs_ddot, p, q, s):

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
            dqdq = np.zeros((self.NC, self.NQ, self.NQ))

        return dqdq

    """
    ===============================================================================
    """

    def c_dqdp(self, xs, xs_dot, xs_ddot, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

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