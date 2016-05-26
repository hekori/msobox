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

    def integrate_interval_ds(self, interval, p, q, s):

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

        # allocate memory
        xs_dot = np.zeros((self.NTSI, self.NX, self.NS))

        # integrate
        self.integrator.fo_forward(tsi,
                                   x0, x0_dot,
                                   p, p_dot,
                                   q_interval, q_dot)

        xs_dot[:, :, interval * self.NX:(interval + 1) * self.NX] = self.integrator.xs_dot

        return self.integrator.xs, xs_dot

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
        q_interval = q[:, interval, 0]

        # set up directions for differentation
        x0_dot = np.zeros((self.NX, self.NP))
        p_dot  = np.eye(self.NP)
        q_dot  = np.zeros((self.NU, self.NP))

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

    def integrate_interval_dsds(self, interval, p, q, s):

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
        x0_dot  = np.eye(self.NX)
        x0_ddot = np.zeros(x0_dot.shape + (self.NX,))
        p_dot   = np.zeros((self.NP, self.NX))
        p_ddot  = np.zeros(p_dot.shape + (self.NX,))
        q_dot   = np.zeros((self.NU, self.NX))
        q_ddot  = np.zeros(q_dot.shape + (self.NX,))

        # allocate memory
        xs_dot1 = np.zeros((self.NTSI, self.NX, self.NS))
        xs_dot2 = np.zeros((self.NTSI, self.NX, self.NS))
        xs_ddot = np.zeros((self.NTSI, self.NX, self.NS, self.NS))

        # integrate
        self.integrator.so_forward(tsi,
                                   x0, x0_dot, x0_dot, x0_ddot,
                                   p, p_dot, p_dot, p_ddot,
                                   q_interval, q_dot, q_dot, q_ddot)

        xs_dot1[:, :, interval * self.NX:(interval + 1) * self.NX]                                              = self.integrator.xs_dot1
        xs_dot2[:, :, interval * self.NX:(interval + 1) * self.NX]                                              = self.integrator.xs_dot2
        xs_ddot[:, :, interval * self.NX:(interval + 1) * self.NX, interval * self.NX:(interval + 1) * self.NX] = self.integrator.xs_ddot

        return self.integrator.xs, xs_dot1, xs_dot2, xs_ddot

    """
    ===============================================================================
    """

    def integrate_interval_dpdp(self, interval, p, q, s):

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
        x0_dot  = np.zeros((self.NX, self.NP))
        x0_ddot = np.zeros(x0_dot.shape + (self.NP,))
        p_dot   = np.eye(self.NP)
        p_ddot  = np.zeros(p_dot.shape + (self.NP,))
        q_dot   = np.zeros((self.NU, self.NP))
        q_ddot  = np.zeros(q_dot.shape + (self.NP,))

        # integrate
        self.integrator.so_forward(tsi,
                                   x0, x0_dot, x0_dot, x0_ddot,
                                   p, p_dot, p_dot, p_ddot,
                                   q_interval, q_dot, q_dot, q_ddot)

        return self.integrator.xs, self.integrator.xs_dot1, self.integrator.xs_dot2, self.integrator.xs_ddot

    """
    ===============================================================================
    """

    def integrate_interval_dqdq(self, interval, p, q, s):

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
        x0_dot  = np.zeros((self.NX, self.NU))
        x0_ddot = np.zeros(x0_dot.shape + (self.NU,))
        p_dot   = np.zeros((self.NP, self.NU))
        p_ddot  = np.zeros(p_dot.shape + (self.NU,))
        q_dot   = np.eye(self.NU)
        q_ddot  = np.zeros(q_dot.shape + (self.NU,))

        # allocate memory
        xs_dot1 = np.zeros((self.NTSI, self.NX, self.NQ))
        xs_dot2 = np.zeros((self.NTSI, self.NX, self.NQ))
        xs_ddot = np.zeros((self.NTSI, self.NX, self.NQ, self.NQ))

        # integrate
        self.integrator.so_forward(tsi,
                                   x0, x0_dot, x0_dot, x0_ddot,
                                   p, p_dot, p_dot, p_ddot,
                                   q_interval, q_dot, q_dot, q_ddot)

        xs_dot1[:, :, interval * self.NU:(interval + 1) * self.NU]                                              = self.integrator.xs_dot1
        xs_dot2[:, :, interval * self.NU:(interval + 1) * self.NU]                                              = self.integrator.xs_dot2
        xs_ddot[:, :, interval * self.NU:(interval + 1) * self.NU, interval * self.NU:(interval + 1) * self.NU] = self.integrator.xs_ddot

        return self.integrator.xs, xs_dot1, xs_dot2, xs_ddot

    """
    ===============================================================================
    """

    def integrate_interval_dsdp(self, interval, p, q, s):

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
        x0_dot1 = np.eye(self.NX)
        x0_dot2 = np.zeros((self.NX, self.NP))
        x0_ddot = np.zeros(x0_dot1.shape + (self.NP,))
        p_dot1  = np.zeros((self.NP, self.NX))
        p_dot2  = np.eye(self.NP)
        p_ddot  = np.zeros(p_dot1.shape + (self.NP,))
        q_dot1  = np.zeros((self.NU, self.NX))
        q_dot2  = np.zeros((self.NU, self.NP))
        q_ddot  = np.zeros(q_dot1.shape + (self.NP,))

        # allocate memory
        xs_dot1 = np.zeros((self.NTSI, self.NX, self.NS))
        xs_ddot = np.zeros((self.NTSI, self.NX, self.NS, self.NP))

        # integrate
        self.integrator.so_forward(tsi,
                                   x0, x0_dot2, x0_dot1, x0_ddot,
                                   p, p_dot2, p_dot1, p_ddot,
                                   q_interval, q_dot2, q_dot1, q_ddot)

        xs_dot1[:, :, interval * self.NX:(interval + 1) * self.NX]    = self.integrator.xs_dot1
        xs_ddot[:, :, interval * self.NX:(interval + 1) * self.NX, :] = self.integrator.xs_ddot

        return self.integrator.xs, xs_dot1, self.integrator.xs_dot2, xs_ddot

    """
    ===============================================================================
    """

    def integrate_interval_dsdq(self, interval, p, q, s):

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
        x0_dot1 = np.eye(self.NX)
        x0_dot2 = np.zeros((self.NX, self.NU))
        x0_ddot = np.zeros(x0_dot1.shape + (self.NU,))
        p_dot1  = np.zeros((self.NP, self.NX))
        p_dot2  = np.zeros((self.NP, self.NU))
        p_ddot  = np.zeros(p_dot1.shape + (self.NU,))
        q_dot1  = np.zeros((self.NU, self.NX))
        q_dot2  = np.eye(self.NU)
        q_ddot  = np.zeros(q_dot1.shape + (self.NU,))

        # allocate memory
        xs_dot1 = np.zeros((self.NTSI, self.NX, self.NS))
        xs_dot2 = np.zeros((self.NTSI, self.NX, self.NQ))
        xs_ddot = np.zeros((self.NTSI, self.NX, self.NS, self.NQ))

        # integrate
        self.integrator.so_forward(tsi,
                                   x0, x0_dot2, x0_dot1, x0_ddot,
                                   p, p_dot2, p_dot1, p_ddot,
                                   q_interval, q_dot2, q_dot1, q_ddot)

        xs_dot1[:, :, interval * self.NX:(interval + 1) * self.NX]                                              = self.integrator.xs_dot1
        xs_dot2[:, :, interval * self.NU:(interval + 1) * self.NU]                                              = self.integrator.xs_dot2
        xs_ddot[:, :, interval * self.NX:(interval + 1) * self.NX, interval * self.NU:(interval + 1) * self.NU] = self.integrator.xs_ddot

        return self.integrator.xs, xs_dot1, xs_dot2, xs_ddot

    """
    ===============================================================================
    """

    def integrate_interval_dpdq(self, interval, p, q, s):

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
        x0_dot1 = np.zeros((self.NX, self.NP))
        x0_dot2 = np.zeros((self.NX, self.NU))
        x0_ddot = np.zeros(x0_dot1.shape + (self.NU,))
        p_dot1  = np.eye(self.NP)
        p_dot2  = np.zeros((self.NP, self.NU))
        p_ddot  = np.zeros(p_dot1.shape + (self.NU,))
        q_dot1  = np.zeros((self.NU, self.NP))
        q_dot2  = np.eye(self.NU)
        q_ddot  = np.zeros(q_dot1.shape + (self.NU,))

        # allocate memory
        xs_dot2 = np.zeros((self.NTSI, self.NX, self.NQ))
        xs_ddot = np.zeros((self.NTSI, self.NX, self.NP, self.NQ))

        # integrate
        self.integrator.so_forward(tsi,
                                   x0, x0_dot2, x0_dot1, x0_ddot,
                                   p, p_dot2, p_dot1, p_ddot,
                                   q_interval, q_dot2, q_dot1, q_ddot)

        xs_dot2[:, :, interval * self.NU:(interval + 1) * self.NU]    = self.integrator.xs_dot2
        xs_ddot[:, :, :, interval * self.NU:(interval + 1) * self.NU] = self.integrator.xs_ddot

        return self.integrator.xs, self.integrator.xs_dot1, xs_dot2, xs_ddot

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
            xs[i, :, :], xs_dot[i, :, :, :] = self.integrate_interval_ds(i, p, q, s)

        return xs, xs_dot

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
        xs     = np.zeros((self.NTS - 1, self.NTSI, self.NX))
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

    def integrate_dsds(self, p, q, s):

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
        xs      = np.zeros((self.NTS - 1, self.NTSI, self.NX))
        xs_dot1 = np.zeros((self.NTS - 1, self.NTSI, self.NX, self.NS))
        xs_dot2 = np.zeros((self.NTS - 1, self.NTSI, self.NX, self.NS))
        xs_ddot = np.zeros((self.NTS - 1, self.NTSI, self.NX, self.NS, self.NS))

        # integrate on shooting intervals
        for i in xrange(0, self.NTS - 1):
            xs[i, :, :], xs_dot1[i, :, :, :], xs_dot2[i, :, :, :], xs_ddot[i, :, :, :, :] = self.integrate_interval_dsds(i, p, q, s)

        return xs, xs_dot1, xs_dot2, xs_ddot

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

        # allocate memory
        xs      = np.zeros((self.NTS - 1, self.NTSI, self.NX))
        xs_dot1 = np.zeros((self.NTS - 1, self.NTSI, self.NX, self.NP))
        xs_dot2 = np.zeros((self.NTS - 1, self.NTSI, self.NX, self.NP))
        xs_ddot = np.zeros((self.NTS - 1, self.NTSI, self.NX, self.NP, self.NP))

        # integrate on shooting intervals
        for i in xrange(0, self.NTS - 1):
            xs[i, :, :], xs_dot1[i, :, :, :], xs_dot2[i, :, :, :], xs_ddot[i, :, :, :, :] = self.integrate_interval_dpdp(i, p, q, s)

        return xs, xs_dot1, xs_dot2, xs_ddot

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

        # allocate memory
        xs      = np.zeros((self.NTS - 1, self.NTSI, self.NX))
        xs_dot1 = np.zeros((self.NTS - 1, self.NTSI, self.NX, self.NQ))
        xs_dot2 = np.zeros((self.NTS - 1, self.NTSI, self.NX, self.NQ))
        xs_ddot = np.zeros((self.NTS - 1, self.NTSI, self.NX, self.NQ, self.NQ))

        # integrate on shooting intervals
        for i in xrange(0, self.NTS - 1):
            xs[i, :, :], xs_dot1[i, :, :, :], xs_dot2[i, :, :, :], xs_ddot[i, :, :, :, :] = self.integrate_interval_dqdq(i, p, q, s)

        return xs, xs_dot1, xs_dot2, xs_ddot

    """
    ===============================================================================
    """

    def integrate_dsdp(self, p, q, s):

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
        xs      = np.zeros((self.NTS - 1, self.NTSI, self.NX))
        xs_dot1 = np.zeros((self.NTS - 1, self.NTSI, self.NX, self.NS))
        xs_dot2 = np.zeros((self.NTS - 1, self.NTSI, self.NX, self.NP))
        xs_ddot = np.zeros((self.NTS - 1, self.NTSI, self.NX, self.NS, self.NP))

        # integrate on shooting intervals
        for i in xrange(0, self.NTS - 1):
            xs[i, :, :], xs_dot1[i, :, :, :], xs_dot2[i, :, :, :], xs_ddot[i, :, :, :, :] = self.integrate_interval_dsdp(i, p, q, s)

        return xs, xs_dot1, xs_dot2, xs_ddot

    """
    ===============================================================================
    """

    def integrate_dsdq(self, p, q, s):

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
        xs      = np.zeros((self.NTS - 1, self.NTSI, self.NX))
        xs_dot1 = np.zeros((self.NTS - 1, self.NTSI, self.NX, self.NS))
        xs_dot2 = np.zeros((self.NTS - 1, self.NTSI, self.NX, self.NQ))
        xs_ddot = np.zeros((self.NTS - 1, self.NTSI, self.NX, self.NS, self.NQ))

        # integrate on shooting intervals
        for i in xrange(0, self.NTS - 1):
            xs[i, :, :], xs_dot1[i, :, :, :], xs_dot2[i, :, :, :], xs_ddot[i, :, :, :, :] = self.integrate_interval_dsdq(i, p, q, s)

        return xs, xs_dot1, xs_dot2, xs_ddot

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

        # allocate memory
        xs      = np.zeros((self.NTS - 1, self.NTSI, self.NX))
        xs_dot1 = np.zeros((self.NTS - 1, self.NTSI, self.NX, self.NP))
        xs_dot2 = np.zeros((self.NTS - 1, self.NTSI, self.NX, self.NQ))
        xs_ddot = np.zeros((self.NTS - 1, self.NTSI, self.NX, self.NP, self.NQ))

        # integrate on shooting intervals
        for i in xrange(0, self.NTS - 1):
            xs[i, :, :], xs_dot1[i, :, :, :], xs_dot2[i, :, :, :], xs_ddot[i, :, :, :, :] = self.integrate_interval_dpdq(i, p, q, s)

        return xs, xs_dot1, xs_dot2, xs_ddot

    """
    ===============================================================================
    """

    def mc(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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
        mc = np.zeros((self.NS - self.NX,))

        # evaluate matching conditions
        for i in xrange(0, self.NTS - 1):
            mc[i * self.NX:(i + 1) * self.NX] = xs[i, -1, :] - self.s_array2ind(s)[i + 1, :]

        return mc

    """
    ===============================================================================
    """

    def mc_ds(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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
        mc    = np.zeros((self.NS - self.NX,))
        mc_ds = np.zeros((self.NS - self.NX, self.NS))

        # evaluate matching conditions
        for i in xrange(0, self.NTS - 1):
            mc[i * self.NX:(i + 1) * self.NX]                                         = xs[i, -1, :] - self.s_array2ind(s)[i + 1, :]
            mc_ds[i * self.NX:(i + 1) * self.NX, :]                                   = xs_dot1[i, -1, :, :]
            mc_ds[i * self.NX:(i + 1) * self.NX, (i + 1) * self.NX:(i + 2) * self.NX] = -np.eye(self.NX)

        return mc, mc_ds

    """
    ===============================================================================
    """

    def mc_dp(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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
        mc    = np.zeros((self.NS - self.NX,))
        mc_dp = np.zeros((self.NS - self.NX, self.NP))

        # evaluate matching conditions
        for i in xrange(0, self.NTS - 1):
            mc[i * self.NX:(i + 1) * self.NX]       = xs[i, -1, :] - self.s_array2ind(s)[i + 1, :]
            mc_dp[i * self.NX:(i + 1) * self.NX, :] = xs_dot1[i, -1, :, :]

        return mc, mc_dp

    """
    ===============================================================================
    """

    def mc_dq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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
        mc    = np.zeros((self.NS - self.NX,))
        mc_dq = np.zeros((self.NS - self.NX, self.NQ))

        # evaluate matching conditions
        for i in xrange(0, self.NTS - 1):
            mc[i * self.NX:(i + 1) * self.NX]       = xs[i, -1, :] - self.s_array2ind(s)[i + 1, :]
            mc_dq[i * self.NX:(i + 1) * self.NX, :] = xs_dot1[i, -1, :, :]

        return mc, mc_dq

    """
    ===============================================================================
    """

    def mc_dsds(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        c    = None
        dp   = None
        dq   = None
        dpdq = None

        return c, dp, dq, dpdq

    """
    ===============================================================================
    """

    def mc_dpdp(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        c    = None
        dp   = None
        dq   = None
        dpdq = None

        return c, dp, dq, dpdq

    """
    ===============================================================================
    """

    def mc_dqdq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        c    = None
        dp   = None
        dq   = None
        dpdq = None

        return c, dp, dq, dpdq

    """
    ===============================================================================
    """

    def mc_dsdp(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        c    = None
        dp   = None
        dq   = None
        dpdq = None

        return c, dp, dq, dpdq

    """
    ===============================================================================
    """

    def mc_dsdq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        c    = None
        dp   = None
        dq   = None
        dpdq = None

        return c, dp, dq, dpdq

    """
    ===============================================================================
    """

    def mc_dpdq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        c    = None
        dp   = None
        dq   = None
        dpdq = None

        return c, dp, dq, dpdq

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
            x = np.zeros((self.NX,))
            g = np.zeros((self.NG,))
            u = np.zeros((self.NU,))

            # loop through all time steps
            for i in xrange(0, self.NTS):

                # set state and controls for this time step
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

        c    = None
        c_ds = None

        if self.NG > 0:

            # allocate memory
            c     = np.zeros((self.NC,))
            c_ds  = np.zeros((self.NC, self.NS))

            x     = np.zeros((self.NX,))
            x_dot = np.zeros((self.NX, self.NX))
            g     = np.zeros((self.NG,))
            g_dot = np.zeros((self.NG, self.NX))
            p_dot = np.zeros((self.NP, self.NX))
            u     = np.zeros((self.NU,))
            u_dot = np.zeros((self.NU, self.NX))

            # loop through all time steps
            for i in xrange(0, self.NTS):

                # set state and controls for this time step
                if i == self.NTS - 1:
                    x     = xs[i - 1, -1, :]
                    x_dot = np.reshape(xs_dot1[i - 1, -1, :, (i - 1) * self.NX:i * self.NX], x_dot.shape)

                else:
                    x     = xs[i, 0, :]
                    x_dot = np.reshape(xs_dot1[i, 0, :, i * self.NX:(i + 1) * self.NX], x_dot.shape)

                for l in xrange(0, self.NU):
                    u[l] = q[i + l * self.NTS]

                # call fortran backend to calculate derivatives of constraint functions
                self.backend_fortran.gfcn_dot(g, g_dot,
                                              self.ts[i:i + 1],
                                              x, x_dot,
                                              p, p_dot,
                                              u, u_dot)

                # store derivatives
                if i == self.NTS - 1:
                    for m in xrange(0, self.NG):
                        c[i + m * self.NTS] = g[m]

                        for l in xrange(0, self.NX):
                            c_ds[i + m * self.NTS, l + (i - 1) * self.NX] = g_dot[m, l]

                else:
                    for m in xrange(0, self.NG):
                        c[i + m * self.NTS] = g[m]

                        for l in xrange(0, self.NX):
                            c_ds[i + m * self.NTS, l + i * self.NX] = g_dot[m, l]

        return c, c_ds

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

        c    = None
        c_dp = None

        if self.NG > 0:

            # allocate memory
            c     = np.zeros((self.NC,))
            c_dp  = np.zeros((self.NC, self.NP))

            x     = np.zeros((self.NX,))
            x_dot = np.zeros((self.NX, self.NP))
            g     = np.zeros((self.NG,))
            g_dot = np.zeros((self.NG, self.NP))
            p_dot = np.eye(self.NP)
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
                self.backend_fortran.gfcn_dot(g, g_dot,
                                              self.ts[i:i + 1],
                                              x, x_dot,
                                              p, p_dot,
                                              u, u_dot)

                # store gradient
                for k in xrange(0, self.NG):
                    c[i + k * self.NTS]       = g[k]
                    c_dp[i + k * self.NTS, :] = g_dot[k, :]


        return c, c_dp

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

        c    = None
        c_dq = None

        if self.NG > 0:

            # allocate memory
            c     = np.zeros((self.NC,))
            c_dq  = np.zeros((self.NC, self.NQ))

            x     = np.zeros((self.NX,))
            x_dot = np.zeros((self.NX, self.NU))
            g     = np.zeros((self.NG,))
            g_dot = np.zeros((self.NG, self.NU))
            p_dot = np.zeros((self.NP, self.NU))
            u     = np.zeros((self.NU,))
            u_dot = np.zeros((self.NU, self.NU))

            # set gradient for constraints of last time step
            x     = xs[self.NTS - 2, -1, :]
            x_dot = np.reshape(xs_dot1[self.NTS - 2, -1, :, self.NTS - 2], x_dot.shape)

            for k in xrange(0, self.NU):
                u[k] = q[self.NTS - 1 + k * self.NTS]

            # call fortran backend to calculate derivatives of constraint functions
            self.backend_fortran.gfcn_dot(g, g_dot,
                                          self.ts[self.NTS - 1:self.NTS],
                                          x, x_dot,
                                          p, p_dot,
                                          u, u_dot)

            # store derivatives
            for k in xrange(0, self.NG):
                c[self.NTS - 1 + k * self.NTS] = g[k]

                for l in xrange(0, self.NU):
                    c_dq[self.NTS - 1 + k * self.NTS, self.NTS - 2 + l * self.NTS] = g_dot[k, l]

        return c, c_dq

    """
    ===============================================================================
    """

    def c_dsds(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

        """

        ...

        input

        output:

        TODO:

        """

        c      = None
        c_ds1  = None
        c_ds2  = None
        c_dsds = None

        if self.NG > 0:

            # allocate memory
            c      = np.zeros((self.NC,))
            c_ds1  = np.zeros((self.NC, self.NS))
            c_ds2  = np.zeros((self.NC, self.NS))
            c_dsds = np.zeros((self.NC, self.NS, self.NS))

            x      = np.zeros((self.NX,))
            x_dot1 = np.zeros((self.NX, self.NX))
            x_dot2 = np.zeros((self.NX, self.NX))
            x_ddot = np.zeros((self.NX, self.NX, self.NX))
            g      = np.zeros((self.NG,))
            g_dot1 = np.zeros((self.NG, self.NX))
            g_dot2 = np.zeros((self.NG, self.NX))
            g_ddot = np.zeros((self.NG, self.NX, self.NX))
            p_dot  = np.zeros((self.NP, self.NX))
            p_ddot = np.zeros((self.NP, self.NX, self.NX))
            u      = np.zeros((self.NU,))
            u_dot  = np.zeros((self.NU, self.NX))
            u_ddot = np.zeros((self.NU, self.NX, self.NX))

            # loop through all time steps
            for i in xrange(0, self.NTS):

                # set state and controls for this time step
                if i == self.NTS - 1:
                    x      = xs[i - 1, -1, :]
                    x_dot1 = np.reshape(xs_dot1[i - 1, -1, :, (i - 1) * self.NX:i * self.NX], x_dot1.shape)
                    x_dot2 = np.reshape(xs_dot2[i - 1, -1, :, (i - 1) * self.NX:i * self.NX], x_dot2.shape)
                    x_ddot = np.reshape(xs_ddot[i - 1, -1, :, (i - 1) * self.NX:i * self.NX, (i - 1) * self.NX:i * self.NX], x_ddot.shape)

                else:
                    x      = xs[i, 0, :]
                    x_dot1 = np.reshape(xs_dot1[i, 0, :, i * self.NX:(i + 1) * self.NX], x_dot1.shape)
                    x_dot2 = np.reshape(xs_dot2[i, 0, :, i * self.NX:(i + 1) * self.NX], x_dot2.shape)
                    x_ddot = np.reshape(xs_ddot[i, 0, :, i * self.NX:(i + 1) * self.NX, i * self.NX:(i + 1) * self.NX], x_ddot.shape)

                for l in xrange(0, self.NU):
                    u[l] = q[i + l * self.NTS]

                # call fortran backend to calculate derivatives of constraint functions
                self.backend_fortran.gfcn_ddot(g, g_dot2, g_dot1, g_ddot,
                                               self.ts[i:i + 1],
                                               x, x_dot2, x_dot1, x_ddot,
                                               p, p_dot, p_dot, p_ddot,
                                               u, u_dot, u_dot, u_ddot)

                # store derivatives
                if i == self.NTS - 1:
                    for m in xrange(0, self.NG):
                        c[i + m * self.NTS] = g[m]

                        for l in xrange(0, self.NX):
                            c_ds1[i + m * self.NTS, l + (i - 1) * self.NX] = g_dot1[m, l]
                            c_ds2[i + m * self.NTS, l + (i - 1) * self.NX] = g_dot2[m, l]

                            for j in xrange(0, self.NX):
                                c_dsds[i + m * self.NTS, l + (i - 1) * self.NX, j + (i - 1) * self.NX] = g_ddot[m, l, j]

                else:
                    for m in xrange(0, self.NG):
                        c[i + m * self.NTS] = g[m]

                        for l in xrange(0, self.NX):
                            c_ds1[i + m * self.NTS, l + i * self.NX] = g_dot1[m, l]
                            c_ds2[i + m * self.NTS, l + i * self.NX] = g_dot2[m, l]

                            for j in xrange(0, self.NX):
                                c_dsds[i + m * self.NTS, l + i * self.NX, j + i * self.NX] = g_ddot[m, l, j]

        return c, c_ds1, c_ds2, c_dsds

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

        c      = None
        c_dp1  = None
        c_dp2  = None
        c_dpdp = None

        if self.NG > 0:

            # allocate memory
            c      = np.zeros((self.NC,))
            c_dp1  = np.zeros((self.NC, self.NP))
            c_dp2  = np.zeros((self.NC, self.NP))
            c_dpdp = np.zeros((self.NC, self.NP, self.NP))

            x      = np.zeros((self.NX,))
            x_dot1 = np.zeros((self.NX, self.NP))
            x_dot2 = np.zeros((self.NX, self.NP))
            x_ddot = np.zeros((self.NX, self.NP, self.NP))
            g      = np.zeros((self.NG,))
            g_dot1 = np.zeros((self.NG, self.NP))
            g_dot2 = np.zeros((self.NG, self.NP))
            g_ddot = np.zeros((self.NG, self.NP, self.NP))
            p_dot  = np.eye(self.NP)
            p_ddot = np.zeros((self.NP, self.NP, self.NP))
            u      = np.zeros((self.NU,))
            u_dot  = np.zeros((self.NU, self.NP))
            u_ddot = np.zeros((self.NU, self.NP, self.NP))

            # loop through all time steps
            for i in xrange(0, self.NTS):

                # set state and controls for this time step
                if i == self.NTS - 1:
                    x      = xs[i - 1, -1, :]
                    x_dot1 = np.reshape(xs_dot1[i - 1, -1, :, :], x_dot1.shape)
                    x_dot2 = np.reshape(xs_dot2[i - 1, -1, :, :], x_dot2.shape)
                    x_ddot = np.reshape(xs_ddot[i - 1, -1, :, :, :], x_ddot.shape)

                else:
                    x      = xs[i, 0, :]
                    x_dot1 = np.reshape(xs_dot1[i, 0, :, :], x_dot1.shape)
                    x_dot2 = np.reshape(xs_dot2[i, 0, :, :], x_dot2.shape)
                    x_ddot = np.reshape(xs_ddot[i, 0, :, :, :], x_ddot.shape)

                for l in xrange(0, self.NU):
                    u[l] = q[i + l * self.NTS]

                # call fortran backend to calculate derivatives of constraint functions
                self.backend_fortran.gfcn_ddot(g, g_dot2, g_dot1, g_ddot,
                                               self.ts[i:i + 1],
                                               x, x_dot2, x_dot1, x_ddot,
                                               p, p_dot, p_dot, p_ddot,
                                               u, u_dot, u_dot, u_ddot)

                # store gradient
                for k in xrange(0, self.NG):
                    c[i + k * self.NTS]            = g[k]
                    c_dp1[i + k * self.NTS, :]     = g_dot1[k, :]
                    c_dp2[i + k * self.NTS, :]     = g_dot2[k, :]
                    c_dpdp[i + k * self.NTS, :, :] = g_ddot[k, :, :]

        return c, c_dp1, c_dp2, c_dpdp

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

        c      = None
        c_dq1  = None
        c_dq2  = None
        c_dqdq = None

        if self.NG > 0:

            # allocate memory
            c      = np.zeros((self.NC,))
            c_dq1  = np.zeros((self.NC, self.NQ))
            c_dq2  = np.zeros((self.NC, self.NQ))
            c_dqdq = np.zeros((self.NC, self.NQ, self.NQ))

            x      = np.zeros((self.NX,))
            x_dot1 = np.zeros((self.NX, self.NU))
            x_dot2 = np.zeros((self.NX, self.NU))
            x_ddot = np.zeros((self.NX, self.NU, self.NU))
            g      = np.zeros((self.NG,))
            g_dot1 = np.zeros((self.NG, self.NU))
            g_dot2 = np.zeros((self.NG, self.NU))
            g_ddot = np.zeros((self.NG, self.NU, self.NU))
            p_dot  = np.zeros((self.NP, self.NU))
            p_ddot = np.zeros((self.NP, self.NU, self.NU))
            u      = np.zeros((self.NU,))
            u_dot  = np.eye(self.NU)
            u_ddot = np.zeros((self.NU, self.NU, self.NU))

            # set state and controls of last time step
            x      = xs[self.NTS - 2, -1, :]
            x_dot1 = np.reshape(xs_dot1[self.NTS - 2, -1, :, self.NTS - 2], x_dot1.shape)
            x_dot2 = np.reshape(xs_dot2[self.NTS - 2, -1, :, self.NTS - 2], x_dot2.shape)
            x_ddot = np.reshape(xs_ddot[self.NTS - 2, -1, :, self.NTS - 2, self.NTS - 2], x_ddot.shape)

            for k in xrange(0, self.NU):
                u[k] = q[self.NTS - 1 + k * self.NTS]

            # call fortran backend to calculate derivatives of constraint functions
            self.backend_fortran.gfcn_ddot(g, g_dot2, g_dot1, g_ddot,
                                           self.ts[self.NTS - 1:self.NTS],
                                           x, x_dot2, x_dot1, x_ddot,
                                           p, p_dot, p_dot, p_ddot,
                                           u, u_dot, u_dot, u_ddot)

            # store derivatives
            for k in xrange(0, self.NG):
                c[self.NTS - 1 + k * self.NTS] = g[k]

                for l in xrange(0, self.NU):
                    c_dq1[self.NTS - 1 + k * self.NTS, self.NTS - 2 + l * self.NTS] = g_dot1[k, l]
                    c_dq2[self.NTS - 1 + k * self.NTS, self.NTS - 2 + l * self.NTS] = g_dot2[k, l]

                    for m in xrange(0, self.NU):
                        c_dqdq[self.NTS - 1 + k * self.NTS, self.NTS - 2 + l * self.NTS, self.NTS - 2 + l * self.NTS] = g_ddot[k, l, m]

        return c, c_dq1, c_dq2, c_dqdq

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

        c      = None
        c_ds   = None
        c_dp   = None
        c_dsdp = None

        if self.NG > 0:

            # allocate memory
            c      = np.zeros((self.NC,))
            c_ds   = np.zeros((self.NC, self.NS))
            c_dp   = np.zeros((self.NC, self.NP))
            c_dsdp = np.zeros((self.NC, self.NS, self.NP))

            x      = np.zeros((self.NX,))
            x_dot1 = np.zeros((self.NX, self.NX))
            x_dot2 = np.zeros((self.NX, self.NP))
            x_ddot = np.zeros((self.NX, self.NX, self.NP))
            g      = np.zeros((self.NG,))
            g_dot1 = np.zeros((self.NG, self.NX))
            g_dot2 = np.zeros((self.NG, self.NP))
            g_ddot = np.zeros((self.NG, self.NX, self.NP))
            p_dot1 = np.zeros((self.NP, self.NX))
            p_dot2 = np.eye(self.NP)
            p_ddot = np.zeros((self.NP, self.NX, self.NP))
            u      = np.zeros((self.NU,))
            u_dot  = np.zeros((self.NU, self.NX))
            u_ddot = np.zeros((self.NU, self.NX, self.NP))

            # loop through all time steps
            for i in xrange(0, self.NTS):

                # set state and controls for this time step
                if i == self.NTS - 1:
                    x      = xs[i - 1, -1, :]
                    x_dot1 = np.reshape(xs_dot1[i - 1, -1, :, (i - 1) * self.NX:i * self.NX], x_dot1.shape)
                    x_dot2 = np.reshape(xs_dot2[i - 1, -1, :, :], x_dot2.shape)
                    x_ddot = np.reshape(xs_ddot[i - 1, -1, :, (i - 1) * self.NX:i * self.NX, :], x_ddot.shape)

                else:
                    x      = xs[i, 0, :]
                    x_dot1 = np.reshape(xs_dot1[i, 0, :, i * self.NX:(i + 1) * self.NX], x_dot1.shape)
                    x_dot2 = np.reshape(xs_dot2[i, 0, :, :], x_dot2.shape)
                    x_ddot = np.reshape(xs_ddot[i, 0, :, i * self.NX:(i + 1) * self.NX, :], x_ddot.shape)

                for l in xrange(0, self.NU):
                    u[l] = q[i + l * self.NTS]

                # call fortran backend to calculate derivatives of constraint functions
                self.backend_fortran.gfcn_ddot(g, g_dot2, g_dot1, g_ddot,
                                               self.ts[i:i + 1],
                                               x, x_dot2, x_dot1, x_ddot,
                                               p, p_dot2, p_dot1, p_ddot,
                                               u, u_dot, u_dot, u_ddot)

                # store derivatives
                if i == self.NTS - 1:
                    for m in xrange(0, self.NG):
                        c[i + m * self.NTS]       = g[m]
                        c_dp[i + m * self.NTS, :] = g_dot2[m, :]

                        for l in xrange(0, self.NX):
                            c_ds[i + m * self.NTS, l + (i - 1) * self.NX]      = g_dot1[m, l]
                            c_dsdp[i + m * self.NTS, l + (i - 1) * self.NX, :] = g_ddot[m, l, :]

                else:
                    for m in xrange(0, self.NG):
                        c[i + m * self.NTS]       = g[m]
                        c_dp[i + m * self.NTS, :] = g_dot2[m, :]

                        for l in xrange(0, self.NX):
                            c_ds[i + m * self.NTS, l + i * self.NX]      = g_dot1[m, l]
                            c_dsdp[i + m * self.NTS, l + i * self.NX, :] = g_ddot[m, l, :]

        return c, c_ds, c_dp, c_dsdp

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

        c      = None
        c_ds   = None
        c_dq   = None
        c_dsdq = None

        if self.NG > 0:

            # allocate memory
            c      = np.zeros((self.NC,))
            c_ds   = np.zeros((self.NC, self.NS))
            c_dq   = np.zeros((self.NC, self.NQ))
            c_dsdq = np.zeros((self.NC, self.NS, self.NQ))

            x      = np.zeros((self.NX,))
            x_dot1 = np.zeros((self.NX, self.NX))
            x_dot2 = np.zeros((self.NX, self.NU))
            x_ddot = np.zeros((self.NX, self.NX, self.NU))
            g      = np.zeros((self.NG,))
            g_dot1 = np.zeros((self.NG, self.NX))
            g_dot2 = np.zeros((self.NG, self.NU))
            g_ddot = np.zeros((self.NG, self.NX, self.NU))
            p_dot  = np.zeros((self.NP, self.NX))
            p_ddot = np.zeros((self.NP, self.NX, self.NU))
            u      = np.zeros((self.NU,))
            u_dot1 = np.zeros((self.NU, self.NX))
            u_dot2 = np.eye(self.NU)
            u_ddot = np.zeros((self.NU, self.NX, self.NU))

           # set state and controls of last time step
            x      = xs[self.NTS - 2, -1, :]
            x_dot1 = np.reshape(xs_dot1[self.NTS - 2, -1, :, (self.NTS - 2) * self.NX:(self.NTS - 1) * self.NX], x_dot1.shape)
            x_dot2 = np.reshape(xs_dot2[self.NTS - 2, -1, :, self.NTS - 2], x_dot2.shape)
            x_ddot = np.reshape(xs_ddot[self.NTS - 2, -1, :, (self.NTS - 2) * self.NX:(self.NTS - 1) * self.NX, self.NTS - 2], x_ddot.shape)

            for k in xrange(0, self.NU):
                u[k] = q[self.NTS - 1 + k * self.NTS]

            # call fortran backend to calculate derivatives of constraint functions
            self.backend_fortran.gfcn_ddot(g, g_dot2, g_dot1, g_ddot,
                                           self.ts[self.NTS - 1:self.NTS],
                                           x, x_dot2, x_dot1, x_ddot,
                                           p, p_dot, p_dot, p_ddot,
                                           u, u_dot2, u_dot1, u_ddot)

            # store derivatives
            for k in xrange(0, self.NG):
                c[self.NTS - 1 + k * self.NTS] = g[k]

                for l in xrange(0, self.NX):
                    c_ds[self.NTS - 1 + k * self.NTS, l + (self.NTS - 2) * self.NX] = g_dot1[k, l]

                    for j in xrange(0, self.NU):
                        c_dsdq[self.NTS - 1 + k * self.NTS, l + (self.NTS - 2) * self.NX, self.NTS - 2 + j * self.NTS] = g_ddot[k, l, j]

                for m in xrange(0, self.NU):
                    c_dq[self.NTS - 1 + k * self.NTS, self.NTS - 2 + m * self.NTS] = g_dot2[k, m]

        return c, c_ds, c_dq, c_dsdq

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

        if self.NG > 0:

            # allocate memory
            c      = np.zeros((self.NC,))
            c_dp   = np.zeros((self.NC, self.NP))
            c_dq   = np.zeros((self.NC, self.NQ))
            c_dpdq = np.zeros((self.NC, self.NP, self.NQ))

            x      = np.zeros((self.NX,))
            x_dot1 = np.zeros((self.NX, self.NP))
            x_dot2 = np.zeros((self.NX, self.NU))
            x_ddot = np.zeros((self.NX, self.NP, self.NU))
            g      = np.zeros((self.NG,))
            g_dot1 = np.zeros((self.NG, self.NP))
            g_dot2 = np.zeros((self.NG, self.NU))
            g_ddot = np.zeros((self.NG, self.NP, self.NU))
            p_dot1 = np.eye(self.NP)
            p_dot2 = np.zeros((self.NP, self.NU))
            p_ddot = np.zeros((self.NP, self.NP, self.NU))
            u      = np.zeros((self.NU,))
            u_dot1 = np.zeros((self.NU, self.NP))
            u_dot2 = np.eye(self.NU)
            u_ddot = np.zeros((self.NU, self.NP, self.NU))

           # set state and controls of last time step
            x      = xs[self.NTS - 2, -1, :]
            x_dot1 = np.reshape(xs_dot1[self.NTS - 2, -1, :, :], x_dot1.shape)
            x_dot2 = np.reshape(xs_dot2[self.NTS - 2, -1, :, self.NTS - 2], x_dot2.shape)
            x_ddot = np.reshape(xs_ddot[self.NTS - 2, -1, :, :, self.NTS - 2], x_ddot.shape)

            for k in xrange(0, self.NU):
                u[k] = q[self.NTS - 1 + k * self.NTS]

            # call fortran backend to calculate derivatives of constraint functions
            self.backend_fortran.gfcn_ddot(g, g_dot2, g_dot1, g_ddot,
                                           self.ts[self.NTS - 1:self.NTS],
                                           x, x_dot2, x_dot1, x_ddot,
                                           p, p_dot2, p_dot1, p_ddot,
                                           u, u_dot2, u_dot1, u_ddot)

            # store derivatives
            for k in xrange(0, self.NG):
                c[self.NTS - 1 + k * self.NTS]       = g[k]
                c_dp[self.NTS - 1 + k * self.NTS, :] = g_dot1[k, :]

                for m in xrange(0, self.NU):
                    c_dq[self.NTS - 1 + k * self.NTS, self.NTS - 2 + m * self.NTS]      = g_dot2[k, m]
                    c_dpdq[self.NTS - 1 + k * self.NTS, :, self.NTS - 2 + m * self.NTS] = g_ddot[k, :, m]

        return c, c_dp, c_dq, c_dpdq

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

    def obj_dsds(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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

    def obj_dsdp(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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

    def obj_dsdq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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

        return self.sign * xs[-1, -1, -1], self.sign * xs_dot1[-1, -1, -1, :], self.sign * xs_dot2[-1, -1, -1, :], self.sign * xs_ddot[-1, -1, -1, :, :]

"""
===============================================================================
"""
