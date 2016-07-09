#!/usr/bin/env python
# -*- coding: utf-8 -*-

# system imports
import numpy as np
import json
import matplotlib.pyplot as pl
import datetime as datetime

# project imports
from msobox.mf.tapenade import Differentiator
from msobox.mf.fortran import BackendFortran

from msobox.ind.rk4classic import RK4Classic
from msobox.ind.explicit_euler import ExplicitEuler

# =============================================================================

class Problem(object):

    """

    provides functionalities for ...

    """

    # =========================================================================

    def flat2array_q(self, q):

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
        q_array = np.zeros((self.NU, self.NTS, self.NQI))

        # convert controls from one-dimensional to 3-dimensional
        for i in xrange(0, self.NU):
            q_array[i, :, 0] = q[i * self.NTS:(i + 1) * self.NTS]

        return q_array

    # =========================================================================

    def flat2array_s(self, s):



        # set up array
        s_array = np.zeros((self.NTS, self.NX))

        # convert shooting variables from one-dimensional to 3-dimensional
        for i in xrange(0, self.NTS):
            s_array[i, :] = s[i * self.NX:(i + 1) * self.NX]

        return s_array

    # =========================================================================

    def approximate_s(self):



        # allocate memory
        self.s = np.zeros((self.NTS * self.NX,))

        # approximate shooting variables by linear interpolation if possible
        for i in xrange(0, self.NX):

            # set initial shooting variables to x0 if possible
            if self.x0[i] is not None:
                self.s[i] = self.x0[i]

            for j in xrange(1, self.NTS):

                # interpolate from x0 to xend if possible
                if self.xend[i] is not None:
                    self.s[j * self.NX + i] = self.x0[i] + float(j) / (self.NTS - 1) * (self.xend[i] - self.x0[i]) / (self.ts[-1] - self.ts[0])

    # =========================================================================

    def prepare(self):



        self.NCG = self.NG * self.NTS             # number of inequality constraints
        self.NCH = self.NH * self.NTS             # number of equality constraints
        self.NC  = self.NCG + self.NCH            # total number of constraints
        self.NQI = 1                              # number of controls per shooting interval
        self.NQ  = self.NU * self.NTS * self.NQI  # number of controls
        self.NS  = self.NTS * self.NX             # number of shooting variables
        self.NMC = self.NS - self.NX              # number of matching conditions

        # assert right dimensions of data
        assert self.ts.size      == self.NTS
        assert self.p.size       == self.NP
        assert self.q.size       == self.NQ
        assert self.s.size       == self.NS
        assert len(self.x0)      == self.NX
        assert len(self.xend)    == self.NX

        # set whether to minimize or maximize
        if self.minormax == "min":
            self.sign = 1

        elif self.minormax == "max":
            self.sign = -1

        else:
            print "No valid input for minormax."
            raise Exception

        # load json containing data structure for differentiator
        with open(self.path + "ds.json", "r") as f:
            ds = json.load(f)

        # differentiate model functions
        Differentiator(self.path, ds=ds)
        self.backend_fortran = BackendFortran(self.path + "gen/libproblem.so")

        # set integrator
        if self.integrator == "rk4classic":
            self.ind = RK4Classic(self.backend_fortran)

        elif self.integrator == "explicit_euler":
            self.ind = ExplicitEuler(self.backend_fortran)

        else:
            print "Chosen integrator is not available."
            raise NotImplementedError

    # =========================================================================

    def plot(self):



        p = self.p
        q = self.q
        s = self.s

        q_plot = self.flat2array_q(q)[:, :, 0]
        x      = self.integrate(p, q, s)
        x_plot = np.zeros((self.NTS + (self.NTSI - 2) * (self.NTS - 1), x.shape[2]))

        # copy data
        for i in xrange(0, x.shape[0]):
            x_plot[i * (self.NTSI - 1):(i + 1) * (self.NTSI - 1), :] = x[i, :-1, :]

        # set last time step
        x_plot[-1, :] = x[-1, -1, :]

        colors = ["blue", "red", "green", "yellow"]
        for i in xrange(0, self.NU):
            pl.plot(self.ts, q_plot[i], color=colors[i], linewidth=2, linestyle="dashed", label="u_" + str(i))

        for i in xrange(0, self.NX):
            pl.plot(np.linspace(0, 1, x_plot[:, i].size), x_plot[:, i], color=colors[i], linewidth=2, linestyle="solid", label="x_" + str(i))

        # set layout, save and show plot
        pl.xlabel("t")
        pl.ylabel("")
        pl.title("solution of ocp")
        pl.grid(True)
        pl.legend(loc="best")
        pl.savefig(self.path + "/output/" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "-plot.png", bbox_inches="tight")
        pl.savefig(self.path + "/output/" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "-plot.pdf", bbox_inches="tight")
        pl.show()

    # =========================================================================

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

        # convert controls and shooting variables
        q = self.flat2array_q(q)
        s = self.flat2array_s(s)

        # set initial conditions
        x0 = s[interval, :]

        # set time steps for this interval
        tsi = np.linspace(self.ts[interval], self.ts[interval + 1], self.NTSI)

        # set constant controls for this interval
        q_interval = q[:, interval, 0]

        # integrate
        self.ind.zo_forward(tsi,
                            x0,
                            p,
                            q_interval)

        return self.ind.xs

    # =========================================================================

    def integrate_interval_ds(self, interval, p, q, s):



        # convert controls and shooting variables
        q = self.flat2array_q(q)
        s = self.flat2array_s(s)

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
        self.ind.fo_forward(tsi,
                            x0, x0_dot,
                            p, p_dot,
                            q_interval, q_dot)

        xs_dot[:, :, interval * self.NX:(interval + 1) * self.NX] = self.ind.xs_dot

        return self.ind.xs, xs_dot

    # =========================================================================

    def integrate_interval_dp(self, interval, p, q, s):



        # convert controls and shooting variables
        q = self.flat2array_q(q)
        s = self.flat2array_s(s)

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
        self.ind.fo_forward(tsi,
                            x0, x0_dot,
                            p, p_dot,
                            q_interval, q_dot)

        return self.ind.xs, self.ind.xs_dot

    # =========================================================================

    def integrate_interval_dq(self, interval, p, q, s):



        # convert controls and shooting variables
        q = self.flat2array_q(q)
        s = self.flat2array_s(s)

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
        self.ind.fo_forward(tsi,
                            x0, x0_dot,
                            p, p_dot,
                            q_interval, q_dot)

        xs_dot[:, :, interval * self.NU:(interval + 1) * self.NU] = self.ind.xs_dot

        return self.ind.xs, xs_dot

    # =========================================================================

    def integrate_interval_dsds(self, interval, p, q, s):



        # convert controls and shooting variables
        q = self.flat2array_q(q)
        s = self.flat2array_s(s)

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
        self.ind.so_forward(tsi,
                            x0, x0_dot, x0_dot, x0_ddot,
                            p, p_dot, p_dot, p_ddot,
                            q_interval, q_dot, q_dot, q_ddot)

        xs_dot1[:, :, interval * self.NX:(interval + 1) * self.NX]                                              = self.ind.xs_dot1
        xs_dot2[:, :, interval * self.NX:(interval + 1) * self.NX]                                              = self.ind.xs_dot2
        xs_ddot[:, :, interval * self.NX:(interval + 1) * self.NX, interval * self.NX:(interval + 1) * self.NX] = self.ind.xs_ddot

        return self.ind.xs, xs_dot1, xs_dot2, xs_ddot

    # =========================================================================

    def integrate_interval_dpdp(self, interval, p, q, s):



        # convert controls and shooting variables
        q = self.flat2array_q(q)
        s = self.flat2array_s(s)

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
        self.ind.so_forward(tsi,
                            x0, x0_dot, x0_dot, x0_ddot,
                            p, p_dot, p_dot, p_ddot,
                            q_interval, q_dot, q_dot, q_ddot)

        return self.ind.xs, self.ind.xs_dot1, self.ind.xs_dot2, self.ind.xs_ddot

    # =========================================================================

    def integrate_interval_dqdq(self, interval, p, q, s):



        # convert controls and shooting variables
        q = self.flat2array_q(q)
        s = self.flat2array_s(s)

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
        self.ind.so_forward(tsi,
                            x0, x0_dot, x0_dot, x0_ddot,
                            p, p_dot, p_dot, p_ddot,
                            q_interval, q_dot, q_dot, q_ddot)

        xs_dot1[:, :, interval * self.NU:(interval + 1) * self.NU]                                              = self.ind.xs_dot1
        xs_dot2[:, :, interval * self.NU:(interval + 1) * self.NU]                                              = self.ind.xs_dot2
        xs_ddot[:, :, interval * self.NU:(interval + 1) * self.NU, interval * self.NU:(interval + 1) * self.NU] = self.ind.xs_ddot

        return self.ind.xs, xs_dot1, xs_dot2, xs_ddot

    # =========================================================================

    def integrate_interval_dsdp(self, interval, p, q, s):



        # convert controls and shooting variables
        q = self.flat2array_q(q)
        s = self.flat2array_s(s)

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
        self.ind.so_forward(tsi,
                            x0, x0_dot2, x0_dot1, x0_ddot,
                            p, p_dot2, p_dot1, p_ddot,
                            q_interval, q_dot2, q_dot1, q_ddot)

        xs_dot1[:, :, interval * self.NX:(interval + 1) * self.NX]    = self.ind.xs_dot1
        xs_ddot[:, :, interval * self.NX:(interval + 1) * self.NX, :] = self.ind.xs_ddot

        return self.ind.xs, xs_dot1, self.ind.xs_dot2, xs_ddot

    # =========================================================================

    def integrate_interval_dsdq(self, interval, p, q, s):



        # convert controls and shooting variables
        q = self.flat2array_q(q)
        s = self.flat2array_s(s)

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
        self.ind.so_forward(tsi,
                            x0, x0_dot2, x0_dot1, x0_ddot,
                            p, p_dot2, p_dot1, p_ddot,
                            q_interval, q_dot2, q_dot1, q_ddot)

        xs_dot1[:, :, interval * self.NX:(interval + 1) * self.NX]                                              = self.ind.xs_dot1
        xs_dot2[:, :, interval * self.NU:(interval + 1) * self.NU]                                              = self.ind.xs_dot2
        xs_ddot[:, :, interval * self.NX:(interval + 1) * self.NX, interval * self.NU:(interval + 1) * self.NU] = self.ind.xs_ddot

        return self.ind.xs, xs_dot1, xs_dot2, xs_ddot

    # =========================================================================

    def integrate_interval_dpdq(self, interval, p, q, s):



        # convert controls and shooting variables
        q = self.flat2array_q(q)
        s = self.flat2array_s(s)

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
        self.ind.so_forward(tsi,
                            x0, x0_dot2, x0_dot1, x0_ddot,
                            p, p_dot2, p_dot1, p_ddot,
                            q_interval, q_dot2, q_dot1, q_ddot)

        xs_dot2[:, :, interval * self.NU:(interval + 1) * self.NU]    = self.ind.xs_dot2
        xs_ddot[:, :, :, interval * self.NU:(interval + 1) * self.NU] = self.ind.xs_ddot

        return self.ind.xs, self.ind.xs_dot1, xs_dot2, xs_ddot

    # =========================================================================

    def integrate(self, p, q, s):



        # allocate memory
        xs = np.zeros((self.NTS - 1, self.NTSI, self.NX))

        # integrate on shooting intervals
        for i in xrange(0, self.NTS - 1):
            xs[i, :, :] = self.integrate_interval(i, p, q, s)

        return xs

    # =========================================================================

    def integrate_ds(self, p, q, s):



        # allocate memory
        xs     = np.zeros((self.NTS - 1, self.NTSI, self.NX))
        xs_dot = np.zeros((self.NTS - 1, self.NTSI, self.NX, self.NS))

        # integrate on shooting intervals
        for i in xrange(0, self.NTS - 1):
            xs[i, :, :], xs_dot[i, :, :, :] = self.integrate_interval_ds(i, p, q, s)

        return xs, xs_dot

    # =========================================================================

    def integrate_dp(self, p, q, s):



        # allocate memory
        xs     = np.zeros((self.NTS - 1, self.NTSI, self.NX))
        xs_dot = np.zeros((self.NTS - 1, self.NTSI, self.NX, self.NP))

        # integrate on shooting intervals
        for i in xrange(0, self.NTS - 1):
            xs[i, :, :], xs_dot[i, :, :, :] = self.integrate_interval_dp(i, p, q, s)

        return xs, xs_dot

    # =========================================================================

    def integrate_dq(self, p, q, s):



        # allocate memory
        xs     = np.zeros((self.NTS - 1, self.NTSI, self.NX))
        xs_dot = np.zeros((self.NTS - 1, self.NTSI, self.NX, self.NQ))

        # integrate on shooting intervals
        for i in xrange(0, self.NTS - 1):
            xs[i, :, :], xs_dot[i, :, :, :] = self.integrate_interval_dq(i, p, q, s)

        return xs, xs_dot

    # =========================================================================

    def integrate_dsds(self, p, q, s):



        # allocate memory
        xs      = np.zeros((self.NTS - 1, self.NTSI, self.NX))
        xs_dot1 = np.zeros((self.NTS - 1, self.NTSI, self.NX, self.NS))
        xs_dot2 = np.zeros((self.NTS - 1, self.NTSI, self.NX, self.NS))
        xs_ddot = np.zeros((self.NTS - 1, self.NTSI, self.NX, self.NS, self.NS))

        # integrate on shooting intervals
        for i in xrange(0, self.NTS - 1):
            xs[i, :, :], xs_dot1[i, :, :, :], xs_dot2[i, :, :, :], xs_ddot[i, :, :, :, :] = self.integrate_interval_dsds(i, p, q, s)

        return xs, xs_dot1, xs_dot2, xs_ddot

    # =========================================================================

    def integrate_dpdp(self, p, q, s):



        # allocate memory
        xs      = np.zeros((self.NTS - 1, self.NTSI, self.NX))
        xs_dot1 = np.zeros((self.NTS - 1, self.NTSI, self.NX, self.NP))
        xs_dot2 = np.zeros((self.NTS - 1, self.NTSI, self.NX, self.NP))
        xs_ddot = np.zeros((self.NTS - 1, self.NTSI, self.NX, self.NP, self.NP))

        # integrate on shooting intervals
        for i in xrange(0, self.NTS - 1):
            xs[i, :, :], xs_dot1[i, :, :, :], xs_dot2[i, :, :, :], xs_ddot[i, :, :, :, :] = self.integrate_interval_dpdp(i, p, q, s)

        return xs, xs_dot1, xs_dot2, xs_ddot

    # =========================================================================

    def integrate_dqdq(self, p, q, s):



        # allocate memory
        xs      = np.zeros((self.NTS - 1, self.NTSI, self.NX))
        xs_dot1 = np.zeros((self.NTS - 1, self.NTSI, self.NX, self.NQ))
        xs_dot2 = np.zeros((self.NTS - 1, self.NTSI, self.NX, self.NQ))
        xs_ddot = np.zeros((self.NTS - 1, self.NTSI, self.NX, self.NQ, self.NQ))

        # integrate on shooting intervals
        for i in xrange(0, self.NTS - 1):
            xs[i, :, :], xs_dot1[i, :, :, :], xs_dot2[i, :, :, :], xs_ddot[i, :, :, :, :] = self.integrate_interval_dqdq(i, p, q, s)

        return xs, xs_dot1, xs_dot2, xs_ddot

    # =========================================================================

    def integrate_dsdp(self, p, q, s):



        # allocate memory
        xs      = np.zeros((self.NTS - 1, self.NTSI, self.NX))
        xs_dot1 = np.zeros((self.NTS - 1, self.NTSI, self.NX, self.NS))
        xs_dot2 = np.zeros((self.NTS - 1, self.NTSI, self.NX, self.NP))
        xs_ddot = np.zeros((self.NTS - 1, self.NTSI, self.NX, self.NS, self.NP))

        # integrate on shooting intervals
        for i in xrange(0, self.NTS - 1):
            xs[i, :, :], xs_dot1[i, :, :, :], xs_dot2[i, :, :, :], xs_ddot[i, :, :, :, :] = self.integrate_interval_dsdp(i, p, q, s)

        return xs, xs_dot1, xs_dot2, xs_ddot

    # =========================================================================

    def integrate_dsdq(self, p, q, s):



        # allocate memory
        xs      = np.zeros((self.NTS - 1, self.NTSI, self.NX))
        xs_dot1 = np.zeros((self.NTS - 1, self.NTSI, self.NX, self.NS))
        xs_dot2 = np.zeros((self.NTS - 1, self.NTSI, self.NX, self.NQ))
        xs_ddot = np.zeros((self.NTS - 1, self.NTSI, self.NX, self.NS, self.NQ))

        # integrate on shooting intervals
        for i in xrange(0, self.NTS - 1):
            xs[i, :, :], xs_dot1[i, :, :, :], xs_dot2[i, :, :, :], xs_ddot[i, :, :, :, :] = self.integrate_interval_dsdq(i, p, q, s)

        return xs, xs_dot1, xs_dot2, xs_ddot

    # =========================================================================

    def integrate_dpdq(self, p, q, s):



        # allocate memory
        xs      = np.zeros((self.NTS - 1, self.NTSI, self.NX))
        xs_dot1 = np.zeros((self.NTS - 1, self.NTSI, self.NX, self.NP))
        xs_dot2 = np.zeros((self.NTS - 1, self.NTSI, self.NX, self.NQ))
        xs_ddot = np.zeros((self.NTS - 1, self.NTSI, self.NX, self.NP, self.NQ))

        # integrate on shooting intervals
        for i in xrange(0, self.NTS - 1):
            xs[i, :, :], xs_dot1[i, :, :, :], xs_dot2[i, :, :, :], xs_ddot[i, :, :, :, :] = self.integrate_interval_dpdq(i, p, q, s)

        return xs, xs_dot1, xs_dot2, xs_ddot

    # =========================================================================

    def ineqc(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):



        c = None

        if self.NG > 0:

            # allocate memory
            c = np.zeros((self.NCG,))
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

    # =========================================================================

    def ineqc_ds(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):



        c    = None
        c_ds = None

        if self.NG > 0:

            # allocate memory
            c     = np.zeros((self.NCG,))
            c_ds  = np.zeros((self.NCG, self.NS))

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

    # =========================================================================

    def ineqc_dp(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):



        c    = None
        c_dp = None

        if self.NG > 0:

            # allocate memory
            c     = np.zeros((self.NCG,))
            c_dp  = np.zeros((self.NCG, self.NP))

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

    # =========================================================================

    def ineqc_dq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):



        c    = None
        c_dq = None

        if self.NG > 0:

            # allocate memory
            c     = np.zeros((self.NCG,))
            c_dq  = np.zeros((self.NCG, self.NQ))

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

    # =========================================================================

    def ineqc_dsds(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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
            c      = np.zeros((self.NCG,))
            c_ds1  = np.zeros((self.NCG, self.NS))
            c_ds2  = np.zeros((self.NCG, self.NS))
            c_dsds = np.zeros((self.NCG, self.NS, self.NS))

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

    # =========================================================================

    def ineqc_dpdp(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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
            c      = np.zeros((self.NCG,))
            c_dp1  = np.zeros((self.NCG, self.NP))
            c_dp2  = np.zeros((self.NCG, self.NP))
            c_dpdp = np.zeros((self.NCG, self.NP, self.NP))

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

    # =========================================================================

    def ineqc_dqdq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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
            c      = np.zeros((self.NCG,))
            c_dq1  = np.zeros((self.NCG, self.NQ))
            c_dq2  = np.zeros((self.NCG, self.NQ))
            c_dqdq = np.zeros((self.NCG, self.NQ, self.NQ))

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

    # =========================================================================

    def ineqc_dsdp(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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
            c      = np.zeros((self.NCG,))
            c_ds   = np.zeros((self.NCG, self.NS))
            c_dp   = np.zeros((self.NCG, self.NP))
            c_dsdp = np.zeros((self.NCG, self.NS, self.NP))

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

    # =========================================================================

    def ineqc_dsdq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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
            c      = np.zeros((self.NCG,))
            c_ds   = np.zeros((self.NCG, self.NS))
            c_dq   = np.zeros((self.NCG, self.NQ))
            c_dsdq = np.zeros((self.NCG, self.NS, self.NQ))

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

    # =========================================================================

    def ineqc_dpdq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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
            c      = np.zeros((self.NCG,))
            c_dp   = np.zeros((self.NCG, self.NP))
            c_dq   = np.zeros((self.NCG, self.NQ))
            c_dpdq = np.zeros((self.NCG, self.NP, self.NQ))

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
            for k in xrange(0, self.N):
                c[self.NTS - 1 + k * self.NTS]       = g[k]
                c_dp[self.NTS - 1 + k * self.NTS, :] = g_dot1[k, :]

                for m in xrange(0, self.NU):
                    c_dq[self.NTS - 1 + k * self.NTS, self.NTS - 2 + m * self.NTS]      = g_dot2[k, m]
                    c_dpdq[self.NTS - 1 + k * self.NTS, :, self.NTS - 2 + m * self.NTS] = g_ddot[k, :, m]

        return c, c_dp, c_dq, c_dpdq

    # =========================================================================

    def eqc(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):



        c = None

        if self.NH > 0:

            # allocate memory
            c = np.zeros((self.NCH,))
            x = np.zeros((self.NX,))
            h = np.zeros((self.NH,))
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
                self.backend_fortran.hfcn(h, self.ts[i:i + 1], x, p, u)

                # build constraints
                for k in xrange(0, self.NH):
                    c[i + k * self.NTS] = h[k]

        return c

    # =========================================================================

    def eqc_ds(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):



        c    = None
        c_ds = None

        if self.NH > 0:

            # allocate memory
            c     = np.zeros((self.NCH,))
            c_ds  = np.zeros((self.NCH, self.NS))

            x     = np.zeros((self.NX,))
            x_dot = np.zeros((self.NX, self.NX))
            h     = np.zeros((self.NH,))
            h_dot = np.zeros((self.NH, self.NX))
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
                self.backend_fortran.hfcn_dot(h, h_dot,
                                              self.ts[i:i + 1],
                                              x, x_dot,
                                              p, p_dot,
                                              u, u_dot)

                # store derivatives
                if i == self.NTS - 1:
                    for m in xrange(0, self.NH):
                        c[i + m * self.NTS] = h[m]

                        for l in xrange(0, self.NX):
                            c_ds[i + m * self.NTS, l + (i - 1) * self.NX] = h_dot[m, l]

                else:
                    for m in xrange(0, self.NH):
                        c[i + m * self.NTS] = h[m]

                        for l in xrange(0, self.NX):
                            c_ds[i + m * self.NTS, l + i * self.NX] = h_dot[m, l]

        return c, c_ds

    # =========================================================================

    def eqc_dp(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):



        c    = None
        c_dp = None

        if self.NH > 0:

            # allocate memory
            c     = np.zeros((self.NCH,))
            c_dp  = np.zeros((self.NCH, self.NP))

            x     = np.zeros((self.NX,))
            x_dot = np.zeros((self.NX, self.NP))
            h     = np.zeros((self.NH,))
            h_dot = np.zeros((self.NH, self.NP))
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
                self.backend_fortran.hfcn_dot(h, h_dot,
                                              self.ts[i:i + 1],
                                              x, x_dot,
                                              p, p_dot,
                                              u, u_dot)

                # store gradient
                for k in xrange(0, self.NH):
                    c[i + k * self.NTS]       = h[k]
                    c_dp[i + k * self.NTS, :] = h_dot[k, :]


        return c, c_dp

    # =========================================================================

    def eqc_dq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):



        c    = None
        c_dq = None

        if self.NH > 0:

            # allocate memory
            c     = np.zeros((self.NCH,))
            c_dq  = np.zeros((self.NCH, self.NQ))

            x     = np.zeros((self.NX,))
            x_dot = np.zeros((self.NX, self.NU))
            h     = np.zeros((self.NH,))
            h_dot = np.zeros((self.NH, self.NU))
            p_dot = np.zeros((self.NP, self.NU))
            u     = np.zeros((self.NU,))
            u_dot = np.zeros((self.NU, self.NU))

            # set gradient for constraints of last time step
            x     = xs[self.NTS - 2, -1, :]
            x_dot = np.reshape(xs_dot1[self.NTS - 2, -1, :, self.NTS - 2], x_dot.shape)

            for k in xrange(0, self.NU):
                u[k] = q[self.NTS - 1 + k * self.NTS]

            # call fortran backend to calculate derivatives of constraint functions
            self.backend_fortran.hfcn_dot(h, h_dot,
                                          self.ts[self.NTS - 1:self.NTS],
                                          x, x_dot,
                                          p, p_dot,
                                          u, u_dot)

            # store derivatives
            for k in xrange(0, self.NH):
                c[self.NTS - 1 + k * self.NTS] = h[k]

                for l in xrange(0, self.NU):
                    c_dq[self.NTS - 1 + k * self.NTS, self.NTS - 2 + l * self.NTS] = h_dot[k, l]

        return c, c_dq

    # =========================================================================

    def eqc_dsds(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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

        if self.NH > 0:

            # allocate memory
            c      = np.zeros((self.NCH,))
            c_ds1  = np.zeros((self.NCH, self.NS))
            c_ds2  = np.zeros((self.NCH, self.NS))
            c_dsds = np.zeros((self.NCH, self.NS, self.NS))

            x      = np.zeros((self.NX,))
            x_dot1 = np.zeros((self.NX, self.NX))
            x_dot2 = np.zeros((self.NX, self.NX))
            x_ddot = np.zeros((self.NX, self.NX, self.NX))
            h      = np.zeros((self.NH,))
            h_dot1 = np.zeros((self.NH, self.NX))
            h_dot2 = np.zeros((self.NH, self.NX))
            h_ddot = np.zeros((self.NH, self.NX, self.NX))
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
                self.backend_fortran.hfcn_ddot(h, h_dot2, h_dot1, h_ddot,
                                               self.ts[i:i + 1],
                                               x, x_dot2, x_dot1, x_ddot,
                                               p, p_dot, p_dot, p_ddot,
                                               u, u_dot, u_dot, u_ddot)

                # store derivatives
                if i == self.NTS - 1:
                    for m in xrange(0, self.NH):
                        c[i + m * self.NTS] = h[m]

                        for l in xrange(0, self.NX):
                            c_ds1[i + m * self.NTS, l + (i - 1) * self.NX] = h_dot1[m, l]
                            c_ds2[i + m * self.NTS, l + (i - 1) * self.NX] = h_dot2[m, l]

                            for j in xrange(0, self.NX):
                                c_dsds[i + m * self.NTS, l + (i - 1) * self.NX, j + (i - 1) * self.NX] = h_ddot[m, l, j]

                else:
                    for m in xrange(0, self.NH):
                        c[i + m * self.NTS] = h[m]

                        for l in xrange(0, self.NX):
                            c_ds1[i + m * self.NTS, l + i * self.NX] = h_dot1[m, l]
                            c_ds2[i + m * self.NTS, l + i * self.NX] = h_dot2[m, l]

                            for j in xrange(0, self.NX):
                                c_dsds[i + m * self.NTS, l + i * self.NX, j + i * self.NX] = h_ddot[m, l, j]

        return c, c_ds1, c_ds2, c_dsds

    # =========================================================================

    def eqc_dpdp(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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

        if self.NH > 0:

            # allocate memory
            c      = np.zeros((self.NCH,))
            c_dp1  = np.zeros((self.NCH, self.NP))
            c_dp2  = np.zeros((self.NCH, self.NP))
            c_dpdp = np.zeros((self.NCH, self.NP, self.NP))

            x      = np.zeros((self.NX,))
            x_dot1 = np.zeros((self.NX, self.NP))
            x_dot2 = np.zeros((self.NX, self.NP))
            x_ddot = np.zeros((self.NX, self.NP, self.NP))
            h      = np.zeros((self.NH,))
            h_dot1 = np.zeros((self.NH, self.NP))
            h_dot2 = np.zeros((self.NH, self.NP))
            h_ddot = np.zeros((self.NH, self.NP, self.NP))
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
                self.backend_fortran.hfcn_ddot(h, h_dot2, h_dot1, h_ddot,
                                               self.ts[i:i + 1],
                                               x, x_dot2, x_dot1, x_ddot,
                                               p, p_dot, p_dot, p_ddot,
                                               u, u_dot, u_dot, u_ddot)

                # store gradient
                for k in xrange(0, self.NH):
                    c[i + k * self.NTS]            = h[k]
                    c_dp1[i + k * self.NTS, :]     = h_dot1[k, :]
                    c_dp2[i + k * self.NTS, :]     = h_dot2[k, :]
                    c_dpdp[i + k * self.NTS, :, :] = h_ddot[k, :, :]

        return c, c_dp1, c_dp2, c_dpdp

    # =========================================================================

    def eqc_dqdq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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

        if self.NH > 0:

            # allocate memory
            c      = np.zeros((self.NCH,))
            c_dq1  = np.zeros((self.NCH, self.NQ))
            c_dq2  = np.zeros((self.NCH, self.NQ))
            c_dqdq = np.zeros((self.NCH, self.NQ, self.NQ))

            x      = np.zeros((self.NX,))
            x_dot1 = np.zeros((self.NX, self.NU))
            x_dot2 = np.zeros((self.NX, self.NU))
            x_ddot = np.zeros((self.NX, self.NU, self.NU))
            h      = np.zeros((self.NH,))
            h_dot1 = np.zeros((self.NH, self.NU))
            h_dot2 = np.zeros((self.NH, self.NU))
            h_ddot = np.zeros((self.NH, self.NU, self.NU))
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
            self.backend_fortran.hfcn_ddot(h, h_dot2, h_dot1, h_ddot,
                                           self.ts[self.NTS - 1:self.NTS],
                                           x, x_dot2, x_dot1, x_ddot,
                                           p, p_dot, p_dot, p_ddot,
                                           u, u_dot, u_dot, u_ddot)

            # store derivatives
            for k in xrange(0, self.NH):
                c[self.NTS - 1 + k * self.NTS] = h[k]

                for l in xrange(0, self.NU):
                    c_dq1[self.NTS - 1 + k * self.NTS, self.NTS - 2 + l * self.NTS] = h_dot1[k, l]
                    c_dq2[self.NTS - 1 + k * self.NTS, self.NTS - 2 + l * self.NTS] = h_dot2[k, l]

                    for m in xrange(0, self.NU):
                        c_dqdq[self.NTS - 1 + k * self.NTS, self.NTS - 2 + l * self.NTS, self.NTS - 2 + l * self.NTS] = h_ddot[k, l, m]

        return c, c_dq1, c_dq2, c_dqdq

    # =========================================================================

    def eqc_dsdp(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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

        if self.NH > 0:

            # allocate memory
            c      = np.zeros((self.NCH,))
            c_ds   = np.zeros((self.NCH, self.NS))
            c_dp   = np.zeros((self.NCH, self.NP))
            c_dsdp = np.zeros((self.NCH, self.NS, self.NP))

            x      = np.zeros((self.NX,))
            x_dot1 = np.zeros((self.NX, self.NX))
            x_dot2 = np.zeros((self.NX, self.NP))
            x_ddot = np.zeros((self.NX, self.NX, self.NP))
            h      = np.zeros((self.NH,))
            h_dot1 = np.zeros((self.NH, self.NX))
            h_dot2 = np.zeros((self.NH, self.NP))
            h_ddot = np.zeros((self.NH, self.NX, self.NP))
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
                self.backend_fortran.hfcn_ddot(h, h_dot2, h_dot1, h_ddot,
                                               self.ts[i:i + 1],
                                               x, x_dot2, x_dot1, x_ddot,
                                               p, p_dot2, p_dot1, p_ddot,
                                               u, u_dot, u_dot, u_ddot)

                # store derivatives
                if i == self.NTS - 1:
                    for m in xrange(0, self.NH):
                        c[i + m * self.NTS]       = h[m]
                        c_dp[i + m * self.NTS, :] = h_dot2[m, :]

                        for l in xrange(0, self.NX):
                            c_ds[i + m * self.NTS, l + (i - 1) * self.NX]      = h_dot1[m, l]
                            c_dsdp[i + m * self.NTS, l + (i - 1) * self.NX, :] = h_ddot[m, l, :]

                else:
                    for m in xrange(0, self.NH):
                        c[i + m * self.NTS]       = h[m]
                        c_dp[i + m * self.NTS, :] = h_dot2[m, :]

                        for l in xrange(0, self.NX):
                            c_ds[i + m * self.NTS, l + i * self.NX]      = h_dot1[m, l]
                            c_dsdp[i + m * self.NTS, l + i * self.NX, :] = h_ddot[m, l, :]

        return c, c_ds, c_dp, c_dsdp

    # =========================================================================

    def eqc_dsdq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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

        if self.NH > 0:

            # allocate memory
            c      = np.zeros((self.NCH,))
            c_ds   = np.zeros((self.NCH, self.NS))
            c_dq   = np.zeros((self.NCH, self.NQ))
            c_dsdq = np.zeros((self.NCH, self.NS, self.NQ))

            x      = np.zeros((self.NX,))
            x_dot1 = np.zeros((self.NX, self.NX))
            x_dot2 = np.zeros((self.NX, self.NU))
            x_ddot = np.zeros((self.NX, self.NX, self.NU))
            h      = np.zeros((self.NH,))
            h_dot1 = np.zeros((self.NH, self.NX))
            h_dot2 = np.zeros((self.NH, self.NU))
            h_ddot = np.zeros((self.NH, self.NX, self.NU))
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
            self.backend_fortran.hfcn_ddot(h, h_dot2, h_dot1, h_ddot,
                                           self.ts[self.NTS - 1:self.NTS],
                                           x, x_dot2, x_dot1, x_ddot,
                                           p, p_dot, p_dot, p_ddot,
                                           u, u_dot2, u_dot1, u_ddot)

            # store derivatives
            for k in xrange(0, self.NH):
                c[self.NTS - 1 + k * self.NTS] = h[k]

                for l in xrange(0, self.NX):
                    c_ds[self.NTS - 1 + k * self.NTS, l + (self.NTS - 2) * self.NX] = h_dot1[k, l]

                    for j in xrange(0, self.NU):
                        c_dsdq[self.NTS - 1 + k * self.NTS, l + (self.NTS - 2) * self.NX, self.NTS - 2 + j * self.NTS] = h_ddot[k, l, j]

                for m in xrange(0, self.NU):
                    c_dq[self.NTS - 1 + k * self.NTS, self.NTS - 2 + m * self.NTS] = h_dot2[k, m]

        return c, c_ds, c_dq, c_dsdq

    # =========================================================================

    def eqc_dpdq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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

        if self.NH > 0:

            # allocate memory
            c      = np.zeros((self.NCH,))
            c_dp   = np.zeros((self.NCH, self.NP))
            c_dq   = np.zeros((self.NCH, self.NQ))
            c_dpdq = np.zeros((self.NCH, self.NP, self.NQ))

            x      = np.zeros((self.NX,))
            x_dot1 = np.zeros((self.NX, self.NP))
            x_dot2 = np.zeros((self.NX, self.NU))
            x_ddot = np.zeros((self.NX, self.NP, self.NU))
            h      = np.zeros((self.NH,))
            h_dot1 = np.zeros((self.NH, self.NP))
            h_dot2 = np.zeros((self.NH, self.NU))
            h_ddot = np.zeros((self.NH, self.NP, self.NU))
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
            self.backend_fortran.hfcn_ddot(h, h_dot2, h_dot1, h_ddot,
                                           self.ts[self.NTS - 1:self.NTS],
                                           x, x_dot2, x_dot1, x_ddot,
                                           p, p_dot2, p_dot1, p_ddot,
                                           u, u_dot2, u_dot1, u_ddot)

            # store derivatives
            for k in xrange(0, self.NH):
                c[self.NTS - 1 + k * self.NTS]       = h[k]
                c_dp[self.NTS - 1 + k * self.NTS, :] = h_dot1[k, :]

                for m in xrange(0, self.NU):
                    c_dq[self.NTS - 1 + k * self.NTS, self.NTS - 2 + m * self.NTS]      = h_dot2[k, m]
                    c_dpdq[self.NTS - 1 + k * self.NTS, :, self.NTS - 2 + m * self.NTS] = h_ddot[k, :, m]

        return c, c_dp, c_dq, c_dpdq

    # =========================================================================

    def mc(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):



        # allocate memory
        mc = np.zeros((self.NMC,))

        # evaluate matching conditions
        for i in xrange(0, self.NTS - 1):
            mc[i * self.NX:(i + 1) * self.NX] = xs[i, -1, :] - self.flat2array_s(s)[i + 1, :]

        return mc

    # =========================================================================

    def mc_ds(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):



        # allocate memory
        mc    = np.zeros((self.NMC,))
        mc_ds = np.zeros((self.NMC, self.NS))

        # evaluate matching conditions
        for i in xrange(0, self.NTS - 1):
            mc[i * self.NX:(i + 1) * self.NX]                                         = xs[i, -1, :] - self.flat2array_s(s)[i + 1, :]
            mc_ds[i * self.NX:(i + 1) * self.NX, :]                                   = xs_dot1[i, -1, :, :]
            mc_ds[i * self.NX:(i + 1) * self.NX, (i + 1) * self.NX:(i + 2) * self.NX] = -np.eye(self.NX)

        return mc, mc_ds

    # =========================================================================

    def mc_dp(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):



        # allocate memory
        mc    = np.zeros((self.NMC,))
        mc_dp = np.zeros((self.NMC, self.NP))

        # evaluate matching conditions
        for i in xrange(0, self.NTS - 1):
            mc[i * self.NX:(i + 1) * self.NX]       = xs[i, -1, :] - self.flat2array_s(s)[i + 1, :]
            mc_dp[i * self.NX:(i + 1) * self.NX, :] = xs_dot1[i, -1, :, :]

        return mc, mc_dp

    # =========================================================================

    def mc_dq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):



        # allocate memory
        mc    = np.zeros((self.NMC,))
        mc_dq = np.zeros((self.NMC, self.NQ))

        # evaluate matching conditions
        for i in xrange(0, self.NTS - 1):
            mc[i * self.NX:(i + 1) * self.NX]       = xs[i, -1, :] - self.flat2array_s(s)[i + 1, :]
            mc_dq[i * self.NX:(i + 1) * self.NX, :] = xs_dot1[i, -1, :, :]

        return mc, mc_dq

    # =========================================================================

    def mc_dsds(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):



        # allocate memory
        mc      = np.zeros((self.NMC,))
        mc_ds1  = np.zeros((self.NMC, self.NS))
        mc_ds2  = np.zeros((self.NMC, self.NS))
        mc_dsds = np.zeros((self.NMC, self.NS, self.NS))

        # evaluate matching conditions
        for i in xrange(0, self.NTS - 1):
            mc[i * self.NX:(i + 1) * self.NX]                                          = xs[i, -1, :] - self.flat2array_s(s)[i + 1, :]
            mc_ds1[i * self.NX:(i + 1) * self.NX, :]                                   = xs_dot1[i, -1, :, :]
            mc_ds1[i * self.NX:(i + 1) * self.NX, (i + 1) * self.NX:(i + 2) * self.NX] = -np.eye(self.NX)
            mc_ds2[i * self.NX:(i + 1) * self.NX, :]                                   = xs_dot2[i, -1, :, :]
            mc_ds2[i * self.NX:(i + 1) * self.NX, (i + 1) * self.NX:(i + 2) * self.NX] = -np.eye(self.NX)
            mc_dsds[i * self.NX:(i + 1) * self.NX, :, :]                               = xs_ddot[i, -1, :, :, :]

        return mc, mc_ds1, mc_ds2, mc_dsds

    # =========================================================================

    def mc_dpdp(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):



        # allocate memory
        mc      = np.zeros((self.NMC,))
        mc_dp1  = np.zeros((self.NMC, self.NP))
        mc_dp2  = np.zeros((self.NMC, self.NP))
        mc_dpdp = np.zeros((self.NMC, self.NP, self.NP))

        # evaluate matching conditions
        for i in xrange(0, self.NTS - 1):
            mc[i * self.NX:(i + 1) * self.NX]            = xs[i, -1, :] - self.flat2array_s(s)[i + 1, :]
            mc_dp1[i * self.NX:(i + 1) * self.NX, :]     = xs_dot1[i, -1, :, :]
            mc_dp2[i * self.NX:(i + 1) * self.NX, :]     = xs_dot2[i, -1, :, :]
            mc_dpdp[i * self.NX:(i + 1) * self.NX, :, :] = xs_ddot[i, -1, :, :, :]

        return mc, mc_dp1, mc_dp2, mc_dpdp


    # =========================================================================

    def mc_dqdq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):



        # allocate memory
        mc      = np.zeros((self.NMC,))
        mc_dq1  = np.zeros((self.NMC, self.NQ))
        mc_dq2  = np.zeros((self.NMC, self.NQ))
        mc_dqdq = np.zeros((self.NMC, self.NQ, self.NQ))

        # evaluate matching conditions
        for i in xrange(0, self.NTS - 1):
            mc[i * self.NX:(i + 1) * self.NX]            = xs[i, -1, :] - self.flat2array_s(s)[i + 1, :]
            mc_dq1[i * self.NX:(i + 1) * self.NX, :]     = xs_dot1[i, -1, :, :]
            mc_dq2[i * self.NX:(i + 1) * self.NX, :]     = xs_dot2[i, -1, :, :]
            mc_dqdq[i * self.NX:(i + 1) * self.NX, :, :] = xs_ddot[i, -1, :, :, :]

        return mc, mc_dq1, mc_dq2, mc_dqdq

    # =========================================================================

    def mc_dsdp(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):



        # allocate memory
        mc      = np.zeros((self.NMC,))
        mc_ds   = np.zeros((self.NMC, self.NS))
        mc_dp   = np.zeros((self.NMC, self.NP))
        mc_dsdp = np.zeros((self.NMC, self.NS, self.NP))

        # evaluate matching conditions
        for i in xrange(0, self.NTS - 1):
            mc[i * self.NX:(i + 1) * self.NX]                                         = xs[i, -1, :] - self.flat2array_s(s)[i + 1, :]
            mc_ds[i * self.NX:(i + 1) * self.NX, :]                                   = xs_dot1[i, -1, :, :]
            mc_ds[i * self.NX:(i + 1) * self.NX, (i + 1) * self.NX:(i + 2) * self.NX] = -np.eye(self.NX)
            mc_dp[i * self.NX:(i + 1) * self.NX, :]                                   = xs_dot2[i, -1, :, :]
            mc_dsdp[i * self.NX:(i + 1) * self.NX, :, :]                              = xs_ddot[i, -1, :, :, :]

        return mc, mc_ds, mc_dp, mc_dsdp

    # =========================================================================

    def mc_dsdq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):



        # allocate memory
        mc      = np.zeros((self.NMC,))
        mc_ds   = np.zeros((self.NMC, self.NS))
        mc_dq   = np.zeros((self.NMC, self.NQ))
        mc_dsdq = np.zeros((self.NMC, self.NS, self.NQ))

        # evaluate matching conditions
        for i in xrange(0, self.NTS - 1):
            mc[i * self.NX:(i + 1) * self.NX]                                         = xs[i, -1, :] - self.flat2array_s(s)[i + 1, :]
            mc_ds[i * self.NX:(i + 1) * self.NX, :]                                   = xs_dot1[i, -1, :, :]
            mc_ds[i * self.NX:(i + 1) * self.NX, (i + 1) * self.NX:(i + 2) * self.NX] = -np.eye(self.NX)
            mc_dq[i * self.NX:(i + 1) * self.NX, :]                                   = xs_dot2[i, -1, :, :]
            mc_dsdq[i * self.NX:(i + 1) * self.NX, :, :]                              = xs_ddot[i, -1, :, :, :]

        return mc, mc_ds, mc_dq, mc_dsdq

    # =========================================================================

    def mc_dpdq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):



        # allocate memory
        mc      = np.zeros((self.NMC,))
        mc_dp   = np.zeros((self.NMC, self.NP))
        mc_dq   = np.zeros((self.NMC, self.NQ))
        mc_dpdq = np.zeros((self.NMC, self.NP, self.NQ))

        # evaluate matching conditions
        for i in xrange(0, self.NTS - 1):
            mc[i * self.NX:(i + 1) * self.NX]            = xs[i, -1, :] - self.flat2array_s(s)[i + 1, :]
            mc_dp[i * self.NX:(i + 1) * self.NX, :]      = xs_dot1[i, -1, :, :]
            mc_dq[i * self.NX:(i + 1) * self.NX, :]      = xs_dot2[i, -1, :, :]
            mc_dpdq[i * self.NX:(i + 1) * self.NX, :, :] = xs_ddot[i, -1, :, :, :]

        return mc, mc_dp, mc_dq, mc_dpdq

    # =========================================================================

    def bc(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):



        # allocate memory
        bc = np.zeros((self.NQ * 2 + self.NS * 2,))

        # set the lower bnds for the controls q and the shooting variables s
        for i in xrange(0, self.NU):
            bc[i * self.NTS:(i + 1) * self.NTS] = self.bnds[i, 0] - q

        for i in xrange(0, self.NS):
            bc[self.NQ + i] = -1e6 - s[i]

        # set the upper bnds for the controls q and the shooting variables s
        l = self.NQ + self.NS
        for i in xrange(0, self.NU):
            bc[l + i * self.NTS:l + (i + 1) * self.NTS] = q - self.bnds[i, 1]

        for i in xrange(0, self.NS):
            bc[2 * self.NQ + self.NS + i] = -s[i] - 1e6

        # fix the shooting variables s at the boundaries if necessary
        l = self.NQ
        for i in xrange(0, self.NX):
            if self.x0[i] is not None:
                bc[l]     = self.x0[i] - s[i]
                bc[l + 1] = s[i] - self.x0[i]
                l         = l + 2

        l = self.NQ * 2 + self.NS
        for i in xrange(0, self.NX):
            if self.xend[i] is not None:
                bc[l]     = self.xend[i] - s[i]
                bc[l + 1] = s[i] - self.xend[i]
                l         = l + 2

        return bc

    # =========================================================================

    def bc_ds(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):



        # allocate memory
        bc_ds = np.zeros((self.NQ * 2 + self.NS * 2, self.NS))

        # set derivatives
        bc_ds[self.NQ:self.NQ + self.NS, :] = -np.eye(self.NS)
        bc_ds[self.NQ * 2 + self.NS:, :]    = np.eye(self.NS)

        return bc_ds

    # =========================================================================

    def bc_dp(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):



        bc_dp = np.zeros((self.NQ * 2 + self.NS * 2, self.NP))

        return bc_dp

    # =========================================================================

    def bc_dq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):



        # allocate memory
        bc_dq = np.zeros((self.NQ * 2 + self.NS * 2, self.NQ))

        # set derivatives
        bc_dq[0:self.NQ, :]                               = -np.eye(self.NQ)
        bc_dq[self.NQ + self.NS:self.NQ * 2 + self.NS, :] = np.eye(self.NQ)

        return bc_dq

    # =========================================================================

    def bc_dsds(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):



        # allocate memory
        bc_dsds = np.zeros((self.NQ * 2 + self.NS * 2, self.NS, self.NS))

        return bc_dsds

    # =========================================================================

    def bc_dpdp(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):



        # allocate memory
        bc_dpdp = np.zeros((self.NQ * 2 + self.NS * 2, self.NP, self.NP))

        return bc_dpdp

    # =========================================================================

    def bc_dqdq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):



        # allocate memory
        bc_dqdq = np.zeros((self.NQ * 2 + self.NS * 2, self.NQ, self.NQ))

        return bc_dqdq

    # =========================================================================

    def bc_dsdp(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):



        # allocate memory
        bc_dsdp = np.zeros((self.NQ * 2 + self.NS * 2, self.NS, self.NP))

        return bc_dsdp

    # =========================================================================

    def bc_dsdq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):



        # allocate memory
        bc_dsdq = np.zeros((self.NQ * 2 + self.NS * 2, self.NS, self.NQ))

        return bc_dsdq

    # =========================================================================

    def bc_dpdq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):



        # allocate memory
        bc_dpdq = np.zeros((self.NQ * 2 + self.NS * 2, self.NP, self.NQ))

        return bc_dpdq

    # =========================================================================

    def obj(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):



        return self.sign * xs[-1, -1, -1]

    # =========================================================================

    def obj_ds(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):



        return self.sign * xs[-1, -1, -1], self.sign * xs_dot1[-1, -1, -1, :]

    # =========================================================================

    def obj_dp(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):



        return self.sign * xs[-1, -1, -1], self.sign * xs_dot1[-1, -1, -1, :]

    # =========================================================================

    def obj_dq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):



        return self.sign * xs[-1, -1, -1], self.sign * xs_dot1[-1, -1, -1, :]

    # =========================================================================

    def obj_dsds(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):



        return self.sign * xs[-1, -1, -1], self.sign * xs_dot1[-1, -1, -1, :], self.sign * xs_dot2[-1, -1, -1, :], self.sign * xs_ddot[-1, -1, -1, :, :]


    # =========================================================================

    def obj_dsdp(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):



        return self.sign * xs[-1, -1, -1], self.sign * xs_dot1[-1, -1, -1, :], self.sign * xs_dot2[-1, -1, -1, :], self.sign * xs_ddot[-1, -1, -1, :, :]



    # =========================================================================

    def obj_dsdq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):



        return self.sign * xs[-1, -1, -1], self.sign * xs_dot1[-1, -1, -1, :], self.sign * xs_dot2[-1, -1, -1, :], self.sign * xs_ddot[-1, -1, -1, :, :]


    # =========================================================================

    def obj_dpdq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):



        return self.sign * xs[-1, -1, -1], self.sign * xs_dot1[-1, -1, -1, :], self.sign * xs_dot2[-1, -1, -1, :], self.sign * xs_ddot[-1, -1, -1, :, :]


#!/usr/bin/env python
# -*- coding: utf-8 -*-

# system imports
import numpy as np
import datetime as datetime
import scipy.optimize as opt

# project imports
from utilities import rq

# third-party imports ... COMMENT OUT IF NOT AVAILABLE
import snopt

# =============================================================================

def solve(self):

    if self.solver == "snopt":
        self.snopt()

    if self.solver == "scipy":
        self.scipy()

# =========================================================================

def snopt(self):

    # introdude some abbreviations
    x0   = self.ocp.x0
    xend = self.ocp.xend
    p    = self.ocp.p
    q0   = self.ocp.q
    s0   = self.ocp.s
    NQ   = self.ocp.NQ + self.ocp.NS   # add the shooting variables as controls
    NC   = self.ocp.NC + self.ocp.NMC  # add matching conditions for boundary

    # =====================================================================

    def setup(inform, Prob, neF, n, iAfun, jAvar, lenA, neA, A,
              iGfun, jGvar, lenG, neG, ObjAdd, ObjRow, xlow, xupp, Flow,
              Fupp, x, xstate, Fmul):

        # give the problem a name.
        Prob[:3] = list('ocp')

        # assign the dimensions of the constraint Jacobian
        neF[0] = 1 + NC
        n[0]   = NQ

        # set the objective row
        ObjRow[0] = 1
        ObjAdd[0] = 0
        Flow[0]   = -1e6
        Fupp[0]   = 1e6

        # set the upper and lower bounds for the inequality constraints
        Flow[1:1 + self.ocp.NCG] = -1e6
        Fupp[1:1 + self.ocp.NCG] = 0

        # set the upper and lower bounds for the equality constraints
        Flow[1 + self.ocp.NCG:1 + self.ocp.NC] = 0
        Fupp[1 + self.ocp.NCG:1 + self.ocp.NC] = 0

        # set the upper and lower bounds for the matching conditions
        Flow[1 + self.ocp.NC:] = 0
        Fupp[1 + self.ocp.NC:] = 0

        # set the upper and lower bounds for the controls q
        for i in xrange(0, self.ocp.NU):
            xlow[i * self.ocp.NTS:(i + 1) * self.ocp.NTS] = self.ocp.bnds[i, 0]
            xupp[i * self.ocp.NTS:(i + 1) * self.ocp.NTS] = self.ocp.bnds[i, 1]

        # set the upper and lower bounds for the shooting variables s
        xlow[self.ocp.NQ:] = -1e6
        xupp[self.ocp.NQ:] = 1e6

        # fix the shooting variables s at the boundaries if necessary
        for i in xrange(0, self.ocp.NX):

            if x0[i] is not None:
                xlow[self.ocp.NQ + i] = x0[i]
                xupp[self.ocp.NQ + i] = x0[i]

            if xend[i] is not None:
                xlow[self.ocp.NQ + self.ocp.NS - self.ocp.NX + i] = xend[i]
                xupp[self.ocp.NQ + self.ocp.NS - self.ocp.NX + i] = xend[i]

        # set xstate
        xstate[0:NQ] = 0

        # set up pattern for the jacobian
        neG[0] = NQ * (1 + NC)
        l = 0

        for i in xrange(0, NC + 1):
            for j in xrange(0, NQ):

                iGfun[l + j] = i + 1
                jGvar[l + j] = j + 1

            l = l + NQ

    # =====================================================================

    def evaluate(status, x, needF, neF, F, needG, neG, G, cu, iu, ru):

        # separate controls and shooting variables for readability
        q = x[:self.ocp.NQ]
        s = x[self.ocp.NQ:]

        if needF[0] != 0:

            # integrate for current setting
            xs = self.ocp.integrate(p, q, s)

            # calculate objective for current controls
            F[0] = self.ocp.obj(xs, None, None, None, p, q, s)

            # evaluate the inequality constraints
            F[1:self.ocp.NCG + 1] = self.ocp.ineqc(xs, None, None, None, p, q, s)

            # evaluate the equality constraints
            F[self.ocp.NCG + 1:self.ocp.NC + 1] = self.ocp.eqc(xs, None, None, None, p, q, s)

            # evaluate the matching conditions
            F[self.ocp.NC + 1:] = self.ocp.mc(xs, None, None, None, p, q, s)

        if needG[0] != 0:

            # integrate
            xs, xs_dot_q = self.ocp.integrate_dq(p, q, s)
            xs_dot_s     = self.ocp.integrate_ds(p, q, s)[1]

            # calculate derivatives of objective
            obj_dq = self.ocp.obj_dq(xs, xs_dot_q, None, None, p, q, s)[1] # DEBUG STUFF!
            obj_ds = self.ocp.obj_ds(xs, xs_dot_s, None, None, p, q, s)[1] # DEBUG STUFF!

            G[0:self.ocp.NQ]  = self.ocp.obj_dq(xs, xs_dot_q, None, None, p, q, s)[1]  # controls
            G[self.ocp.NQ:NQ] = self.ocp.obj_ds(xs, xs_dot_s, None, None, p, q, s)[1]  # shooting variables
            l                 = NQ

            # calculate derivatives of inequality constraints
            ineqc_dq = self.ocp.ineqc_dq(xs, xs_dot_q, None, None, p, q, s)[1]
            ineqc_ds = self.ocp.ineqc_ds(xs, xs_dot_s, None, None, p, q, s)[1]

            for i in xrange(0, self.ocp.NCG):
                G[l:l + self.ocp.NQ]      = ineqc_dq[i, :] # controls
                G[l + self.ocp.NQ:l + NQ] = ineqc_ds[i, :] # shooting variables
                l                         = l + NQ

            # calculate derivatives of equality constraints
            eqc_dq = self.ocp.eqc_dq(xs, xs_dot_q, None, None, p, q, s)[1]
            eqc_ds = self.ocp.eqc_ds(xs, xs_dot_s, None, None, p, q, s)[1]

            for i in xrange(0, self.ocp.NCH):
                G[l:l + self.ocp.NQ]      = eqc_dq[i, :] # controls
                G[l + self.ocp.NQ:l + NQ] = eqc_ds[i, :] # shooting variables
                l                         = l + NQ

            # calculate derivatives of matching conditions
            mc_dq = self.ocp.mc_dq(xs, xs_dot_q, None, None, p, q, s)[1]
            mc_ds = self.ocp.mc_ds(xs, xs_dot_s, None, None, p, q, s)[1]

            for i in xrange(0, self.ocp.NMC):
                G[l:l + self.ocp.NQ]      = mc_dq[i, :] # controls
                G[l + self.ocp.NQ:l + NQ] = mc_ds[i, :] # shooting variables
                l                         = l + NQ

            # START DEBUG STUFF ------------------------

            # import scipy.sparse as sps
            # import matplotlib.pylab as pl

            # M = sps.csr_matrix(obj_dq)
            # pl.spy(M)
            # pl.show()

            # M = sps.csr_matrix(obj_ds)
            # pl.spy(M)
            # pl.show()

            # M = sps.csr_matrix(ineqc_dq)
            # pl.spy(M)
            # pl.show()

            # M = sps.csr_matrix(ineqc_ds)
            # pl.spy(M)
            # pl.show()

            # M = sps.csr_matrix(eqc_dq)
            # pl.spy(M)
            # pl.show()

            # M = sps.csr_matrix(eqc_ds)
            # pl.spy(M)
            # pl.show()

            # M = sps.csr_matrix(mc_dq)
            # pl.spy(M)
            # pl.show()

            # M = sps.csr_matrix(mc_ds)
            # pl.spy(M)
            # pl.show()

            # END DEBUG STUFF ------------------------

    snopt.check_memory_compatibility()
    minrw = np.zeros((1), dtype=np.int32)
    miniw = np.zeros((1), dtype=np.int32)
    mincw = np.zeros((1), dtype=np.int32)

    rw = np.zeros((1000000,), dtype=np.float64)
    iw = np.zeros((1000000,), dtype=np.int32)
    cw = np.zeros((10000,), dtype=np.character)

    Cold  = np.array([0], dtype=np.int32)
    Basis = np.array([1], dtype=np.int32)
    Warm  = np.array([2], dtype=np.int32)

    x    = np.append(q0, s0)
    x    = np.array(x, dtype=np.float64)
    xlow = np.zeros((NQ,), dtype=np.float64)
    xupp = np.zeros((NQ,), dtype=np.float64)
    xmul = np.zeros((NQ,), dtype=np.float64)
    F    = np.zeros((1 + NC,), dtype=np.float64)
    Flow = np.zeros((1 + NC,), dtype=np.float64)
    Fupp = np.zeros((1 + NC,), dtype=np.float64)
    Fmul = np.zeros((1 + NC,), dtype=np.float64)

    ObjAdd = np.zeros((1,), dtype=np.float64)

    xstate = np.zeros((NQ,), dtype=np.int32)
    Fstate = np.zeros((1 + NC,), dtype=np.int32)

    INFO   = np.zeros((1,), dtype=np.int32)
    ObjRow = np.zeros((1,), dtype=np.int32)
    n      = np.zeros((1,), dtype=np.int32)
    neF    = np.zeros((1,), dtype=np.int32)

    lenA    = np.zeros((1,), dtype=np.int32)
    lenA[0] = NQ * (1 + NC)

    iAfun = np.zeros((lenA[0],), dtype=np.int32)
    jAvar = np.zeros((lenA[0],), dtype=np.int32)

    A = np.zeros((lenA[0],), dtype=np.float64)

    lenG    = np.zeros((1,), dtype=np.int32)
    lenG[0] = NQ * (1 + NC)

    iGfun = np.zeros((lenG[0],), dtype=np.int32)
    jGvar = np.zeros((lenG[0],), dtype=np.int32)

    neA = np.zeros((1,), dtype=np.int32)
    neG = np.zeros((1,), dtype=np.int32)

    nxname = np.zeros((1,), dtype=np.int32)
    nFname = np.zeros((1,), dtype=np.int32)

    nxname[0] = 1
    nFname[0] = 1

    xnames = np.zeros((1 * 8,), dtype=np.character)
    Fnames = np.zeros((1 * 8,), dtype=np.character)
    Prob   = np.zeros((200 * 8,), dtype=np.character)

    iSpecs  = np.zeros((1,), dtype=np.int32)
    iSumm   = np.zeros((1,), dtype=np.int32)
    iPrint  = np.zeros((1,), dtype=np.int32)

    iSpecs[0]   = 4
    iSumm[0]    = 6
    iPrint[0]   = 9

    printname = np.zeros((200 * 8,), dtype=np.character)
    specname  = np.zeros((200 * 8,), dtype=np.character)

    nS   = np.zeros((1,), dtype=np.int32)
    nInf = np.zeros((1,), dtype=np.int32)
    sInf = np.zeros((1,), dtype=np.float64)

    # open output files using snfilewrappers.[ch] */
    specn  = self.ocp.path + "/snopt.spc"
    printn = self.ocp.path + "/output/" + \
             datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + \
             "-snopt.out"
    specname[:len(specn)]   = list(specn)
    printname[:len(printn)] = list(printn)

    # Open the print file, fortran style */
    snopt.snopenappend(iPrint, printname, INFO)

    # initialize snopt to its default parameter
    snopt.sninit(iPrint, iSumm, cw, iw, rw)

    # set up problem to be solved
    setup(INFO, Prob, neF, n, iAfun, jAvar, lenA, neA, A,
          iGfun, jGvar, lenG, neG, ObjAdd, ObjRow, xlow, xupp, Flow,
          Fupp, x, xstate, Fmul)

    # open spec file
    snopt.snfilewrapper(specname, iSpecs, INFO, cw, iw, rw)

    if INFO[0] != 101:
        print("Warning: Trouble reading specs file %s \n" % (specname))

    # set options not specified in the spec file
#        iPrt   = np.array([0], dtype=np.int32)
#        iSum   = np.array([0], dtype=np.int32)
#        strOpt = np.zeros((200*8,), dtype=np.character)

#        DerOpt = np.zeros((1,), dtype=np.int32)
#        DerOpt[0] = 1
#        strOpt_s = "Derivative option"
#        strOpt[:len(strOpt_s)] = list(strOpt_s)
#        snopt.snseti(strOpt, DerOpt, iPrt, iSum, INFO, cw, iw, rw)

    # call snopt
    snopt.snopta(Cold, neF, n, nxname, nFname,
                 ObjAdd, ObjRow, Prob, evaluate,
                 iAfun, jAvar, lenA, neA, A,
                 iGfun, jGvar, lenG, neG,
                 xlow, xupp, xnames, Flow, Fupp, Fnames,
                 x, xstate, xmul, F, Fstate, Fmul,
                 INFO, mincw, miniw, minrw,
                 nS, nInf, sInf, cw, iw, rw, cw, iw, rw)

    snopt.snclose(iPrint)
    snopt.snclose(iSpecs)


    # save results
    p = p
    q = x[:self.ocp.NQ]
    s = x[self.ocp.NQ:]
    xs, xs_dot_q = self.ocp.integrate_dq(p, q, s)
    xs_dot_s = self.ocp.integrate_ds(p, q, s)[1]

    self.ocp.optimal = {}
    self.ocp.optimal["q"] = q
    self.ocp.optimal["s"] = s
    self.ocp.optimal["xs"] = xs
    self.ocp.optimal["F"] = F[0]
    self.ocp.optimal["cineq"] = F[1:self.ocp.NCG + 1]
    self.ocp.optimal["cineq_mul"] = -Fmul[1:self.ocp.NCG + 1]
    self.ocp.optimal["ceq"] = F[self.ocp.NCG + 1:self.ocp.NC + 1]
    self.ocp.optimal["ceq_mul"] = -Fmul[self.ocp.NCG + 1:self.ocp.NC + 1]
    self.ocp.optimal["mc"] = F[self.ocp.NC + 1:]
    self.ocp.optimal["mc_mul"] = -Fmul[self.ocp.NC + 1:]
    self.ocp.optimal["bcq"] = self.ocp.bcq(xs, None, None, None, p, q, s)
    self.ocp.optimal["bcs"] = self.ocp.bcs(xs, None, None, None, p, q, s)

    # =====================================================================

    # allocate memory
    self.NINEQCA = 0
    self.NMCA = 0
    self.NBCQA = 0
    self.NBCSA = 0
    self.NCA = 0

    self.ineqca = []
    self.mca = []
    self.bcqa = []
    self.bcsa = []

    # evaluate the active inequality constraints
    for i in xrange(0, self.ocp.NCG):
        if self.ocp.optimal["cineq_mul"][i] != 0:
            self.ineqca.append(i)

    self.NINEQCA = len(self.ineqca)

    # evaluate the active matching conditions
    for i in xrange(0, self.ocp.NMC):
        if self.ocp.optimal["mc_mul"][i] != 0:
            self.mca.append(i)

    self.NMCA = len(self.mca)

    # evaluate the active box constraints for q
    for i in xrange(0, 2 * self.ocp.NQ):
        if self.ocp.optimal["bcq"][i] >= -1e-6:
            self.bcqa.append(i)

    self.NBCQA = len(self.bcqa)

    # evaluate the active box constraints for s
    for i in xrange(0, self.ocp.NS):
        if self.ocp.optimal["bcs"][i] >= -1e-6:
            self.bcsa.append(i)

    self.NBCSA = len(self.bcsa)

    # allocate memory for jacobian of active constraints
    self.NCA = (self.NINEQCA + self.ocp.NCH +
                self.NMCA + self.NBCQA + self.NBCSA)
    J_ca = np.zeros((self.NCA, self.ocp.NQ + self.ocp.NS))

    # calculate dq and ds of all active constraints
    if self.ocp.NG > 0:
        ineqca_dq = self.ocp.ineqc_dq(
            xs, xs_dot_q, None, None, p, q, s)[1][self.ineqca]
        ineqca_ds = self.ocp.ineqc_ds(
            xs, xs_dot_s, None, None, p, q, s)[1][self.ineqca]

    if self.ocp.NH > 0:
        eqc_dq = self.ocp.eqc_dq(
            xs, xs_dot_q, None, None, p, q, s)[1]
        eqc_ds = self.ocp.eqc_ds(
            xs, xs_dot_s, None, None, p, q, s)[1]

    mca_dq = self.ocp.mc_dq(xs, xs_dot_q, None, None, p, q, s)[1][self.mca]
    mca_ds = self.ocp.mc_ds(xs, xs_dot_s, None, None, p, q, s)[1][self.mca]

    bcq_dq = self.ocp.bcq_dq(xs, xs_dot_q, None, None, p, q, s)[self.bcqa]
    bcq_ds = self.ocp.bcq_ds(xs, xs_dot_s, None, None, p, q, s)[self.bcqa]

    bcs_dq = self.ocp.bcs_dq(xs, xs_dot_q, None, None, p, q, s)[self.bcsa]
    bcs_ds = self.ocp.bcs_ds(xs, xs_dot_s, None, None, p, q, s)[self.bcsa]

    if self.NINEQCA > 0:
        J_ca[:self.NINEQCA, :self.ocp.NQ] = ineqca_dq
        J_ca[:self.NINEQCA, self.ocp.NQ:] = ineqca_ds

    l = self.NINEQCA
    if self.ocp.NCH > 0:
        J_ca[l:l + self.ocp.NCH, :self.ocp.NQ] = eqc_dq
        J_ca[l:l + self.ocp.NCH, self.ocp.NQ:] = eqc_ds

    l = l + self.ocp.NCH
    if self.NMCA > 0:
        J_ca[l:l + self.NMCA, :self.ocp.NQ] = mca_dq
        J_ca[l:l + self.NMCA, self.ocp.NQ:] = mca_ds

    l = l + self.NMCA
    if self.NBCQA > 0:
        J_ca[l:l + self.NBCQA, :self.ocp.NQ] = bcq_dq
        J_ca[l:l + self.NBCQA, self.ocp.NQ:] = bcq_ds

    l = l + self.NBCQA
    if self.NBCSA > 0:
        J_ca[l:l + self.NBCSA, :self.ocp.NQ] = bcs_dq
        J_ca[l:l + self.NBCSA, self.ocp.NQ:] = bcs_ds

    # calculate jacobian of objective
    J_obj = np.zeros((1, self.ocp.NQ + self.ocp.NS))
    J_obj[0, :self.ocp.NQ] = self.ocp.obj_dq(
        xs, xs_dot_q, None, None, p, q, s)[1]
    J_obj[0, self.ocp.NQ:] = self.ocp.obj_ds(
        xs, xs_dot_s, None, None, p, q, s)[1]

    # calculate multipliers post-optimally
    R, Q = rq(J_ca)
    eta_a = (-J_obj.dot((Q.T).dot(np.linalg.inv(R)))).T

    print "Mulitpliers of active constraints:", eta_a

# =========================================================================

def scipy(self):

    # introdude some abbreviations
    x0 = self.ocp.x0
    xend = self.ocp.xend
    p = self.ocp.p
    q0 = self.ocp.q
    s0 = self.ocp.s
    NQ = self.ocp.NQ + self.ocp.NS
    NC = self.ocp.NC + self.ocp.NMC

    # set bounds
    bnds = []

    # set the upper and lower bounds for the controls q
    for j in xrange(0, self.ocp.NU):
        for k in xrange(0, self.ocp.NTS):
            bnds.append((self.ocp.bnds[j, 0], self.ocp.bnds[j, 1]))

    # fix the shooting variables s at the boundaries if necessary
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

    # =====================================================================

    def obj(x):

        # separate controls and shooting variables for readability
        q = x[:self.ocp.NQ]
        s = x[self.ocp.NQ:]

        # integrate
        xs, xs_dot_q = self.ocp.integrate_dq(p, q, s)
        xs_dot_s     = self.ocp.integrate_ds(p, q, s)[1]

        # evaluate gradient of objective
        obj, obj_dq = self.ocp.obj_dq(xs, xs_dot_q, None, None, p, q, s)

        # allocate memory
        jac = np.zeros((NQ,))

        # build jacobian
        jac[0:self.ocp.NQ] = obj_dq
        jac[self.ocp.NQ:]  = self.ocp.obj_ds(xs, xs_dot_s, None, None, p, q, s)[1]

        return obj, jac

    # =====================================================================

    def ineqc(x):

        # separate controls and shooting variables for readability
        q = x[:self.ocp.NQ]
        s = x[self.ocp.NQ:]

        # integrate and evaluate constraints
        xs = self.ocp.integrate(p, q, s)
        c  = -self.ocp.ineqc(xs, None, None, None, p, q, s)

        return c

    # =====================================================================

    def ineqc_jac(x):

        # separate controls and shooting variables for readability
        q = x[:self.ocp.NQ]
        s = x[self.ocp.NQ:]

        # integrate
        xs, xs_dot_q = self.ocp.integrate_dq(p, q, s)
        xs_dot_s     = self.ocp.integrate_ds(p, q, s)[1]

        # allocate memory
        jac = np.zeros((self.ocp.NCG, NQ))

        # build jacobian
        jac[:, 0:self.ocp.NQ] = -self.ocp.ineqc_dq(xs, xs_dot_q, None, None, p, q, s)[1]
        jac[:, self.ocp.NQ:]  = -self.ocp.ineqc_ds(xs, xs_dot_s, None, None, p, q, s)[1]

        return jac

    # =====================================================================

    def eqc(x):

        # separate controls and shooting variables for readability
        q = x[:self.ocp.NQ]
        s = x[self.ocp.NQ:]

        # integrate and evaluate constraints
        xs = self.ocp.integrate(p, q, s)
        c  = self.ocp.eqc(xs, None, None, None, p, q, s)

        return c

    # =====================================================================

    def eqc_jac(x):

        # separate controls and shooting variables for readability
        q = x[:self.ocp.NQ]
        s = x[self.ocp.NQ:]

        # integrate
        xs, xs_dot_q = self.ocp.integrate_dq(p, q, s)
        xs_dot_s     = self.ocp.integrate_ds(p, q, s)[1]

        # allocate memory
        jac = np.zeros((self.ocp.NCH, NQ))

        # build jacobian
        jac[:, 0:self.ocp.NQ] = self.ocp.eqc_dq(xs, xs_dot_q, None, None, p, q, s)[1]
        jac[:, self.ocp.NQ:]  = self.ocp.eqc_ds(xs, xs_dot_s, None, None, p, q, s)[1]

        return jac

    # =====================================================================

    def mc(x):

        # separate controls and shooting variables for readability
        q = x[:self.ocp.NQ]
        s = x[self.ocp.NQ:]

        # integrate and evaluate constraints
        xs = self.ocp.integrate(p, q, s)
        mc = self.ocp.mc(xs, None, None, None, p, q, s)

        return mc

    # =====================================================================

    def mc_jac(x):

        # separate controls and shooting variables for readability
        q = x[:self.ocp.NQ]
        s = x[self.ocp.NQ:]

        # integrate
        xs, xs_dot_q = self.ocp.integrate_dq(p, q, s)
        xs_dot_s     = self.ocp.integrate_ds(p, q, s)[1]

        # allocate memory
        jac = np.zeros((self.ocp.NMC, NQ))

        # build jacobian
        jac[:, 0:self.ocp.NQ] = self.ocp.mc_dq(xs, xs_dot_q, None, None, p, q, s)[1]
        jac[:, self.ocp.NQ:]  = self.ocp.mc_ds(xs, xs_dot_s, None, None, p, q, s)[1]

        return jac

    # =====================================================================

    # set initial guess
    x = np.append(q0, s0)

    # inequality constraints only
    if self.ocp.NG > 0 and self.ocp.NH == 0:

        # call solver
        scipy_results = opt.minimize(obj, x, args=(), method="SLSQP", jac=True, bounds=bnds,
                                          constraints=({"type":"ineq", "fun":ineqc, "jac":ineqc_jac},
                                                       {"type":"eq", "fun":mc, "jac":mc_jac}),
                                          options={"disp":True, "iprint":2, "ftol":1e-9})

        # detailed output
        print scipy_results

        self.ocp.optimal = {}
        self.ocp.optimal["q"]         = scipy_results.x[:self.ocp.NQ]
        self.ocp.optimal["s"]         = scipy_results.x[self.ocp.NQ:]
        self.ocp.optimal["F"]         = scipy_results.fun
        self.ocp.optimal["cineq"]     = ineqc(scipy_results.x)
        self.ocp.optimal["cineq_mul"] = []
        self.ocp.optimal["ceq"]       = []
        self.ocp.optimal["ceq_mul"]   = []
        self.ocp.optimal["mc"]        = mc(scipy_results.x)
        self.ocp.optimal["mc_mul"]    = []
        self.ocp.optimal["xs"]        = self.ocp.integrate(p,
                                                           scipy_results.x[:self.ocp.NQ],
                                                           scipy_results.x[self.ocp.NQ:])

    # equality constraints only
    elif self.ocp.NG == 0 and self.ocp.NH > 0:

        # call solver
        scipy_results = opt.minimize(obj, x, args=(), method="SLSQP", jac=True, bounds=bndsb,
                                          constraints=({"type":"eq", "fun":eqc, "jac":eqc_jac},
                                                       {"type":"eq", "fun":mc, "jac":mc_jac}),
                                          options={"disp":True, "iprint":2, "ftol":1e-9})

        # detailed output
        print scipy_results

        self.ocp.optimal = {}
        self.ocp.optimal["q"]         = scipy_results.x[:self.ocp.NQ]
        self.ocp.optimal["s"]         = scipy_results.x[self.ocp.NQ:]
        self.ocp.optimal["F"]         = scipy_results.fun
        self.ocp.optimal["cineq"]     = []
        self.ocp.optimal["cineq_mul"] = []
        self.ocp.optimal["ceq"]       = eqc(scipy_results.x)
        self.ocp.optimal["ceq_mul"]   = []
        self.ocp.optimal["mc"]        = mc(scipy_results.x)
        self.ocp.optimal["mc_mul"]    = []
        self.ocp.optimal["xs"]        = self.ocp.integrate(p,
                                                           scipy_results.x[:self.ocp.NQ],
                                                           scipy_results.x[self.ocp.NQ:])

    # inequality and equality constraints
    elif self.ocp.NG > 0 and self.ocp.NH > 0:

        # call solver
        scipy_results = opt.minimize(obj, x, args=(), method="SLSQP", jac=True, bounds=bnds,
                                          constraints=({"type":"ineq", "fun":ineqc, "jac":ineqc_jac},
                                                       {"type":"eq", "fun":eqc, "jac":eqc_jac},
                                                       {"type":"eq", "fun":mc, "jac":mc_jac}),
                                          options={"disp":True, "iprint":2, "ftol":1e-9})

        # detailed output
        print scipy_results

        self.ocp.optimal = {}
        self.ocp.optimal["q"]         = scipy_results.x[:self.ocp.NQ]
        self.ocp.optimal["s"]         = scipy_results.x[self.ocp.NQ:]
        self.ocp.optimal["F"]         = scipy_results.fun
        self.ocp.optimal["cineq"]     = ineqc(scipy_results.x)
        self.ocp.optimal["cineq_mul"] = []
        self.ocp.optimal["ceq"]       = eqc(scipy_results.x)
        self.ocp.optimal["ceq_mul"]   = []
        self.ocp.optimal["mc"]        = mc(scipy_results.x)
        self.ocp.optimal["mc_mul"]    = []
        self.ocp.optimal["xs"]        = self.ocp.integrate(p,
                                                           scipy_results.x[:self.ocp.NQ],
                                                           scipy_results.x[self.ocp.NQ:])

    # no additional constraints
    else:

        # call solver
        scipy_results = opt.minimize(obj, x, args=(), method="SLSQP", jac=True, bounds=bnds,
                                          constraints=({"type":"eq", "fun":mc, "jac":mc_jac}),
                                          options={"disp":True, "iprint":2, "ftol":1e-9})

        # detailed output
        print scipy_results

        self.ocp.optimal = {}
        self.ocp.optimal["q"]         = scipy_results.x[:self.ocp.NQ]
        self.ocp.optimal["s"]         = scipy_results.x[self.ocp.NQ:]
        self.ocp.optimal["F"]         = scipy_results.fun
        self.ocp.optimal["cineq"]     = []
        self.ocp.optimal["cineq_mul"] = []
        self.ocp.optimal["ceq"]       = []
        self.ocp.optimal["ceq_mul"]   = []
        self.ocp.optimal["mc"]        = mc(scipy_results.x)
        self.ocp.optimal["mc_mul"]    = []
        self.ocp.optimal["xs"]        = self.ocp.integrate(p,
                                                           scipy_results.x[:self.ocp.NQ],
                                                           scipy_results.x[self.ocp.NQ:])

# =============================================================================