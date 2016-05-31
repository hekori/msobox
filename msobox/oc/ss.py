# -*- coding: utf-8 -*-

"""
===============================================================================

discretized optimal control problem for single shooting

===============================================================================
"""

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

"""
===============================================================================
"""

class Problem(object):

    """

    provides functionalities for ...

    """

    """
    ===============================================================================
    """

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

    """
    ===============================================================================
    """

    def flat2array_s(self, s):

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
        s_array = np.zeros((2, self.NX))

        # convert shooting variables from one-dimensional to 3-dimensional
        for i in xrange(0, 2):
            s_array[i, :] = s[i * self.NX:(i + 1) * self.NX]

        return s_array

    """
    ===============================================================================
    """

    def approximate_s(self):

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
        self.s = np.zeros((2 * self.NX,))

        # approximate shooting variables by linear interpolation if possible
        for i in xrange(0, self.NX):

            # set initial shooting variables to x0 if possible
            if self.x0[i] is not None:
                self.s[i] = self.x0[i]

            # set end shooting variables to xend if possible
            if self.xend[i] is not None:
                self.s[self.NX + i] = self.xend[i]

    """
    ===============================================================================
    """

    def prepare(self):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        self.NCG = self.NG * self.NTS             # number of inequality constraints
        self.NCH = self.NH * self.NTS             # number of equality constraints
        self.NC  = self.NCG + self.NCH            # total number of constraints
        self.NQI = 1                              # number of controls per shooting interval
        self.NQ  = self.NU * self.NTS * self.NQI  # number of controls
        self.NS  = 2 * self.NX                    # number of shooting variables
        self.NMC = self.NX                        # number of matching conditions

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

    """
    ===============================================================================
    """

    def plot(self):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        p = self.p
        q = self.q
        s = self.s

        q_plot = self.flat2array_q(q)[:, :, 0]
        x_plot = self.integrate(p, q, s)

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

        # convert controls and shooting variables
        q = self.flat2array_q(q)
        s = self.flat2array_s(s)

        # allocate memory
        xs = np.zeros((self.NTS, self.NX))

        # set initial conditions
        xs[0, :] = s[0, :]

        # integrate
        for i in xrange(0, self.NTS - 1):

            # set initial conditions
            x0 = xs[i, :]

            # set time steps for this interval
            tsi = np.linspace(self.ts[i], self.ts[i + 1], self.NTSI)

            # set constant controls for this interval
            q_interval = q[:, i, 0]

            xs[i + 1, :] = self.ind.zo_forward(tsi,
                                               x0,
                                               p,
                                               q_interval)

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

        # convert controls and shooting variables
        q = self.flat2array_q(q)
        s = self.flat2array_s(s)

        # allocate memory
        xs     = np.zeros((self.NTS, self.NX))
        xs_dot = np.zeros((self.NTS, self.NX, self.NS))

        # set up directions and initial conditions
        xs[0, :]                = s[0, :]
        xs_dot[0, :, 0:self.NX] = np.eye(self.NX)
        p_dot                   = np.zeros((self.NP, self.NX))
        q_dot                   = np.zeros((self.NU, self.NX))

        # integrate
        for i in xrange(0, self.NTS - 1):

            # set initial conditions
            x0     = xs[i, :]
            x0_dot = xs_dot[i, :, 0:self.NX]

            # set time steps for this interval
            tsi = np.linspace(self.ts[i], self.ts[i + 1], self.NTSI)

            # set constant controls for this interval
            q_interval = q[:, i, 0]

            # integrate
            xs[i + 1, :], xs_dot[i + 1, :, 0:self.NX] = self.ind.fo_forward(tsi,
                                                                            x0, x0_dot,
                                                                            p, p_dot,
                                                                            q_interval, q_dot)

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

        # convert controls and shooting variables
        q = self.flat2array_q(q)
        s = self.flat2array_s(s)

        # allocate memory
        xs     = np.zeros((self.NTS, self.NX))
        xs_dot = np.zeros((self.NTS, self.NX, self.NP))

        # set up directions for differentation
        x0_dot = np.zeros((self.NX, self.NP))
        p_dot  = np.eye(self.NP)
        q_dot  = np.zeros((self.NU, self.NP))

        # set initial conditions
        xs[0, :]        = s[0, :]
        xs_dot[0, :, :] = x0_dot

        # integrate
        for i in xrange(0, self.NTS - 1):

            # set initial conditions
            x0     = xs[i, :]
            x0_dot = xs_dot[i, :, :]

            # set time steps for this interval
            tsi = np.linspace(self.ts[i], self.ts[i + 1], self.NTSI)

            # set constant controls for this interval
            q_interval = q[:, i, 0]

            xs[i + 1, :], xs_dot[i + 1, :, :] = self.ind.fo_forward(tsi,
                                                                    x0, x0_dot,
                                                                    p, p_dot,
                                                                    q_interval, q_dot)

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

        # convert controls and shooting variables
        q = self.flat2array_q(q)
        s = self.flat2array_s(s)

        # allocate memory
        xs     = np.zeros((self.NTS, self.NX))
        xs_dot = np.zeros((self.NTS, self.NX, self.NQ))

        # set initial conditions
        xs[0, :] = s[0, :]

        # integrate
        for i in xrange(0, self.NTS - 1):

            # set time steps for this interval
            tsi = np.linspace(self.ts[i], self.ts[i + 1], self.NTSI)

            # set constant controls for this interval
            q_interval = q[:, i, 0]

            # set initial conditions and directions
            x0                                      = xs[i, :]
            x0_dot                                  = xs_dot[i, :, 0:(i + 1) * self.NU]
            p_dot                                   = np.zeros((self.NP, (i + 1) * self.NU))
            q_dot                                   = np.zeros((self.NU, (i + 1) * self.NU))
            q_dot[:, i * self.NU:(i + 1) * self.NU] = np.eye(self.NU)

            # integrate
            xs[i + 1, :], xs_dot[i + 1, :, 0:(i + 1) * self.NU] = self.ind.fo_forward(tsi,
                                                                                      x0, x0_dot,
                                                                                      p, p_dot,
                                                                                      q_interval, q_dot)

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

        # convert controls and shooting variables
        q = self.flat2array_q(q)
        s = self.flat2array_s(s)

        # allocate memory
        xs      = np.zeros((self.NTS, self.NX))
        xs_dot1 = np.zeros((self.NTS, self.NX, self.NS))
        xs_dot2 = np.zeros((self.NTS, self.NX, self.NS))
        xs_ddot = np.zeros((self.NTS, self.NX, self.NS, self.NS))

        # set up directions and initial conditions
        xs[0, :]                 = s[0, :]
        xs_dot1[0, :, 0:self.NX] = np.eye(self.NX)
        xs_dot2[0, :, 0:self.NX] = np.eye(self.NX)
        x0_ddot                  = np.zeros((self.NX, self.NX, self.NX))
        p_dot                    = np.zeros((self.NP, self.NX))
        p_ddot                   = np.zeros((self.NP, self.NX, self.NX))
        q_dot                    = np.zeros((self.NU, self.NX))
        q_ddot                   = np.zeros((self.NU, self.NX, self.NX))

        # integrate
        for i in xrange(0, self.NTS - 1):

            # set initial conditions
            x0      = xs[i, :]
            x0_dot1 = xs_dot1[i, :, 0:self.NX]
            x0_dot2 = xs_dot2[i, :, 0:self.NX]

            # set time steps for this interval
            tsi = np.linspace(self.ts[i], self.ts[i + 1], self.NTSI)

            # set constant controls for this interval
            q_interval = q[:, i, 0]

            # integrate
            xs[i + 1, :], xs_dot1[i + 1, :, 0:self.NX], xs_dot2[i + 1, :, 0:self.NX], xs_ddot[i + 1, :, 0:self.NX, 0:self.NX] = self.ind.so_forward(tsi,
                                                                                                                                                    x0, x0_dot2, x0_dot1, x0_ddot,
                                                                                                                                                    p, p_dot, p_dot, p_ddot,
                                                                                                                                                    q_interval, q_dot, q_dot, q_ddot)

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

        # convert controls and shooting variables
        q = self.flat2array_q(q)
        s = self.flat2array_s(s)

        # allocate memory
        xs      = np.zeros((self.NTS, self.NX))
        xs_dot1 = np.zeros((self.NTS, self.NX, self.NP))
        xs_dot2 = np.zeros((self.NTS, self.NX, self.NP))
        xs_ddot = np.zeros((self.NTS, self.NX, self.NP, self.NP))

        # set up directions and initial conditions
        xs[0, :] = s[0, :]
        x0_ddot  = np.zeros((self.NX, self.NP, self.NP))
        p_dot    = np.eye(self.NP)
        p_ddot   = np.zeros((self.NP, self.NP, self.NP))
        q_dot    = np.zeros((self.NU, self.NP))
        q_ddot   = np.zeros((self.NU, self.NP, self.NP))

        # integrate
        for i in xrange(0, self.NTS - 1):

            # set initial conditions
            x0      = xs[i, :]
            x0_dot1 = xs_dot1[i, :, :]
            x0_dot2 = xs_dot2[i, :, :]

            # set time steps for this interval
            tsi = np.linspace(self.ts[i], self.ts[i + 1], self.NTSI)

            # set constant controls for this interval
            q_interval = q[:, i, 0]

            # integrate
            xs[i + 1, :], xs_dot1[i + 1, :, :], xs_dot2[i + 1, :, :], xs_ddot[i + 1, :, :, :] = self.ind.so_forward(tsi,
                                                                                                                    x0, x0_dot2, x0_dot1, x0_ddot,
                                                                                                                    p, p_dot, p_dot, p_ddot,
                                                                                                                    q_interval, q_dot, q_dot, q_ddot)

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

        # convert controls and shooting variables
        q = self.flat2array_q(q)
        s = self.flat2array_s(s)

        # allocate memory
        xs      = np.zeros((self.NTS, self.NX))
        xs_dot1 = np.zeros((self.NTS, self.NX, self.NQ))
        xs_dot2 = np.zeros((self.NTS, self.NX, self.NQ))
        xs_ddot = np.zeros((self.NTS, self.NX, self.NQ, self.NQ))

        # set initial conditions
        xs[0, :] = s[0, :]

        # integrate
        for i in xrange(0, self.NTS - 1):

            # set time steps for this interval
            tsi = np.linspace(self.ts[i], self.ts[i + 1], self.NTSI)

            # set constant controls for this interval
            q_interval = q[:, i, 0]

            # set up directions and initial conditions
            x0                                      = xs[i, :]
            x0_dot1                                 = xs_dot1[i, :, 0:(i + 1) * self.NU]
            x0_dot2                                 = xs_dot2[i, :, 0:(i + 1) * self.NU]
            x0_ddot                                 = np.zeros((self.NX, (i + 1) * self.NU, (i + 1) * self.NU))
            p_dot                                   = np.zeros((self.NP, (i + 1) * self.NU))
            p_ddot                                  = np.zeros((self.NP, (i + 1) * self.NU, (i + 1) * self.NU))
            q_dot                                   = np.zeros((self.NU, (i + 1) * self.NU))
            q_dot[:, i * self.NU:(i + 1) * self.NU] = np.eye(self.NU)
            q_ddot                                  = np.zeros((self.NU, (i + 1) * self.NU, (i + 1) * self.NU))

            # integrate
            xs[i + 1, :], xs_dot1[i + 1, :, 0:(i + 1) * self.NU], xs_dot2[i + 1, :, 0:(i + 1) * self.NU], xs_ddot[i + 1, :, 0:(i + 1) * self.NU, 0:(i + 1) * self.NU] = self.ind.so_forward(tsi,
                                                                                                                                                                                            x0, x0_dot2, x0_dot1, x0_ddot,
                                                                                                                                                                                            p, p_dot, p_dot, p_ddot,
                                                                                                                                                                                            q_interval, q_dot, q_dot, q_ddot)

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

        # convert controls and shooting variables
        q = self.flat2array_q(q)
        s = self.flat2array_s(s)

        # allocate memory
        xs      = np.zeros((self.NTS, self.NX))
        xs_dot1 = np.zeros((self.NTS, self.NX, self.NS))
        xs_dot2 = np.zeros((self.NTS, self.NX, self.NP))
        xs_ddot = np.zeros((self.NTS, self.NX, self.NS, self.NP))

        # set up directions and initial conditions
        xs[0, :]                 = s[0, :]
        xs_dot1[0, :, 0:self.NX] = np.eye(self.NX)
        x0_ddot                  = np.zeros((self.NX, self.NX, self.NX))
        p_dot1                   = np.zeros((self.NP, self.NX))
        p_dot2                   = np.eye(self.NP)
        p_ddot                   = np.zeros((self.NP, self.NX, self.NX))
        q_dot1                   = np.zeros((self.NU, self.NX))
        q_dot2                   = np.zeros((self.NU, self.NP))
        q_ddot                   = np.zeros((self.NU, self.NX, self.NX))

        # integrate
        for i in xrange(0, self.NTS - 1):

            # set initial conditions
            x0      = xs[i, :]
            x0_dot1 = xs_dot1[i, :, 0:self.NX]
            x0_dot2 = xs_dot2[i, :, :]

            # set time steps for this interval
            tsi = np.linspace(self.ts[i], self.ts[i + 1], self.NTSI)

            # set constant controls for this interval
            q_interval = q[:, i, 0]

            # integrate
            xs[i + 1, :], xs_dot1[i + 1, :, 0:self.NX], xs_dot2[i + 1, :, :], xs_ddot[i + 1, :, 0:self.NX, :] = self.ind.so_forward(tsi,
                                                                                                                                    x0, x0_dot2, x0_dot1, x0_ddot,
                                                                                                                                    p, p_dot2, p_dot1, p_ddot,
                                                                                                                                    q_interval, q_dot2, q_dot1, q_ddot)

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

        # convert controls and shooting variables
        q = self.flat2array_q(q)
        s = self.flat2array_s(s)

        # allocate memory
        xs      = np.zeros((self.NTS, self.NX))
        xs_dot1 = np.zeros((self.NTS, self.NX, self.NS))
        xs_dot2 = np.zeros((self.NTS, self.NX, self.NQ))
        xs_ddot = np.zeros((self.NTS, self.NX, self.NS, self.NQ))

        # set up directions and initial conditions
        xs[0, :]                 = s[0, :]
        xs_dot1[0, :, 0:self.NX] = np.eye(self.NX)

        # integrate
        for i in xrange(0, self.NTS - 1):

            # set up directions and initial conditions
            x0                                       = xs[i, :]
            x0_dot1                                  = xs_dot1[i, :, 0:self.NX]
            x0_dot2                                  = xs_dot2[i, :, 0:(i + 1) * self.NU]
            x0_ddot                                  = np.zeros((self.NX, self.NX, (i + 1) * self.NU))
            p_dot1                                   = np.zeros((self.NP, self.NX))
            p_dot2                                   = np.zeros((self.NP, (i + 1) * self.NU))
            p_ddot                                   = np.zeros((self.NP, self.NX, (i + 1) * self.NU))
            q_dot1                                   = np.zeros((self.NU, self.NX))
            q_dot2                                   = np.zeros((self.NU, (i + 1) * self.NU))
            q_dot2[:, i * self.NU:(i + 1) * self.NU] = np.eye(self.NU)
            q_ddot                                   = np.zeros((self.NU, self.NX, (i + 1) * self.NU))

            # set time steps for this interval
            tsi = np.linspace(self.ts[i], self.ts[i + 1], self.NTSI)

            # set constant controls for this interval
            q_interval = q[:, i, 0]

            # integrate
            xs[i + 1, :], xs_dot1[i + 1, :, 0:self.NX], xs_dot2[i + 1, :, 0:(i + 1) * self.NU], xs_ddot[i + 1, :, 0:self.NX, 0:(i + 1) * self.NU] = self.ind.so_forward(tsi,
                                                                                                                                                                        x0, x0_dot2, x0_dot1, x0_ddot,
                                                                                                                                                                        p, p_dot2, p_dot1, p_ddot,
                                                                                                                                                                        q_interval, q_dot2, q_dot1, q_ddot)

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

        # convert controls and shooting variables
        q = self.flat2array_q(q)
        s = self.flat2array_s(s)

        # allocate memory
        xs      = np.zeros((self.NTS, self.NX))
        xs_dot1 = np.zeros((self.NTS, self.NX, self.NP))
        xs_dot2 = np.zeros((self.NTS, self.NX, self.NQ))
        xs_ddot = np.zeros((self.NTS, self.NX, self.NP, self.NQ))

        # set up directions and initial conditions
        xs[0, :] = s[0, :]

        # integrate
        for i in xrange(0, self.NTS - 1):

            # set up directions and initial conditions
            x0                                       = xs[i, :]
            x0_dot1                                  = xs_dot1[i, :, :]
            x0_dot2                                  = xs_dot2[i, :, 0:(i + 1) * self.NU]
            x0_ddot                                  = np.zeros((self.NX, self.NP, (i + 1) * self.NU))
            p_dot1                                   = np.eye(self.NP)
            p_dot2                                   = np.zeros((self.NP, (i + 1) * self.NU))
            p_ddot                                   = np.zeros((self.NP, self.NP, (i + 1) * self.NU))
            q_dot1                                   = np.zeros((self.NU, self.NP))
            q_dot2                                   = np.zeros((self.NU, (i + 1) * self.NU))
            q_dot2[:, i * self.NU:(i + 1) * self.NU] = np.eye(self.NU)
            q_ddot                                   = np.zeros((self.NU, self.NP, (i + 1) * self.NU))

            # set time steps for this interval
            tsi = np.linspace(self.ts[i], self.ts[i + 1], self.NTSI)

            # set constant controls for this interval
            q_interval = q[:, i, 0]

            # integrate
            xs[i + 1, :], xs_dot1[i + 1, :, :], xs_dot2[i + 1, :, 0:(i + 1) * self.NU], xs_ddot[i + 1, :, :, 0:(i + 1) * self.NU] = self.ind.so_forward(tsi,
                                                                                                                                                        x0, x0_dot2, x0_dot1, x0_ddot,
                                                                                                                                                        p, p_dot2, p_dot1, p_ddot,
                                                                                                                                                        q_interval, q_dot2, q_dot1, q_ddot)

        return xs, xs_dot1, xs_dot2, xs_ddot

    """
    ===============================================================================
    """

    def ineqc(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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
            c = np.zeros((self.NCG,))

            x = np.zeros((self.NX,))
            g = np.zeros((self.NG,))
            u = np.zeros((self.NU,))

            # loop through all time steps
            for i in xrange(0, self.NTS):

                # state and controls for this time step
                x  = xs[i, :]

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

    def ineqc_ds(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        c   = None
        ds = None

        if self.NG > 0:

            # allocate memory
            c     = np.zeros((self.NCG,))
            ds    = np.zeros((self.NCG, self.NS))

            x     = np.zeros((self.NX,))
            x_dot = np.zeros((self.NX, self.NX))
            g     = np.zeros((self.NG,))
            g_dot = np.zeros((self.NG, self.NX))
            p_dot = np.zeros((self.NP, self.NX))
            u     = np.zeros((self.NU,))
            u_dot = np.zeros((self.NU, self.NX))

            # loop through all time step
            for i in xrange(0, self.NTS):

                # state and controls for this time step
                x     = xs[i, :]
                x_dot = np.reshape(xs_dot1[i, :, 0:self.NX], x_dot.shape)

                for l in xrange(0, self.NU):
                    u[l] = q[i + l * self.NTS]

                # call fortran backend to calculate derivatives of constraint functions
                self.backend_fortran.gfcn_dot(g, g_dot, self.ts[i:i + 1], x, x_dot, p, p_dot, u, u_dot)

                # store gradient
                for k in xrange(0, self.NG):
                    ds[i + k * self.NTS, 0:self.NX] = g_dot[k, :]
                    c[i + k * self.NTS]             = g[k]

        return c, ds

    """
    ===============================================================================
    """

    def ineqc_dp(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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

                # state and controls for this time step
                x     = xs[i, :]
                x_dot = np.reshape(xs_dot1[i, :, :], x_dot.shape)

                for l in xrange(0, self.NU):
                    u[l] = q[i + l * self.NTS]

                # call fortran backend to calculate derivatives of constraint functions
                self.backend_fortran.gfcn_dot(g, g_dot, self.ts[i:i + 1], x, x_dot, p, p_dot, u, u_dot)

                # store gradient
                for k in xrange(0, self.NG):
                    c[i + k * self.NTS]       = g[k]
                    c_dp[i + k * self.NTS, :] = g_dot[k, :]

        return c, c_dp

    """
    ===============================================================================
    """

    def ineqc_dq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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
            c     = np.zeros((self.NCG,))
            dq    = np.zeros((self.NCG, self.NQ))

            x     = np.zeros((self.NX,))
            x_dot = np.zeros((self.NX, self.NU))
            g     = np.zeros((self.NG,))
            g_dot = np.zeros((self.NG, self.NU))
            p_dot = np.zeros((self.NP, self.NU))
            u     = np.zeros((self.NU,))
            u_dot = np.zeros((self.NU, self.NU))

            # loop through all time steps
            for i in xrange(0, self.NTS):

                # loop through all previous controls including the current one
                for j in xrange(0, i + 1):

                    # state and controls for this time step
                    x     = xs[i, :]
                    x_dot = np.reshape(xs_dot1[i, :, j], x_dot.shape)

                    for k in xrange(0, self.NU):
                        u[k] = q[i + k * self.NTS]

                    if i == j:
                        u_dot = np.eye(self.NU)
                    else:
                        u_dot = np.zeros((self.NU, self.NU))

                    # call fortran backend to calculate derivatives of constraint functions
                    self.backend_fortran.gfcn_dot(g, g_dot, self.ts[i:i + 1], x, x_dot, p, p_dot, u, u_dot)

                    # store gradient
                    for k in xrange(0, self.NG):
                        c[i + k * self.NTS] = g[k]

                        for l in xrange(0, self.NU):
                            dq[i + k * self.NTS, j + l * self.NTS] = g_dot[k, l]

        return c, dq

    """
    ===============================================================================
    """

    def ineqc_dsds(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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
        ds1  = None
        ds2  = None
        dsds = None

        if self.NG > 0:

            # allocate memory
            c    = np.zeros((self.NCG,))
            ds1  = np.zeros((self.NCG, self.NS))
            ds2  = np.zeros((self.NCG, self.NS))
            dsds = np.zeros((self.NCG, self.NS, self.NS))

            x      = np.zeros((self.NX,))
            x_dot1 = np.zeros((self.NX, self.NX))
            x_dot2 = np.zeros((self.NX, self.NX))
            x_ddot = np.zeros(x_dot1.shape + (self.NX,))
            g      = np.zeros((self.NG,))
            g_dot1 = np.zeros((self.NG, self.NX))
            g_dot2 = np.zeros((self.NG, self.NX))
            g_ddot = np.zeros(g_dot1.shape + (self.NX,))
            p_dot  = np.zeros((self.NP, self.NX))
            p_ddot = np.zeros(p_dot.shape + (self.NX,))
            u      = np.zeros((self.NU,))
            u_dot  = np.zeros((self.NU, self.NX))
            u_ddot = np.zeros(u_dot.shape + (self.NX,))

            # loop through all time step
            for i in xrange(0, self.NTS):

                # state and controls for this time step
                x      = xs[i, :]
                x_dot1 = np.reshape(xs_dot1[i, :, 0:self.NX], x_dot1.shape)
                x_dot2 = np.reshape(xs_dot2[i, :, 0:self.NX], x_dot2.shape)
                x_ddot = np.reshape(xs_ddot[i, :, 0:self.NX, 0:self.NX], x_ddot.shape)

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
                    dsds[i + k * self.NTS, 0:self.NX, 0:self.NX] = g_ddot[k, :, :]
                    ds1[i + k * self.NTS, 0:self.NX]             = g_dot1[k, :]
                    ds2[i + k * self.NTS, 0:self.NX]             = g_dot2[k, :]
                    c[i + k * self.NTS]                          = g[k]

        return c, ds1, ds2, dsds

    """
    ===============================================================================
    """

    def ineqc_dpdp(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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
        dp1  = None
        dp2  = None
        dpdp = None

        if self.NG > 0:

            # allocate memory
            c      = np.zeros((self.NCG,))
            dp1    = np.zeros((self.NCG, self.NP))
            dp2    = np.zeros((self.NCG, self.NP))
            dpdp   = np.zeros((self.NCG, self.NP, self.NP))

            t      = np.zeros((1,))
            x      = np.zeros((self.NX,))
            x_dot1 = np.zeros((self.NX, self.NP))
            x_dot2 = np.zeros((self.NX, self.NP))
            x_ddot = np.zeros(x_dot1.shape + (self.NP,))
            g      = np.zeros((self.NG,))
            g_dot1 = np.zeros((self.NG, self.NP))
            g_dot2 = np.zeros((self.NG, self.NP))
            g_ddot = np.zeros(g_dot1.shape + (self.NP,))
            p_dot  = np.eye(self.NP)
            p_ddot = np.zeros(p_dot.shape + (self.NP,))
            u      = np.zeros((self.NU,))
            u_dot  = np.zeros((self.NU, self.NP))
            u_ddot = np.zeros(u_dot.shape + (self.NP,))

            # loop through all time step
            for i in xrange(0, self.NTS):

                # state and controls for this time step
                x      = xs[i, :]
                x_dot1 = np.reshape(xs_dot1[i, :, :], x_dot1.shape)
                x_dot2 = np.reshape(xs_dot2[i, :, :], x_dot2.shape)
                x_ddot = np.reshape(xs_ddot[i, :, :, :], x_ddot.shape)

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
                    dpdp[i + k * self.NTS, :, :] = g_ddot[k, :, :]
                    dp1[i + k * self.NTS, :]     = g_dot1[k, :]
                    dp2[i + k * self.NTS, :]     = g_dot2[k, :]
                    c[i + k * self.NTS]          = g[k]

        return c, dp1, dp2, dpdp

    """
    ===============================================================================
    """

    def ineqc_dqdq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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
        dq1  = None
        dq2  = None
        dqdq = None

        if self.NG > 0:

            # allocate memory
            c      = np.zeros((self.NCG,))
            dq1    = np.zeros((self.NCG, self.NQ))
            dq2    = np.zeros((self.NCG, self.NQ))
            dqdq   = np.zeros((self.NCG, self.NQ, self.NQ))

            x      = np.zeros((self.NX,))
            x_dot1 = np.zeros((self.NX, self.NU))
            x_dot2 = np.zeros((self.NX, self.NU))
            x_ddot = np.zeros(x_dot1.shape + (self.NU,))
            g      = np.zeros((self.NG,))
            g_dot1 = np.zeros((self.NG, self.NU))
            g_dot2 = np.zeros((self.NG, self.NU))
            g_ddot = np.zeros(g_dot1.shape + (self.NU,))
            p_dot  = np.zeros((self.NP, self.NU))
            p_ddot = np.zeros(p_dot.shape + (self.NU,))
            u      = np.zeros((self.NU,))
            u_dot1 = np.zeros((self.NU, self.NU))
            u_dot2 = np.zeros((self.NU, self.NU))
            u_ddot = np.zeros(u_dot1.shape + (self.NU,))

            # loop through all time steps three times
            for i in xrange(0, self.NTS):

                # loop through all previous controls including the current one
                for j in xrange(0, i + 1):

                    # loop through all previous controls including the current one
                    for m in xrange(0, i + 1):

                        # state and controls for this time step
                        x      = xs[i, :]
                        x_dot1 = np.reshape(xs_dot1[i, :, j], x_dot1.shape)
                        x_dot2 = np.reshape(xs_dot2[i, :, m], x_dot2.shape)
                        x_ddot = np.reshape(xs_ddot[i, :, j, m], x_ddot.shape)

                        for k in xrange(0, self.NU):
                            u[k] = q[i + k * self.NTS]

                        if i == j:
                            u_dot1 = np.eye(self.NU)
                        else:
                            u_dot1 = np.zeros((self.NU, self.NU))

                        if i == m:
                            u_dot2 = np.eye(self.NU)
                        else:
                            u_dot2 = np.zeros((self.NU, self.NU))

                        # call fortran backend to calculate derivatives of constraint functions
                        self.backend_fortran.gfcn_ddot(g, g_dot2, g_dot1, g_ddot,
                                                       self.ts[i:i + 1],
                                                       x, x_dot2, x_dot1, x_ddot,
                                                       p, p_dot, p_dot, p_ddot,
                                                       u, u_dot2, u_dot1, u_ddot)

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

    def ineqc_dsdp(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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
        ds   = None
        dp   = None
        dsdp = None

        if self.NG > 0:

            # allocate memory
            c     = np.zeros((self.NCG,))
            ds    = np.zeros((self.NCG, self.NS))
            dp    = np.zeros((self.NCG, self.NP))
            dsdp  = np.zeros((self.NCG, self.NS, self.NP))

            x      = np.zeros((self.NX,))
            x_dot1 = np.zeros(self.NX, self.NX)
            x_dot2 = np.zeros(self.NX, self.NP)
            x_ddot = np.zeros(x_dot1.shape + (self.NP,))
            g      = np.zeros((self.NG,))
            g_dot1 = np.zeros((self.NG, self.NX))
            g_dot2 = np.zeros((self.NG, self.NP))
            g_ddot = np.zeros(g_dot1.shape + (self.NP,))
            p_dot1 = np.zeros((self.NP, self.NX))
            p_dot2 = np.eye(self.NP)
            p_ddot = np.zeros(p_dot1.shape + (self.NP,))
            u      = np.zeros((self.NU,))
            u_dot  = np.zeros((self.NU, self.NX))
            u_ddot = np.zeros(u_dot.shape + (self.NP,))

            # loop through all time step
            for i in xrange(0, self.NTS):

                # state and controls for this time step
                x      = xs[i, :]
                x_dot1 = np.reshape(xs_dot1[i, :, 0:self.NX], x_dot1.shape)
                x_dot2 = np.reshape(xs_dot2[i, :, :], x_dot2.shape)
                x_ddot = np.reshape(xs_ddot[i, :, 0:self.NX, :], x_ddot.shape)

                for l in xrange(0, self.NU):
                    u[l] = q[i + l * self.NTS]

                # call fortran backend to calculate derivatives of constraint functions
                self.backend_fortran.gfcn_ddot(g, g_dot2, g_dot1, g_ddot,
                                               self.ts[i:i + 1],
                                               x, x_dot2, x_dot1, x_ddot,
                                               p, p_dot2, p_dot1, p_ddot,
                                               u, u_dot, u_dot, u_ddot)

                # store gradient
                for k in xrange(0, self.NG):
                    dsdp[i + k * self.NTS, 0:self.NX:, :] = g_ddot[k, :, :]
                    ds[i + k * self.NTS, 0:self.NX]       = g_dot1[k, :]
                    dp[i + k * self.NTS, :]               = g_dot2[k, :]
                    c[i + k * self.NTS]                   = g[k]

        return c, ds, dp, dsdp

    """
    ===============================================================================
    """

    def ineqc_dsdq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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
        ds   = None
        dq   = None
        dsdq = None

        if self.NG > 0:

            # allocate memory
            c    = np.zeros((self.NCG,))
            ds   = np.zeros((self.NCG, self.NS))
            dq   = np.zeros((self.NCG, self.NP))
            dsdq = np.zeros((self.NCG, self.NS, self.NP))

            t      = np.zeros((1,))
            x      = np.zeros((self.NX,))
            x_dot1 = np.zeros(self.NX, self.NX)
            x_dot2 = np.zeros(self.NX, self.NP)
            x_ddot = np.zeros(x_dot1.shape + (self.NP,))
            g      = np.zeros((self.NG,))
            g_dot1 = np.zeros((self.NG, self.NX))
            g_dot2 = np.zeros((self.NG, self.NP))
            g_ddot = np.zeros(g_dot1.shape + (self.NP,))
            p_dot  = np.zeros((self.NP, self.NX))
            p_ddot = np.zeros(p_dot.shape + (self.NP,))
            u      = np.zeros((self.NU,))
            u_dot1 = np.zeros((self.NU, self.NX))
            u_dot2 = np.zeros((self.NU, self.NU))
            u_ddot = np.zeros(u_dot1.shape + (self.NP,))

            # loop through all time steps
            for i in xrange(0, self.NTS):

                # loop through all time steps including the current one
                for j in xrange(0, i + 1):

                    # state and controls for this time step
                    x      = xs[i, :]
                    x_dot1 = np.reshape(xs_dot1[i, :, 0:self.NX], x_dot1.shape)
                    x_dot2 = np.reshape(xs_dot2[i, :, j], x_dot2.shape)
                    x_ddot = np.reshape(xs_ddot[i, :, 0:self.NX, j], x_ddot.shape)

                    for l in xrange(0, self.NU):
                        u[l] = q[i + l * self.NTS]

                    if i == j:
                        u_dot2 = np.eye(self.NU)
                    else:
                        u_dot2 = np.zeros((self.NU, self.NU))

                    # call fortran backend to calculate derivatives of constraint functions
                    self.backend_fortran.gfcn_ddot(g, g_dot2, g_dot1, g_ddot,
                                                   self.ts[i:i + 1],
                                                   x, x_dot2, x_dot1, x_ddot,
                                                   p, p_dot, p_dot, p_ddot,
                                                   u, u_dot2, u_dot1, u_ddot)

                    # store gradient
                    for k in xrange(0, self.NG):
                        c[i + k * self.NTS]             = g[k]
                        ds[i + k * self.NTS, 0:self.NX] = g_dot1[k, :]

                        for l in xrange(0, self.NU):
                            dsdq[i + k * self.NTS, 0:self.NX, j + l * self.NTS] = g_ddot[k, :, l]
                            dxq[i + k * self.NTS, j + l * self.NTS]             = g_dot2[k, l]


        return c, ds, dq, dsdq

    """
    ===============================================================================
    """

    def ineqc_dpdq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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

        if self.NG > 0:

            # allocate memory
            c    = np.zeros((self.NCG,))
            dp   = np.zeros((self.NCG, self.NP))
            dq   = np.zeros((self.NCG, self.NQ))
            dpdq = np.zeros((self.NCG, self.NP, self.NQ))

            x      = np.zeros((self.NX,))
            x_dot1 = np.zeros((self.NX, self.NP))
            x_dot2 = np.zeros((self.NX, self.NU))
            x_ddot = np.zeros(x_dot1.shape + (self.NU,))
            g      = np.zeros((self.NG,))
            g_dot1 = np.zeros((self.NG, self.NP))
            g_dot2 = np.zeros((self.NG, self.NU))
            g_ddot = np.zeros(g_dot1.shape + (self.NU,))
            p_dot1 = np.eye(self.NP)
            p_dot2 = np.zeros((self.NP, self.NU))
            p_ddot = np.zeros(p_dot1.shape + (self.NU,))
            u      = np.zeros((self.NU,))
            u_dot1 = np.zeros((self.NU, self.NP))
            u_dot2 = np.zeros((self.NU, self.NU))
            u_ddot = np.zeros(u_dot1.shape + (self.NU,))

            # loop through all time step
            for i in xrange(0, self.NTS):

                # loop through all time steps including the current one
                for j in xrange(0, i + 1):

                    # state and controls for this time step
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
                                                   self.ts[i:i + 1],
                                                   x, x_dot2, x_dot1, x_ddot,
                                                   p, p_dot2, p_dot1, p_ddot,
                                                   u, u_dot2, u_dot1, u_ddot)

                    # store gradient
                    for k in xrange(0, self.NG):
                        c[i + k * self.NTS]     = g[k]
                        dp[i + k * self.NTS, :] = g_dot1[k, :]

                        for l in xrange(0, self.NU):
                            dpdq[i + k * self.NTS, :, j + l * self.NTS] = g_ddot[k, :, l]
                            dq[i + k * self.NTS, j + l * self.NTS]      = g_dot2[k, l]


        return c, dp, dq, dpdq

    """
    ===============================================================================
    """

    def eqc(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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

        if self.NH > 0:

            # allocate memory
            c = np.zeros((self.NCH,))

            x = np.zeros((self.NX,))
            g = np.zeros((self.NH,))
            u = np.zeros((self.NU,))

            # loop through all time steps
            for i in xrange(0, self.NTS):

                # state and controls for this time step
                x  = xs[i, :]

                for k in xrange(0, self.NU):
                    u[k] = q[i + k * self.NTS]

                # call fortran backend to calculate constraint functions for every control
                self.backend_fortran.hfcn(g, self.ts[i:i + 1], x, p, u)

                # build constraints
                for k in xrange(0, self.NH):
                    c[i + k * self.NTS] = g[k]

        return c

    """
    ===============================================================================
    """

    def eqc_ds(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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

        if self.NH > 0:

            # allocate memory
            c    = np.zeros((self.NCH,))
            ds   = np.zeros((self.NCH, self.NS))

            x     = np.zeros((self.NX,))
            x_dot = np.zeros((self.NX, self.NX))
            g     = np.zeros((self.NH,))
            g_dot = np.zeros((self.NH, self.NX))
            p_dot = np.zeros((self.NP, self.NX))
            u     = np.zeros((self.NU,))
            u_dot = np.zeros((self.NU, self.NX))

            # loop through all time step
            for i in xrange(0, self.NTS):

                # state and controls for this time step
                x     = xs[i, :]
                x_dot = np.reshape(xs_dot1[i, :, 0:self.NX], x_dot.shape)

                for l in xrange(0, self.NU):
                    u[l] = q[i + l * self.NTS]

                # call fortran backend to calculate derivatives of constraint functions
                self.backend_fortran.hfcn_dot(g, g_dot, self.ts[i:i + 1], x, x_dot, p, p_dot, u, u_dot)

                # store gradient
                for k in xrange(0, self.NH):
                    ds[i + k * self.NTS, 0:self.NX] = g_dot[k, :]
                    c[i + k * self.NTS]             = g[k]

        return c, ds

    """
    ===============================================================================
    """

    def eqc_dp(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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
        c_dp = None

        if self.NH > 0:

            # allocate memory
            c     = np.zeros((self.NCH,))
            c_dp  = np.zeros((self.NCH, self.NP))

            x     = np.zeros((self.NX,))
            x_dot = np.zeros((self.NX, self.NP))
            g     = np.zeros((self.NH,))
            g_dot = np.zeros((self.NH, self.NP))
            p_dot = np.eye(self.NP)
            u     = np.zeros((self.NU,))
            u_dot = np.zeros((self.NU, self.NP))

            # loop through all time steps
            for i in xrange(0, self.NTS):

                # state and controls for this time step
                x     = xs[i, :]
                x_dot = np.reshape(xs_dot1[i, :, :], x_dot.shape)

                for l in xrange(0, self.NU):
                    u[l] = q[i + l * self.NTS]

                # call fortran backend to calculate derivatives of constraint functions
                self.backend_fortran.hfcn_dot(g, g_dot, self.ts[i:i + 1], x, x_dot, p, p_dot, u, u_dot)

                # store gradient
                for k in xrange(0, self.NH):
                    c[i + k * self.NTS]       = g[k]
                    c_dp[i + k * self.NTS, :] = g_dot[k, :]

        return c, c_dp

    """
    ===============================================================================
    """

    def eqc_dq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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

        if self.NH > 0:

            # allocate memory
            c     = np.zeros((self.NCH,))
            dq    = np.zeros((self.NCH, self.NQ))

            x     = np.zeros((self.NX,))
            x_dot = np.zeros((self.NX, self.NU))
            g     = np.zeros((self.NH,))
            g_dot = np.zeros((self.NH, self.NU))
            p_dot = np.zeros((self.NP, self.NU))
            u     = np.zeros((self.NU,))
            u_dot = np.zeros((self.NU, self.NU))

            # loop through all time steps
            for i in xrange(0, self.NTS):

                # loop through all previous controls including the current one
                for j in xrange(0, i + 1):

                    # state and controls for this time step
                    x     = xs[i, :]
                    x_dot = np.reshape(xs_dot1[i, :, j], x_dot.shape)

                    for k in xrange(0, self.NU):
                        u[k] = q[i + k * self.NTS]

                    if i == j:
                        u_dot = np.eye(self.NU)
                    else:
                        u_dot = np.zeros((self.NU, self.NU))

                    # call fortran backend to calculate derivatives of constraint functions
                    self.backend_fortran.hfcn_dot(g, g_dot, self.ts[i:i + 1], x, x_dot, p, p_dot, u, u_dot)

                    # store gradient
                    for k in xrange(0, self.NH):
                        c[i + k * self.NTS] = g[k]

                        for l in xrange(0, self.NU):
                            dq[i + k * self.NTS, j + l * self.NTS] = g_dot[k, l]

        return c, dq

    """
    ===============================================================================
    """

    def eqc_dsds(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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
        ds1  = None
        ds2  = None
        dsds = None

        if self.NH > 0:

            # allocate memory
            c    = np.zeros((self.NCH,))
            ds1  = np.zeros((self.NCH, self.NS))
            ds2  = np.zeros((self.NCH, self.NS))
            dsds = np.zeros((self.NCH, self.NS, self.NS))

            x      = np.zeros((self.NX,))
            x_dot1 = np.zeros((self.NX, self.NX))
            x_dot2 = np.zeros((self.NX, self.NX))
            x_ddot = np.zeros(x_dot1.shape + (self.NX,))
            g      = np.zeros((self.NH,))
            g_dot1 = np.zeros((self.NH, self.NX))
            g_dot2 = np.zeros((self.NH, self.NX))
            g_ddot = np.zeros(g_dot1.shape + (self.NX,))
            p_dot  = np.zeros((self.NP, self.NX))
            p_ddot = np.zeros(p_dot.shape + (self.NX,))
            u      = np.zeros((self.NU,))
            u_dot  = np.zeros((self.NU, self.NX))
            u_ddot = np.zeros(u_dot.shape + (self.NX,))

            # loop through all time step
            for i in xrange(0, self.NTS):

                # state and controls for this time step
                x      = xs[i, :]
                x_dot1 = np.reshape(xs_dot1[i, :, 0:self.NX], x_dot1.shape)
                x_dot2 = np.reshape(xs_dot2[i, :, 0:self.NX], x_dot2.shape)
                x_ddot = np.reshape(xs_ddot[i, :, 0:self.NX, 0:self.NX], x_ddot.shape)

                for l in xrange(0, self.NU):
                    u[l] = q[i + l * self.NTS]

                # call fortran backend to calculate derivatives of constraint functions
                self.backend_fortran.hfcn_ddot(g, g_dot2, g_dot1, g_ddot,
                                               self.ts[i:i + 1],
                                               x, x_dot2, x_dot1, x_ddot,
                                               p, p_dot, p_dot, p_ddot,
                                               u, u_dot, u_dot, u_ddot)

                # store gradient
                for k in xrange(0, self.NH):
                    dsds[i + k * self.NTS, 0:self.NX, 0:self.NX] = g_ddot[k, :, :]
                    ds1[i + k * self.NTS, 0:self.NX]             = g_dot1[k, :]
                    ds2[i + k * self.NTS, 0:self.NX]             = g_dot2[k, :]
                    c[i + k * self.NTS]                          = g[k]

        return c, ds1, ds2, dsds

    """
    ===============================================================================
    """

    def eqc_dpdp(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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
        dp1  = None
        dp2  = None
        dpdp = None

        if self.NH > 0:

            # allocate memory
            c      = np.zeros((self.NCH,))
            dp1    = np.zeros((self.NCH, self.NP))
            dp2    = np.zeros((self.NCH, self.NP))
            dpdp   = np.zeros((self.NCH, self.NP, self.NP))

            t      = np.zeros((1,))
            x      = np.zeros((self.NX,))
            x_dot1 = np.zeros((self.NX, self.NP))
            x_dot2 = np.zeros((self.NX, self.NP))
            x_ddot = np.zeros(x_dot1.shape + (self.NP,))
            g      = np.zeros((self.NH,))
            g_dot1 = np.zeros((self.NH, self.NP))
            g_dot2 = np.zeros((self.NH, self.NP))
            g_ddot = np.zeros(g_dot1.shape + (self.NP,))
            p_dot  = np.eye(self.NP)
            p_ddot = np.zeros(p_dot.shape + (self.NP,))
            u      = np.zeros((self.NU,))
            u_dot  = np.zeros((self.NU, self.NP))
            u_ddot = np.zeros(u_dot.shape + (self.NP,))

            # loop through all time step
            for i in xrange(0, self.NTS):

                # state and controls for this time step
                x      = xs[i, :]
                x_dot1 = np.reshape(xs_dot1[i, :, :], x_dot1.shape)
                x_dot2 = np.reshape(xs_dot2[i, :, :], x_dot2.shape)
                x_ddot = np.reshape(xs_ddot[i, :, :, :], x_ddot.shape)

                for l in xrange(0, self.NU):
                    u[l] = q[i + l * self.NTS]

                # call fortran backend to calculate derivatives of constraint functions
                self.backend_fortran.hfcn_ddot(g, g_dot2, g_dot1, g_ddot,
                                               self.ts[i:i + 1],
                                               x, x_dot2, x_dot1, x_ddot,
                                               p, p_dot, p_dot, p_ddot,
                                               u, u_dot, u_dot, u_ddot)

                # store gradient
                for k in xrange(0, self.NH):
                    dpdp[i + k * self.NTS, :, :] = g_ddot[k, :, :]
                    dp1[i + k * self.NTS, :]     = g_dot1[k, :]
                    dp2[i + k * self.NTS, :]     = g_dot2[k, :]
                    c[i + k * self.NTS]          = g[k]

        return c, dp1, dp2, dpdp

    """
    ===============================================================================
    """

    def eqc_dqdq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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
        dq1  = None
        dq2  = None
        dqdq = None

        if self.NH > 0:

            # allocate memory
            c      = np.zeros((self.NCH,))
            dq1    = np.zeros((self.NCH, self.NQ))
            dq2    = np.zeros((self.NCH, self.NQ))
            dqdq   = np.zeros((self.NCH, self.NQ, self.NQ))

            x      = np.zeros((self.NX,))
            x_dot1 = np.zeros((self.NX, self.NU))
            x_dot2 = np.zeros((self.NX, self.NU))
            x_ddot = np.zeros(x_dot1.shape + (self.NU,))
            g      = np.zeros((self.NH,))
            g_dot1 = np.zeros((self.NH, self.NU))
            g_dot2 = np.zeros((self.NH, self.NU))
            g_ddot = np.zeros(g_dot1.shape + (self.NU,))
            p_dot  = np.zeros((self.NP, self.NU))
            p_ddot = np.zeros(p_dot.shape + (self.NU,))
            u      = np.zeros((self.NU,))
            u_dot1 = np.zeros((self.NU, self.NU))
            u_dot2 = np.zeros((self.NU, self.NU))
            u_ddot = np.zeros(u_dot1.shape + (self.NU,))

            # loop through all time steps three times
            for i in xrange(0, self.NTS):

                # loop through all previous controls including the current one
                for j in xrange(0, i + 1):

                    # loop through all previous controls including the current one
                    for m in xrange(0, i + 1):

                        # state and controls for this time step
                        x      = xs[i, :]
                        x_dot1 = np.reshape(xs_dot1[i, :, j], x_dot1.shape)
                        x_dot2 = np.reshape(xs_dot2[i, :, m], x_dot2.shape)
                        x_ddot = np.reshape(xs_ddot[i, :, j, m], x_ddot.shape)

                        for k in xrange(0, self.NU):
                            u[k] = q[i + k * self.NTS]

                        if i == j:
                            u_dot1 = np.eye(self.NU)
                        else:
                            u_dot1 = np.zeros((self.NU, self.NU))

                        if i == m:
                            u_dot2 = np.eye(self.NU)
                        else:
                            u_dot2 = np.zeros((self.NU, self.NU))

                        # call fortran backend to calculate derivatives of constraint functions
                        self.backend_fortran.hfcn_ddot(g, g_dot2, g_dot1, g_ddot,
                                                       self.ts[i:i + 1],
                                                       x, x_dot2, x_dot1, x_ddot,
                                                       p, p_dot, p_dot, p_ddot,
                                                       u, u_dot2, u_dot1, u_ddot)

                        # store gradient
                        for k in xrange(0, self.NH):
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

    def eqc_dsdp(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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
        ds   = None
        dp   = None
        dsdp = None

        if self.NH > 0:

            # allocate memory
            c     = np.zeros((self.NCH,))
            ds    = np.zeros((self.NCH, self.NS))
            dp    = np.zeros((self.NCH, self.NP))
            dsdp  = np.zeros((self.NCH, self.NS, self.NP))

            x      = np.zeros((self.NX,))
            x_dot1 = np.zeros(self.NX, self.NX)
            x_dot2 = np.zeros(self.NX, self.NP)
            x_ddot = np.zeros(x_dot1.shape + (self.NP,))
            g      = np.zeros((self.NH,))
            g_dot1 = np.zeros((self.NH, self.NX))
            g_dot2 = np.zeros((self.NH, self.NP))
            g_ddot = np.zeros(g_dot1.shape + (self.NP,))
            p_dot1 = np.zeros((self.NP, self.NX))
            p_dot2 = np.eye(self.NP)
            p_ddot = np.zeros(p_dot1.shape + (self.NP,))
            u      = np.zeros((self.NU,))
            u_dot  = np.zeros((self.NU, self.NX))
            u_ddot = np.zeros(u_dot.shape + (self.NP,))

            # loop through all time step
            for i in xrange(0, self.NTS):

                # state and controls for this time step
                x      = xs[i, :]
                x_dot1 = np.reshape(xs_dot1[i, :, 0:self.NX], x_dot1.shape)
                x_dot2 = np.reshape(xs_dot2[i, :, :], x_dot2.shape)
                x_ddot = np.reshape(xs_ddot[i, :, 0:self.NX, :], x_ddot.shape)

                for l in xrange(0, self.NU):
                    u[l] = q[i + l * self.NTS]

                # call fortran backend to calculate derivatives of constraint functions
                self.backend_fortran.hfcn_ddot(g, g_dot2, g_dot1, g_ddot,
                                               self.ts[i:i + 1],
                                               x, x_dot2, x_dot1, x_ddot,
                                               p, p_dot2, p_dot1, p_ddot,
                                               u, u_dot, u_dot, u_ddot)

                # store gradient
                for k in xrange(0, self.NH):
                    dsdp[i + k * self.NTS, 0:self.NX, :] = g_ddot[k, :, :]
                    ds[i + k * self.NTS, 0:self.NX]      = g_dot1[k, :]
                    dxp[i + k * self.NTS, :]             = g_dot2[k, :]
                    c[i + k * self.NTS]                  = g[k]

        return c, ds, dp, dsdp

    """
    ===============================================================================
    """

    def eqc_dsdq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        c     = None
        ds   = None
        dq    = None
        dsdq = None

        if self.NH > 0:

            # allocate memory
            c     = np.zeros((self.NCH,))
            ds    = np.zeros((self.NCH, self.NS))
            dq    = np.zeros((self.NCH, self.NP))
            dsdq  = np.zeros((self.NCH, self.NS, self.NP))

            t      = np.zeros((1,))
            x      = np.zeros((self.NX,))
            x_dot1 = np.zeros(self.NX, self.NX)
            x_dot2 = np.zeros(self.NX, self.NP)
            x_ddot = np.zeros(x_dot1.shape + (self.NP,))
            g      = np.zeros((self.NH,))
            g_dot1 = np.zeros((self.NH, self.NX))
            g_dot2 = np.zeros((self.NH, self.NP))
            g_ddot = np.zeros(g_dot1.shape + (self.NP,))
            p_dot  = np.zeros((self.NP, self.NX))
            p_ddot = np.zeros(p_dot.shape + (self.NP,))
            u      = np.zeros((self.NU,))
            u_dot1 = np.zeros((self.NU, self.NX))
            u_dot2 = np.zeros((self.NU, self.NU))
            u_ddot = np.zeros(u_dot1.shape + (self.NP,))

            # loop through all time steps
            for i in xrange(0, self.NTS):

                # loop through all time steps including the current one
                for j in xrange(0, i + 1):

                    # state and controls for this time step
                    x      = xs[i, :]
                    x_dot1 = np.reshape(xs_dot1[i, :, 0:self.NX], x_dot1.shape)
                    x_dot2 = np.reshape(xs_dot2[i, :, j], x_dot2.shape)
                    x_ddot = np.reshape(xs_ddot[i, :, 0:self.NX, j], x_ddot.shape)

                    for l in xrange(0, self.NU):
                        u[l] = q[i + l * self.NTS]

                    if i == j:
                        u_dot2 = np.eye(self.NU)
                    else:
                        u_dot2 = np.zeros((self.NU, self.NU))

                    # call fortran backend to calculate derivatives of constraint functions
                    self.backend_fortran.hfcn_ddot(g, g_dot2, g_dot1, g_ddot,
                                                   self.ts[i:i + 1],
                                                   x, x_dot2, x_dot1, x_ddot,
                                                   p, p_dot, p_dot, p_ddot,
                                                   u, u_dot2, u_dot1, u_ddot)

                    # store gradient
                    for k in xrange(0, self.NH):
                        c[i + k * self.NTS]             = g[k]
                        ds[i + k * self.NTS, 0:self.NX] = g_dot1[k, :]

                        for l in xrange(0, self.NU):
                            dsdq[i + k * self.NTS, 0:self.NX, j + l * self.NTS] = g_ddot[k, :, l]
                            dxq[i + k * self.NTS, j + l * self.NTS]             = g_dot2[k, l]


        return c, ds, dq, dsdq

    """
    ===============================================================================
    """

    def eqc_dpdq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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

        if self.NH > 0:

            # allocate memory
            c    = np.zeros((self.NCH,))
            dp   = np.zeros((self.NCH, self.NP))
            dq   = np.zeros((self.NCH, self.NQ))
            dpdq = np.zeros((self.NCH, self.NP, self.NQ))

            x      = np.zeros((self.NX,))
            x_dot1 = np.zeros((self.NX, self.NP))
            x_dot2 = np.zeros((self.NX, self.NU))
            x_ddot = np.zeros(x_dot1.shape + (self.NU,))
            g      = np.zeros((self.NH,))
            g_dot1 = np.zeros((self.NH, self.NP))
            g_dot2 = np.zeros((self.NH, self.NU))
            g_ddot = np.zeros(g_dot1.shape + (self.NU,))
            p_dot1 = np.eye(self.NP)
            p_dot2 = np.zeros((self.NP, self.NU))
            p_ddot = np.zeros(p_dot1.shape + (self.NU,))
            u      = np.zeros((self.NU,))
            u_dot1 = np.zeros((self.NU, self.NP))
            u_dot2 = np.zeros((self.NU, self.NU))
            u_ddot = np.zeros(u_dot1.shape + (self.NU,))

            # loop through all time step
            for i in xrange(0, self.NTS):

                # loop through all time steps including the current one
                for j in xrange(0, i + 1):

                    # state and controls for this time step
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
                    self.backend_fortran.hfcn_ddot(g, g_dot2, g_dot1, g_ddot,
                                                   self.ts[i:i + 1],
                                                   x, x_dot2, x_dot1, x_ddot,
                                                   p, p_dot2, p_dot1, p_ddot,
                                                   u, u_dot2, u_dot1, u_ddot)

                    # store gradient
                    for k in xrange(0, self.NH):
                        c[i + k * self.NTS]     = g[k]
                        dp[i + k * self.NTS, :] = g_dot1[k, :]

                        for l in xrange(0, self.NU):
                            dpdq[i + k * self.NTS, :, j + l * self.NTS] = g_ddot[k, :, l]
                            dq[i + k * self.NTS, j + l * self.NTS]      = g_dot2[k, l]


        return c, dp, dq, dpdq

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

        # evaluate matching conditions
        mc = xs[-1, :] - self.flat2array_s(s)[-1, :]

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

        # evaluate matching conditions
        mc                 = xs[-1, :] - self.flat2array_s(s)[-1, :]
        mc_ds              = xs_dot1[-1, :, :]
        mc_ds[:, self.NX:] = -np.eye(self.NX)

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

        # evaluate matching conditions
        mc    = xs[-1, :] - self.flat2array_s(s)[-1, :]
        mc_dp = xs_dot1[-1, :, :]

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

        # evaluate matching conditions
        mc    = xs[-1, :] - self.flat2array_s(s)[-1, :]
        mc_dq = xs_dot1[-1, :, :]

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

        # evaluate matching conditions
        mc                  = xs[-1, :] - self.flat2array_s(s)[-1, :]
        mc_ds1              = xs_dot1[-1, :, :]
        mc_ds1[:, self.NX:] = -np.eye(self.NX)
        mc_ds2              = xs_dot2[-1, :, :]
        mc_ds2[:, self.NX:] = -np.eye(self.NX)
        mc_dsds             = xs_ddot[-1, :, :, :]

        return mc, mc_ds1, mc_ds2, mc_dsds

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

        # evaluate matching conditions
        mc      = xs[-1, :] - self.flat2array_s(s)[-1, :]
        mc_dp1  = xs_dot1[-1, :, :]
        mc_dp2  = xs_dot2[-1, :, :]
        mc_dpdp = xs_ddot[-1, :, :, :]

        return mc, mc_dp1, mc_dp2, mc_dpdp


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

        # evaluate matching conditions
        mc      = xs[-1, :] - self.flat2array_s(s)[-1, :]
        mc_dq1  = xs_dot1[-1, :, :]
        mc_dq2  = xs_dot2[-1, :, :]
        mc_dqdq = xs_ddot[-1, :, :, :]

        return mc, mc_dq1, mc_dq2, mc_dqdq

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

        # evaluate matching conditions
        mc                 = xs[-1, :] - self.flat2array_s(s)[-1, :]
        mc_ds              = xs_dot1[-1, :, :]
        mc_ds[:, self.NX:] = -np.eye(self.NX)
        mc_dp              = xs_dot2[-1, :, :]
        mc_dsdp            = xs_ddot[-1, :, :, :]

        return mc, mc_ds, mc_dp, mc_dsdp

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

        # evaluate matching conditions
        mc                 = xs[-1, :] - self.flat2array_s(s)[-1, :]
        mc_ds              = xs_dot1[-1, :, :]
        mc_ds[:, self.NX:] = -np.eye(self.NX)
        mc_dq              = xs_dot2[-1, :, :]
        mc_dsdq            = xs_ddot[-1, :, :, :]

        return mc, mc_ds, mc_dq, mc_dsdq

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

        # evaluate matching conditions
        mc      = xs[-1, :] - self.flat2array_s(s)[-1, :]
        mc_dp   = xs_dot1[-1, :, :]
        mc_dq   = xs_dot2[-1, :, :]
        mc_dpdq = xs_ddot[-1, :, :, :]

        return mc, mc_dp, mc_dq, mc_dpdq

    """
    ===============================================================================
    """

    def bc(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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

    """
    ===============================================================================
    """

    def bc_ds(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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
        bc_ds = np.zeros((self.NQ * 2 + self.NS * 2, self.NS))

        # set derivatives
        bc_ds[self.NQ:self.NQ + self.NS, :] = -np.eye(self.NS)
        bc_ds[self.NQ * 2 + self.NS:, :]    = np.eye(self.NS)

        return bc_ds

    """
    ===============================================================================
    """

    def bc_dp(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

        """

        description ...

        input:
            ...

        output:
            ...

        TODO:
            ...

        """

        bc_dp = np.zeros((self.NQ * 2 + self.NS * 2, self.NP))

        return bc_dp

    """
    ===============================================================================
    """

    def bc_dq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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
        bc_dq = np.zeros((self.NQ * 2 + self.NS * 2, self.NQ))

        # set derivatives
        bc_dq[0:self.NQ, :]                               = -np.eye(self.NQ)
        bc_dq[self.NQ + self.NS:self.NQ * 2 + self.NS, :] = np.eye(self.NQ)

        return bc_dq

    """
    ===============================================================================
    """

    def bc_dsds(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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
        bc_dsds = np.zeros((self.NQ * 2 + self.NS * 2, self.NS, self.NS))

        return bc_dsds

    """
    ===============================================================================
    """

    def bc_dpdp(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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
        bc_dpdp = np.zeros((self.NQ * 2 + self.NS * 2, self.NP, self.NP))

        return bc_dpdp

    """
    ===============================================================================
    """

    def bc_dqdq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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
        bc_dqdq = np.zeros((self.NQ * 2 + self.NS * 2, self.NQ, self.NQ))

        return bc_dqdq

    """
    ===============================================================================
    """

    def bc_dsdp(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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
        bc_dsdp = np.zeros((self.NQ * 2 + self.NS * 2, self.NS, self.NP))

        return bc_dsdp

    """
    ===============================================================================
    """

    def bc_dsdq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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
        bc_dsdq = np.zeros((self.NQ * 2 + self.NS * 2, self.NS, self.NQ))

        return bc_dsdq

    """
    ===============================================================================
    """

    def bc_dpdq(self, xs, xs_dot1, xs_dot2, xs_ddot, p, q, s):

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
        bc_dpdq = np.zeros((self.NQ * 2 + self.NS * 2, self.NP, self.NQ))

        return bc_dpdq

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

        return self.sign * xs[-1, -1], self.sign * xs_dot1[-1, -1, :]

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

        return self.sign * xs[-1, -1], self.sign * xs_dot1[-1, -1, :]

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

        return self.sign * xs[-1, -1], self.sign * xs_dot1[-1, -1, :]

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

        return self.sign * xs[-1, -1], self.sign * xs_dot1[-1, -1, :]

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

        return self.sign * xs[-1, -1], self.sign * xs_dot1[-1, -1, :]

"""
===============================================================================
"""
