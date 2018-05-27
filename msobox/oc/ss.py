#!/usr/bin/env python
# -*- coding: utf-8 -*-

# system imports
import sys
import datetime
import json
import numpy as np
import matplotlib.pyplot as plt

# project imports
from msobox.mf.tapenade import Differentiator
from msobox.mf.fortran import BackendFortran
from msobox.ind.rk4classic import RK4Classic
from msobox.ind.explicit_euler import ExplicitEuler
from .utilities import rq

# imports for nlp solver
import snopt
import scipy.optimize as opt

# =============================================================================

class SS(object):

    """
    Provide an interface to solve an optimal control problem with ODE dynamics
    via direct single shooting.
    """

    # =========================================================================

    def q2array(self):

        """
        Convert the controls from an one-dimensional array to a
        3-dimensional form.

        Args:
            None

        Returns:
            q: array-like (self.NU, self.NTS, self.NQI)
            converted controls

        Raises:
            None
        """

        if self.q.shape != (self.NU, self.NTS, self.NQI):

            # allocate memory
            q_array = np.zeros((self.NU, self.NTS, self.NQI))

            # convert controls from one-dimensional to 3-dimensional
            for i in range(0, self.NU):
                q_array[i, :, 0] = self.q[i * self.NTS:(i + 1) * self.NTS]

            return q_array

        else:

            return self.q

    # =========================================================================

    def s2array(self):

        """
        Convert the controls from an one-dimensional array to a
        2-dimensional form.

        Args:
            None

        Returns:
            s: array-like (2, self.NX)
            converted shooting variables

        Raises:
            None
        """

        if self.s.shape != (2, self.NX):

            # allocate memory
            s_array = np.zeros((2, self.NX))

            # convert shooting variables from one-dimensional to 3-dimensional
            for i in range(0, 2):
                s_array[i, :] = self.s[i * self.NX:(i + 1) * self.NX]

            return s_array

        else:
            return self.s

    # =========================================================================

    def approximate_s(self):

        """
        Calculate an approximation for the initial values of the shooting
        variables based on linear interpolation.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        # allocate memory
        self.s = np.zeros((2 * self.NX,))

        # approximate shooting variables by linear interpolation if possible
        for i in range(0, self.NX):

            # set initial shooting variables to x0 if possible
            if self.x0[i] is not None:
                self.s[i] = self.x0[i]

            # set end shooting variables to xend if possible
            if self.xend[i] is not None:
                self.s[self.NX + i] = self.xend[i]

    # =========================================================================

    def prepare(self):

        """
        Prepare the optimal control problem (e.g allocate memory,
        provide derivatives, etc.)

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        # set properties
        self.NCG = self.NG * self.NTS
        self.NCH = self.NH * self.NTS
        self.NC = self.NCG + self.NCH
        self.NQI = 1
        self.NQ = self.NU * self.NTS * self.NQI
        self.NS = 2 * self.NX
        self.NMC = self.NX

        # assert right dimensions of data
        assert self.ts.size == self.NTS
        assert self.p.size == self.NP
        assert self.q.size == self.NQ
        assert self.s.size == self.NS
        assert len(self.x0) == self.NX
        assert len(self.xend) == self.NX

        # allocate memory for states and derivatives
        self.xs = np.zeros((self.NTS, self.NX))
        self.xs_ds = np.zeros((self.NTS, self.NX, self.NS))
        self.xs_dp = np.zeros((self.NTS, self.NX, self.NP))
        self.xs_dq = np.zeros((self.NTS, self.NX, self.NQ))
        self.xs_dsds = np.zeros((self.NTS, self.NX, self.NS, self.NS))
        self.xs_dpdp = np.zeros((self.NTS, self.NX, self.NP, self.NP))
        self.xs_dqdq = np.zeros((self.NTS, self.NX, self.NQ, self.NQ))
        self.xs_dsdp = np.zeros((self.NTS, self.NX, self.NS, self.NP))
        self.xs_dsdq = np.zeros((self.NTS, self.NX, self.NS, self.NQ))
        self.xs_dpdq = np.zeros((self.NTS, self.NX, self.NP, self.NQ))

        # allocate memory for objective function
        self.obj = np.zeros((1,))
        self.obj_ds = np.zeros((1, self.NS))
        self.obj_dp = np.zeros((1, self.NP))
        self.obj_dq = np.zeros((1, self.NQ))
        self.obj_dsds = np.zeros((1, self.NS, self.NS))
        self.obj_dpdp = np.zeros((1, self.NP, self.NP))
        self.obj_dqdq = np.zeros((1, self.NQ, self.NQ))
        self.obj_dsdp = np.zeros((1, self.NS, self.NP))
        self.obj_dsdq = np.zeros((1, self.NS, self.NQ))
        self.obj_dpdq = np.zeros((1, self.NP, self.NQ))

        # allocate memory for matching conditions
        self.mc = np.zeros((self.NMC,))
        self.mc_ds = np.zeros((self.NMC, self.NS))
        self.mc_dp = np.zeros((self.NMC, self.NP))
        self.mc_dq = np.zeros((self.NMC, self.NQ))
        self.mc_dsds = np.zeros((self.NMC, self.NS, self.NS))
        self.mc_dpdp = np.zeros((self.NMC, self.NP, self.NP))
        self.mc_dqdq = np.zeros((self.NMC, self.NQ, self.NQ))
        self.mc_dsdp = np.zeros((self.NMC, self.NS, self.NP))
        self.mc_dsdq = np.zeros((self.NMC, self.NS, self.NQ))
        self.mc_dpdq = np.zeros((self.NMC, self.NP, self.NQ))

        # allocate memory for box constraints
        self.bcq = np.zeros((2 * self.NQ,))
        self.bcq_ds = np.zeros((2 * self.NQ, self.NS))
        self.bcq_dp = np.zeros((2 * self.NQ, self.NP))
        self.bcq_dq = np.zeros((2 * self.NQ, self.NQ))
        self.bcq_dsds = np.zeros((2 * self.NQ, self.NS, self.NS))
        self.bcq_dpdp = np.zeros((2 * self.NQ, self.NP, self.NP))
        self.bcq_dqdq = np.zeros((2 * self.NQ, self.NQ, self.NQ))
        self.bcq_dsdp = np.zeros((2 * self.NQ, self.NS, self.NP))
        self.bcq_dsdq = np.zeros((2 * self.NQ, self.NS, self.NQ))
        self.bcq_dpdq = np.zeros((2 * self.NQ, self.NP, self.NQ))

        # allocate memory for constraints on shooting variables
        self.bcs = np.zeros((self.NS,))
        self.bcs_ds = np.zeros((self.NS, self.NS))
        self.bcs_dp = np.zeros((self.NS, self.NP))
        self.bcs_dq = np.zeros((self.NS, self.NQ))
        self.bcs_dsds = np.zeros((self.NS, self.NS, self.NS))
        self.bcs_dpdp = np.zeros((self.NS, self.NP, self.NP))
        self.bcs_dqdq = np.zeros((self.NS, self.NQ, self.NQ))
        self.bcs_dsdp = np.zeros((self.NS, self.NS, self.NP))
        self.bcs_dsdq = np.zeros((self.NS, self.NS, self.NQ))
        self.bcs_dpdq = np.zeros((self.NS, self.NP, self.NQ))

        # allocate memory for constraints and derivatives
        if self.NG > 0:
            self.ineqc = np.zeros((self.NCG,))
            self.ineqc_ds = np.zeros((self.NCG, self.NS))
            self.ineqc_dp = np.zeros((self.NCG, self.NP))
            self.ineqc_dq = np.zeros((self.NCG, self.NQ))
            self.ineqc_dsds = np.zeros((self.NCG, self.NS, self.NS))
            self.ineqc_dpdp = np.zeros((self.NCG, self.NP, self.NP))
            self.ineqc_dqdq = np.zeros((self.NCG, self.NQ, self.NQ))
            self.ineqc_dsdp = np.zeros((self.NCG, self.NS, self.NP))
            self.ineqc_dsdq = np.zeros((self.NCG, self.NS, self.NQ))
            self.ineqc_dpdq = np.zeros((self.NCG, self.NP, self.NQ))

        if self.NH > 0:
            self.eqc = np.zeros((self.NCH,))
            self.eqc_ds = np.zeros((self.NCH, self.NS))
            self.eqc_dp = np.zeros((self.NCH, self.NP))
            self.eqc_dq = np.zeros((self.NCH, self.NQ))
            self.eqc_dsds = np.zeros((self.NCH, self.NS, self.NS))
            self.eqc_dpdp = np.zeros((self.NCH, self.NP, self.NP))
            self.eqc_dqdq = np.zeros((self.NCH, self.NQ, self.NQ))
            self.eqc_dsdp = np.zeros((self.NCH, self.NS, self.NP))
            self.eqc_dsdq = np.zeros((self.NCH, self.NS, self.NQ))
            self.eqc_dpdq = np.zeros((self.NCH, self.NP, self.NQ))

        # set whether to minimize or maximize
        if self.minormax == "min":
            self.sign = 1

        elif self.minormax == "max":
            self.sign = -1

        else:
            print("No valid input for minormax.")
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

        elif self.integrator == "expliciteuler":
            self.ind = ExplicitEuler(self.backend_fortran)

        else:
            print("Chosen integrator is not available.")
            raise NotImplementedError

    # =========================================================================

    def plot(self, title="solution of ocp"):

        """
        Plot the dynamics of the optimal control problem.

        CAUTION: Needs integrate() to be called beforehand.

        Args:
            title: string
                   title of the figure

        Returns:
            None

        Raises:
            None
        """

        # set new figure
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        # set colors and convert q
        colors = ["blue", "red", "green", "yellow"]
        q = self.q2array()

        # plot q and xs
        for i in range(0, self.NU):
            ax.plot(self.ts, q[i, :, 0], color=colors[i], linewidth=2,
                     linestyle="dashed", label="u_" + str(i))

        for i in range(0, self.NX):
            ax.plot(self.ts, self.xs[:, i], color=colors[i], linewidth=2,
                     linestyle="solid", label="x_" + str(i))

        # set layout, save and show figure
        ax.set_ylabel("t")
        ax.set_xlabel("")
        ax.set_title(title)
        ax.grid(True)
        ax.legend(loc="best")
        fig.savefig(self.path + "/output/" + datetime.datetime.now().
            strftime("%Y-%m-%d-%H-%M-%S") + "-plot.png", bbox_inches="tight")
        fig.savefig(self.path + "/output/" + datetime.datetime.now().
            strftime("%Y-%m-%d-%H-%M-%S") + "-plot.pdf", bbox_inches="tight")
        fig.show()

    # =========================================================================

    def integrate(self):

        """
        Integrate the dynamics of the optimal control problem.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        # convert controls and shooting variables
        p = self.p
        q = self.q2array()
        s = self.s2array()

        # set initial conditions
        self.xs[0, :] = s[0, :]

        # integrate
        for i in range(0, self.NTS - 1):

            # set initial conditions
            x0 = self.xs[i, :]

            # set time steps for this interval
            tsi = np.linspace(self.ts[i], self.ts[i + 1], self.NTSI)

            # set constant controls for this interval
            q_interval = q[:, i, 0]

            self.xs[i + 1, :] = self.ind.zo_forward(tsi, x0, self.p, q_interval)

    # =========================================================================

    def integrate_ds(self):

        """
        Integrate the dynamics of the optimal control problem and provide
        the first order sensitivities w.r.t. the shooting variables via IND.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        # convert controls and shooting variables
        p = self.p
        q = self.q2array()
        s = self.s2array()

        # set up directions and initial conditions
        self.xs[0, :] = s[0, :]
        self.xs_ds[0, :, :self.NX] = np.eye(self.NX)
        p_dot = np.zeros((self.NP, self.NX))
        q_dot = np.zeros((self.NU, self.NX))

        # integrate
        for i in range(0, self.NTS - 1):

            # set initial conditions
            x0 = self.xs[i, :]
            x0_dot = self.xs_ds[i, :, :self.NX]

            # set time steps for this interval
            tsi = np.linspace(self.ts[i], self.ts[i + 1], self.NTSI)

            # set constant controls for this interval
            q_interval = q[:, i, 0]

            # integrate
            self.xs[i + 1, :], self.xs_ds[i + 1, :, :self.NX] = \
                self.ind.fo_forward(tsi, x0, x0_dot,
                                    p, p_dot, q_interval, q_dot)

    # =========================================================================

    def integrate_dp(self):

        """
        Integrate the dynamics of the optimal control problem and provide
        the first order sensitivities w.r.t. the parameters via IND.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        # convert controls and shooting variables
        p = self.p
        q = self.q2array()
        s = self.s2array()

        # set up directions and initial conditions
        self.xs[0, :] = s[0, :]
        p_dot = np.eye(self.NP)
        q_dot = np.zeros((self.NU, self.NP))

        # integrate
        for i in range(0, self.NTS - 1):

            # set initial conditions
            x0 = self.xs[i, :]
            x0_dot = self.xs_dp[i, :, :]

            # set time steps for this interval
            tsi = np.linspace(self.ts[i], self.ts[i + 1], self.NTSI)

            # set constant controls for this interval
            q_interval = q[:, i, 0]

            self.xs[i + 1, :], self.xs_dp[i + 1, :, :] = \
                self.ind.fo_forward(tsi, x0, x0_dot,
                                    p, p_dot, q_interval, q_dot)

    # =========================================================================

    def integrate_dq(self):

        """
        Integrate the dynamics of the optimal control problem and provide
        the first order sensitivities w.r.t. the controls via IND.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        # convert controls and shooting variables
        p = self.p
        q = self.q2array()
        s = self.s2array()

        # set initial conditions
        self.xs[0, :] = s[0, :]

        # integrate
        for i in range(0, self.NTS - 1):

            # set time steps for this interval
            tsi = np.linspace(self.ts[i], self.ts[i + 1], self.NTSI)

            # set constant controls for this interval
            q_interval = q[:, i, 0]

            # set initial conditions and directions
            x0 = self.xs[i, :]
            x0_dot = self.xs_dq[i, :, :]
            p_dot = np.zeros((self.NP, self.NQ))
            q_dot = np.zeros((self.NU, self.NQ))
            for j in range(0, self.NU):
                q_dot[j, j * self.NTS + i] = 1

            # integrate
            self.xs[i + 1, :], self.xs_dq[i + 1, :, :] = \
                self.ind.fo_forward(tsi, x0, x0_dot,
                                    p, p_dot, q_interval, q_dot)

    # =========================================================================

    def integrate_dsds(self):

        """
        Integrate the dynamics of the optimal control problem and provide
        the second order sensitivities w.r.t. the shooting variables and
        the shooting variables via IND.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        # convert controls and shooting variables
        p = self.p
        q = self.q2array()
        s = self.s2array()

        # set up directions and initial conditions
        self.xs[0, :] = s[0, :]
        self.xs_ds[0, :, :self.NX] = np.eye(self.NX)
        p_dot = np.zeros((self.NP, self.NX))
        p_ddot = np.zeros((self.NP, self.NX, self.NX))
        q_dot = np.zeros((self.NU, self.NX))
        q_ddot = np.zeros((self.NU, self.NX, self.NX))

        # integrate
        for i in range(0, self.NTS - 1):

            # set initial conditions
            x0 = self.xs[i, :]
            x0_dot = self.xs_ds[i, :, :self.NX]
            x0_ddot = self.xs_dsds[i, :, :self.NX, :self.NX]

            # set time steps for this interval
            tsi = np.linspace(self.ts[i], self.ts[i + 1], self.NTSI)

            # set constant controls for this interval
            q_interval = q[:, i, 0]

            # integrate
            self.xs[i + 1, :], self.xs_ds[i + 1, :, :self.NX], _, \
                self.xs_dsds[i + 1, :, :self.NX, :self.NX] = \
                    self.ind.so_forward(tsi, x0, x0_dot, x0_dot, x0_ddot,
                                        p, p_dot, p_dot, p_ddot,
                                        q_interval, q_dot, q_dot, q_ddot)

    # =========================================================================

    def integrate_dpdp(self):

        """
        Integrate the dynamics of the optimal control problem and provide
        the second order sensitivities w.r.t. the parameters and
        the parameters via IND.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        # convert controls and shooting variables
        p = self.p
        q = self.q2array()
        s = self.s2array()

        # set up directions and initial conditions
        self.xs[0, :] = s[0, :]
        p_dot = np.eye(self.NP)
        p_ddot = np.zeros((self.NP, self.NP, self.NP))
        q_dot = np.zeros((self.NU, self.NP))
        q_ddot = np.zeros((self.NU, self.NP, self.NP))

        # integrate
        for i in range(0, self.NTS - 1):

            # set initial conditions
            x0 = self.xs[i, :]
            x0_dot = self.xs_dp[i, :, :]
            x0_ddot = self.xs_dpdp[i, :, :, :]

            # set time steps for this interval
            tsi = np.linspace(self.ts[i], self.ts[i + 1], self.NTSI)

            # set constant controls for this interval
            q_interval = q[:, i, 0]

            # integrate
            self.xs[i + 1, :], self.xs_dp[i + 1, :, :], _, \
                self.xs_dpdp[i + 1, :, :, :] = \
                    self.ind.so_forward(tsi, x0, x0_dot, x0_dot, x0_ddot,
                                        p, p_dot, p_dot, p_ddot,
                                        q_interval, q_dot, q_dot, q_ddot)

    # =========================================================================

    def integrate_dqdq(self):

        """
        Integrate the dynamics of the optimal control problem and provide
        the second order sensitivities w.r.t. the controls and
        the controls via IND.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        # convert controls and shooting variables
        p = self.p
        q = self.q2array()
        s = self.s2array()

        # set initial conditions
        self.xs[0, :] = s[0, :]

        # integrate
        for i in range(0, self.NTS - 1):

            # set time steps for this interval
            tsi = np.linspace(self.ts[i], self.ts[i + 1], self.NTSI)

            # set constant controls for this interval
            q_interval = q[:, i, 0]

            # set up directions and initial conditions
            x0 = self.xs[i, :]
            x0_dot = self.xs_dq[i, :, :]
            x0_ddot = self.xs_dqdq[i, :, :, :]
            p_dot = np.zeros((self.NP, self.NQ))
            p_ddot = np.zeros((self.NP, self.NQ, self.NQ))
            q_dot1 = np.zeros((self.NU, self.NQ))
            for j in range(0, self.NU):
                q_dot1[j, j * self.NTS + i] = 1
            q_dot2 = np.zeros((self.NU, self.NQ))
            for j in range(0, self.NU):
                q_dot2[j, j * self.NTS + i] = 1
            q_ddot = np.zeros((self.NU, self.NQ, self.NQ))

            # integrate
            self.xs[i + 1, :], self.xs_dq[i + 1, :, :], _, \
                self.xs_dqdq[i + 1, :, :] \
                        = self.ind.so_forward(
                            tsi, x0, x0_dot, x0_dot, x0_ddot,
                            p, p_dot, p_dot, p_ddot,
                            q_interval, q_dot2, q_dot1, q_ddot)

    # =========================================================================

    def integrate_dsdp(self):

        """
        Integrate the dynamics of the optimal control problem and provide
        the second order sensitivities w.r.t. the shooting variables and
        the parameters via IND.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        # convert controls and shooting variables
        p = self.p
        q = self.q2array()
        s = self.s2array()

        # set up directions and initial conditions
        self.xs[0, :] = s[0, :]
        self.xs_ds[0, :, :self.NX] = np.eye(self.NX)
        x0_ddot = np.zeros((self.NX, self.NX, self.NP))
        p_dot1 = np.zeros((self.NP, self.NX))
        p_dot2 = np.eye(self.NP)
        p_ddot = np.zeros((self.NP, self.NX, self.NP))
        q_dot1 = np.zeros((self.NU, self.NX))
        q_dot2 = np.zeros((self.NU, self.NP))
        q_ddot = np.zeros((self.NU, self.NX, self.NP))

        # integrate
        for i in range(0, self.NTS - 1):

            # set initial conditions
            x0 = self.xs[i, :]
            x0_dot1 = self.xs_ds[i, :, :self.NX]
            x0_dot2 = self.xs_dp[i, :, :]
            x0_ddot = self.xs_dsdp[i, :, :self.NX, :]

            # set time steps for this interval
            tsi = np.linspace(self.ts[i], self.ts[i + 1], self.NTSI)

            # set constant controls for this interval
            q_interval = q[:, i, 0]

            # integrate
            self.xs[i + 1, :], self.xs_ds[i + 1, :, :self.NX], \
                self.xs_dp[i + 1, :, :], self.xs_dsdp[i + 1, :,
                    :self.NX, :] = self.ind.so_forward(
                        tsi, x0, x0_dot2, x0_dot1, x0_ddot,
                        p, p_dot2, p_dot1, p_ddot,
                        q_interval, q_dot2, q_dot1, q_ddot)

    # =========================================================================

    def integrate_dsdq(self):

        """
        Integrate the dynamics of the optimal control problem and provide
        the second order sensitivities w.r.t. the shooting variables and
        the controls via IND.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        # convert controls and shooting variables
        p = self.p
        q = self.q2array()
        s = self.s2array()

        # set up directions and initial conditions
        self.xs[0, :] = s[0, :]
        self.xs_ds[0, :, :self.NX] = np.eye(self.NX)

        # integrate
        for i in range(0, self.NTS - 1):

            # set time steps for this interval
            tsi = np.linspace(self.ts[i], self.ts[i + 1], self.NTSI)

            # set constant controls for this interval
            q_interval = q[:, i, 0]

            # set up directions and initial conditions
            x0 = self.xs[i, :]
            x0_dot1 = self.xs_ds[i, :, :self.NX]
            x0_dot2 = self.xs_dq[i, :, :]
            x0_ddot = self.xs_dsdq[i, :, :self.NX, :]
            p_dot1 = np.zeros((self.NP, self.NX))
            p_dot2 = np.zeros((self.NP, self.NQ))
            p_ddot = np.zeros((self.NP, self.NX, self.NQ))
            q_dot1 = np.zeros((self.NU, self.NX))
            q_dot2 = np.zeros((self.NU, self.NQ))
            for j in range(0, self.NU):
                q_dot2[j, j * self.NTS + i] = 1
            q_ddot = np.zeros((self.NU, self.NX, self.NQ))

            # integrate
            self.xs[i + 1, :], self.xs_ds[i + 1, :, :self.NX], \
                self.xs_dq[i + 1, :, :], \
                    self.xs_dsdq[i + 1, :, :self.NX, :] \
                        = self.ind.so_forward(
                            tsi, x0, x0_dot2, x0_dot1, x0_ddot,
                            p, p_dot2, p_dot1, p_ddot,
                            q_interval, q_dot2, q_dot1, q_ddot)

    # =========================================================================

    def integrate_dpdq(self):

        """
        Integrate the dynamics of the optimal control problem and provide
        the second order sensitivities w.r.t. the parameters and
        controls via IND.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        # convert controls and shooting variables
        p = self.p
        q = self.q2array()
        s = self.s2array()

        # set up directions and initial conditions
        self.xs[0, :] = s[0, :]

        # integrate
        for i in range(0, self.NTS - 1):

            # set time steps for this interval
            tsi = np.linspace(self.ts[i], self.ts[i + 1], self.NTSI)

            # set constant controls for this interval
            q_interval = q[:, i, 0]

            # set up directions and initial conditions
            x0 = self.xs[i, :]
            x0_dot1 = self.xs_dp[i, :, :]
            x0_dot2 = self.xs_dq[i, :, :]
            x0_ddot = self.xs_dpdq[i, :, :, :]
            p_dot1 = np.eye(self.NP)
            p_dot2 = np.zeros((self.NP, self.NQ))
            p_ddot = np.zeros((self.NP, self.NP, self.NQ))
            q_dot1 = np.zeros((self.NU, self.NP))
            q_dot2 = np.zeros((self.NU, self.NQ))
            for j in range(0, self.NU):
                q_dot2[j, j * self.NTS + i] = 1
            q_ddot = np.zeros((self.NU, self.NP, self.NQ))

            # integrate
            self.xs[i + 1, :], self.xs_dp[i + 1, :, :], \
                self.xs_dq[i + 1, :, :], \
                    self.xs_dpdq[i + 1, :, :, :] \
                        = self.ind.so_forward(
                            tsi, x0, x0_dot2, x0_dot1, x0_ddot,
                            p, p_dot2, p_dot1, p_ddot,
                            q_interval, q_dot2, q_dot1, q_ddot)

    # =========================================================================

    def calc_ineqc(self):

        """
        Calculate the inequality constraints.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        # allocate memory
        x = np.zeros((self.NX,))
        g = np.zeros((self.NG,))
        u = np.zeros((self.NU,))

        # loop through all time steps
        for i in range(0, self.NTS):

            # state and controls for this time step
            x = self.xs[i, :]

            for k in range(0, self.NU):
                u[k] = q[i + k * self.NTS]

            # call fortran backend
            self.backend_fortran.gfcn(g, self.ts[i:i + 1], x, p, u)

            # build constraints
            for k in range(0, self.NG):
                self.ineqc[i + k * self.NTS] = g[k]

    # =========================================================================

    def calc_ineqc_ds(self):

        """
        Calculate the first order derivatives of the inequality constraints
        w.r.t. the shooting variables.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        x = np.zeros((self.NX,))
        x_dot = np.zeros((self.NX, self.NX))
        g = np.zeros((self.NG,))
        g_dot = np.zeros((self.NG, self.NX))
        p_dot = np.zeros((self.NP, self.NX))
        u = np.zeros((self.NU,))
        u_dot = np.zeros((self.NU, self.NX))

        # loop through all time step
        for i in range(0, self.NTS):

            # state and controls for this time step
            x     = self.xs[i, :]
            x_dot = np.reshape(self.xs_ds[i, :, :self.NX], x_dot.shape)

            for l in range(0, self.NU):
                u[l] = q[i + l * self.NTS]

            # call fortran backend
            self.backend_fortran.gfcn_dot(g, g_dot, self.ts[i:i + 1],
                                          x, x_dot, p, p_dot, u, u_dot)

            # store gradient
            for k in range(0, self.NG):
                self.ineqc_ds[i + k * self.NTS, :self.NX] = g_dot[k, :]
                self.ineqc[i + k * self.NTS] = g[k]

    # =========================================================================

    def calc_ineqc_dp(self):

        """
        Calculate the first order derivatives of the inequality constraints
        w.r.t. the parameters.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        x = np.zeros((self.NX,))
        x_dot = np.zeros((self.NX, self.NP))
        g = np.zeros((self.NG,))
        g_dot = np.zeros((self.NG, self.NP))
        p_dot = np.eye(self.NP)
        u = np.zeros((self.NU,))
        u_dot = np.zeros((self.NU, self.NP))

        # loop through all time steps
        for i in range(0, self.NTS):

            # state and controls for this time step
            x  = self.xs[i, :]
            x_dot = np.reshape(self.xs_dp[i, :, :], x_dot.shape)

            for l in range(0, self.NU):
                u[l] = q[i + l * self.NTS]

            # call fortran backend
            self.backend_fortran.gfcn_dot(g, g_dot, self.ts[i:i + 1],
                                          x, x_dot, p, p_dot, u, u_dot)

            # store gradient
            for k in range(0, self.NG):
                self.ineqc[i + k * self.NTS] = g[k]
                self.ineqc_dp[i + k * self.NTS, :] = g_dot[k, :]

    # =========================================================================

    def calc_ineqc_dq(self):

        """
        Calculate the first order derivatives of the inequality constraints
        w.r.t. the controls.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        x = np.zeros((self.NX,))
        x_dot = np.zeros((self.NX, self.NU))
        g = np.zeros((self.NG,))
        g_dot = np.zeros((self.NG, self.NU))
        p_dot = np.zeros((self.NP, self.NU))
        u  = np.zeros((self.NU,))
        u_dot = np.zeros((self.NU, self.NU))

        # loop through all time steps
        for i in range(0, self.NTS):

            # loop through all previous controls plus the current one
            for j in range(0, i + 1):

                # state and controls for this time step
                x = self.xs[i, :]
                x_dot = np.reshape(self.xs_dq[i, :, j], x_dot.shape)

                for k in range(0, self.NU):
                    u[k] = q[i + k * self.NTS]

                if i == j:
                    u_dot = np.eye(self.NU)
                else:
                    u_dot = np.zeros((self.NU, self.NU))

                # call fortran backend
                self.backend_fortran.gfcn_dot(g, g_dot, self.ts[i:i + 1],
                                              x, x_dot, p, p_dot, u, u_dot)

                # store gradient
                for k in range(0, self.NG):
                    self.ineqc[i + k * self.NTS] = g[k]

                    for l in range(0, self.NU):
                        self.ineqc_dq[i + k * self.NTS,
                            j + l * self.NTS] = g_dot[k, l]

    # =========================================================================

    def calc_ineqc_dsds(self):

        """
        Calculate the second order derivatives of the inequality constraints
        w.r.t. the shooting variables and the shooting variables.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        x = np.zeros((self.NX,))
        x_dot = np.zeros((self.NX, self.NX))
        x_ddot = np.zeros(x_dot.shape + (self.NX,))
        g = np.zeros((self.NG,))
        g_dot1 = np.zeros((self.NG, self.NX))
        g_dot2 = np.zeros((self.NG, self.NX))
        g_ddot = np.zeros(g_dot1.shape + (self.NX,))
        p_dot = np.zeros((self.NP, self.NX))
        p_ddot = np.zeros(p_dot.shape + (self.NX,))
        u = np.zeros((self.NU,))
        u_dot = np.zeros((self.NU, self.NX))
        u_ddot = np.zeros(u_dot.shape + (self.NX,))

        # loop through all time step
        for i in range(0, self.NTS):

            # state and controls for this time step
            x = self.xs[i, :]
            x_dot = np.reshape(self.xs_ds[i, :, :self.NX], x_dot.shape)
            x_ddot = np.reshape(self.xs_dsds[i, :, :self.NX, :self.NX],
                                x_ddot.shape)

            for l in range(0, self.NU):
                u[l] = q[i + l * self.NTS]

            # call fortran backend to calculate derivatives of constraint functions
            self.backend_fortran.gfcn_ddot(g, g_dot2, g_dot1, g_ddot,
                                           self.ts[i:i + 1],
                                           x, x_dot, x_dot, x_ddot,
                                           p, p_dot, p_dot, p_ddot,
                                           u, u_dot, u_dot, u_ddot)

            # store gradient
            for k in range(0, self.NG):
                self.ineqc_dsds[i + k * self.NTS, :self.NX, :self.NX] = \
                    g_ddot[k, :, :]
                self.ineqc_ds[i + k * self.NTS, :self.NX] = g_dot1[k, :]
                self.ineqc[i + k * self.NTS] = g[k]


    # =========================================================================

    def calc_ineqc_dpdp(self):

        """
        Calculate the second order derivatives of the inequality constraints
        w.r.t. the parameters and the parameters.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        x = np.zeros((self.NX,))
        x_dot = np.zeros((self.NX, self.NP))
        x_ddot = np.zeros(x_dot.shape + (self.NP,))
        g = np.zeros((self.NG,))
        g_dot1 = np.zeros((self.NG, self.NP))
        g_dot2 = np.zeros((self.NG, self.NP))
        g_ddot = np.zeros(g_dot1.shape + (self.NP,))
        p_dot  = np.eye(self.NP)
        p_ddot = np.zeros(p_dot.shape + (self.NP,))
        u = np.zeros((self.NU,))
        u_dot = np.zeros((self.NU, self.NP))
        u_ddot = np.zeros(u_dot.shape + (self.NP,))

        # loop through all time step
        for i in range(0, self.NTS):

            # state and controls for this time step
            x = self.xs[i, :]
            x_dot = np.reshape(self.xs_dp[i, :, :], x_dot.shape)
            x_ddot = np.reshape(self.xs_dpdp[i, :, :, :], x_ddot.shape)

            for l in range(0, self.NU):
                u[l] = q[i + l * self.NTS]

            # call fortran backend to calculate derivatives of constraint functions
            self.backend_fortran.gfcn_ddot(g, g_dot2, g_dot1, g_ddot,
                                           self.ts[i:i + 1],
                                           x, x_dot, x_dot, x_ddot,
                                           p, p_dot, p_dot, p_ddot,
                                           u, u_dot, u_dot, u_ddot)

            # store gradient
            for k in range(0, self.NG):
                self.ineqc_dpdp[i + k * self.NTS, :, :] = g_ddot[k, :, :]
                self.ineqc_dp[i + k * self.NTS, :] = g_dot1[k, :]
                self.ineqc[i + k * self.NTS] = g[k]

    # =========================================================================

    def calc_ineqc_dqdq(self):

        """
        Calculate the second order derivatives of the inequality constraints
        w.r.t. the controls and the controls.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        x = np.zeros((self.NX,))
        x_dot1 = np.zeros((self.NX, self.NU))
        x_dot2 = np.zeros((self.NX, self.NU))
        x_ddot = np.zeros(x_dot1.shape + (self.NU,))
        g = np.zeros((self.NG,))
        g_dot1 = np.zeros((self.NG, self.NU))
        g_dot2 = np.zeros((self.NG, self.NU))
        g_ddot = np.zeros(g_dot1.shape + (self.NU,))
        p_dot = np.zeros((self.NP, self.NU))
        p_ddot = np.zeros(p_dot.shape + (self.NU,))
        u = np.zeros((self.NU,))
        u_dot1 = np.zeros((self.NU, self.NU))
        u_dot2 = np.zeros((self.NU, self.NU))
        u_ddot = np.zeros(u_dot1.shape + (self.NU,))

        # loop through all time steps three times
        for i in range(0, self.NTS):
            for j in range(0, i + 1):
                for m in range(0, i + 1):

                    # state and controls for this time step
                    x = self.xs[i, :]
                    x_dot1 = np.reshape(self.xs_dq[i, :, j], x_dot1.shape)
                    x_dot2 = np.reshape(self.xs_dq[i, :, m], x_dot2.shape)
                    x_ddot = np.reshape(self.xs_dqdq[i, :, j, m], x_ddot.shape)

                    for k in range(0, self.NU):
                        u[k] = q[i + k * self.NTS]

                    if i == j:
                        u_dot1 = np.eye(self.NU)
                    else:
                        u_dot1 = np.zeros((self.NU, self.NU))

                    if i == m:
                        u_dot2 = np.eye(self.NU)
                    else:
                        u_dot2 = np.zeros((self.NU, self.NU))

                    # call fortran backend
                    self.backend_fortran.gfcn_ddot(g, g_dot2, g_dot1, g_ddot,
                                                   self.ts[i:i + 1],
                                                   x, x_dot2, x_dot1, x_ddot,
                                                   p, p_dot, p_dot, p_ddot,
                                                   u, u_dot2, u_dot1, u_ddot)

                    # store gradient
                    for k in range(0, self.NG):
                        self.ineqc[i + k * self.NTS] = g[k]

                        for l in range(0, self.NU):
                            for b in range(0, self.NU):
                                self.ineqc_dq[i + k * self.NTS,
                                    j + l * self.NTS] = g_dot1[k, l]
                                self.ineqc_dqdq[i + k * self.NTS,
                                    j + l * self.NTS, m + b * self.NTS] = \
                                        g_ddot[k, l, b]

    # =========================================================================

    def calc_ineqc_dsdp(self):

        """
        Calculate the second order derivatives of the inequality constraints
        w.r.t. the shooting variables and the parameters.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        x = np.zeros((self.NX,))
        x_dot1 = np.zeros((self.NX, self.NX))
        x_dot2 = np.zeros((self.NX, self.NP))
        x_ddot = np.zeros(x_dot1.shape + (self.NP,))
        g = np.zeros((self.NG,))
        g_dot1 = np.zeros((self.NG, self.NX))
        g_dot2 = np.zeros((self.NG, self.NP))
        g_ddot = np.zeros(g_dot1.shape + (self.NP,))
        p_dot1 = np.zeros((self.NP, self.NX))
        p_dot2 = np.eye(self.NP)
        p_ddot = np.zeros(p_dot1.shape + (self.NP,))
        u = np.zeros((self.NU,))
        u_dot = np.zeros((self.NU, self.NX))
        u_ddot = np.zeros(u_dot.shape + (self.NP,))

        # loop through all time step
        for i in range(0, self.NTS):

            # state and controls for this time step
            x      = self.xs[i, :]
            x_dot1 = np.reshape(self.xs_ds[i, :, :self.NX], x_dot1.shape)
            x_dot2 = np.reshape(self.xs_dp[i, :, :], x_dot2.shape)
            x_ddot = np.reshape(self.xs_dsdp[i, :, :self.NX, :], x_ddot.shape)

            for l in range(0, self.NU):
                u[l] = q[i + l * self.NTS]

            # call fortran backend to calculate derivatives of constraint functions
            self.backend_fortran.gfcn_ddot(g, g_dot2, g_dot1, g_ddot,
                                           self.ts[i:i + 1],
                                           x, x_dot2, x_dot1, x_ddot,
                                           p, p_dot2, p_dot1, p_ddot,
                                           u, u_dot, u_dot, u_ddot)

            # store gradient
            for k in range(0, self.NG):
                self.ineqc_dsdp[i + k * self.NTS, :self.NX:, :] = \
                    g_ddot[k, :, :]
                self.ineqc_ds[i + k * self.NTS, :self.NX] = g_dot1[k, :]
                self.ineqc_dp[i + k * self.NTS, :] = g_dot2[k, :]
                self.ineqc[i + k * self.NTS] = g[k]

    # =========================================================================

    def calc_ineqc_dsdq(self):

        """
        Calculate the second order derivatives of the inequality constraints
        w.r.t. the shooting variables and the controls.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        x = np.zeros((self.NX,))
        x_dot1 = np.zeros((self.NX, self.NX))
        x_dot2 = np.zeros((self.NX, self.NU))
        x_ddot = np.zeros(x_dot1.shape + (self.NU,))
        g = np.zeros((self.NG,))
        g_dot1 = np.zeros((self.NG, self.NX))
        g_dot2 = np.zeros((self.NG, self.NU))
        g_ddot = np.zeros(g_dot1.shape + (self.NU,))
        p_dot = np.zeros((self.NP, self.NX))
        p_ddot = np.zeros(p_dot.shape + (self.NU,))
        u = np.zeros((self.NU,))
        u_dot1 = np.zeros((self.NU, self.NX))
        u_dot2 = np.zeros((self.NU, self.NU))
        u_ddot = np.zeros(u_dot1.shape + (self.NU,))

        # loop through all time steps
        for i in range(0, self.NTS):

            # loop through all time steps including the current one
            for j in range(0, i + 1):

                # state and controls for this time step
                x = self.xs[i, :]
                x_dot1 = np.reshape(self.xs_ds[i, :, :self.NX], x_dot1.shape)
                x_dot2 = np.reshape(self.xs_dq[i, :, j], x_dot2.shape)
                x_ddot = np.reshape(self.xs_dsdq[i, :, :self.NX, j],
                                    x_ddot.shape)

                for l in range(0, self.NU):
                    u[l] = q[i + l * self.NTS]

                if i == j:
                    u_dot2 = np.eye(self.NU)
                else:
                    u_dot2 = np.zeros((self.NU, self.NU))

                # call fortran backend
                self.backend_fortran.gfcn_ddot(g, g_dot2, g_dot1, g_ddot,
                                               self.ts[i:i + 1],
                                               x, x_dot2, x_dot1, x_ddot,
                                               p, p_dot, p_dot, p_ddot,
                                               u, u_dot2, u_dot1, u_ddot)

                # store gradient
                for k in range(0, self.NG):
                    self.ineqc[i + k * self.NTS] = g[k]
                    self.ineqc_ds[i + k * self.NTS, :self.NX] = g_dot1[k, :]

                    for l in range(0, self.NU):
                        self.ineqc_dsdq[i + k * self.NTS, :self.NX,
                            j + l * self.NTS] = g_ddot[k, :, l]
                        self.ineqc_dq[i + k * self.NTS, j + l * self.NTS] = \
                            g_dot2[k, l]

    # =========================================================================

    def calc_ineqc_dpdq(self):

        """
        Calculate the second order derivatives of the inequality constraints
        w.r.t. the parameters and the controls.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        x = np.zeros((self.NX,))
        x_dot1 = np.zeros((self.NX, self.NP))
        x_dot2 = np.zeros((self.NX, self.NU))
        x_ddot = np.zeros(x_dot1.shape + (self.NU,))
        g = np.zeros((self.NG,))
        g_dot1 = np.zeros((self.NG, self.NP))
        g_dot2 = np.zeros((self.NG, self.NU))
        g_ddot = np.zeros(g_dot1.shape + (self.NU,))
        p_dot1 = np.eye(self.NP)
        p_dot2 = np.zeros((self.NP, self.NU))
        p_ddot = np.zeros(p_dot1.shape + (self.NU,))
        u = np.zeros((self.NU,))
        u_dot1 = np.zeros((self.NU, self.NP))
        u_dot2 = np.zeros((self.NU, self.NU))
        u_ddot = np.zeros(u_dot1.shape + (self.NU,))

        # loop through all time step
        for i in range(0, self.NTS):

            # loop through all time steps including the current one
            for j in range(0, i + 1):

                # state and controls for this time step
                x = self.xs[i, :]
                x_dot1 = np.reshape(self.xs_dp[i, :, :], x_dot1.shape)
                x_dot2 = np.reshape(self.xs_dq[i, :, j], x_dot2.shape)
                x_ddot = np.reshape(self.xs_dpdq[i, :, :, j], x_ddot.shape)

                for l in range(0, self.NU):
                    u[l] = q[i + l * self.NTS]

                if i == j:
                    u_dot2 = np.eye(self.NU)
                else:
                    u_dot2 = np.zeros((self.NU, self.NU))

                # call fortran backend
                self.backend_fortran.gfcn_ddot(g, g_dot2, g_dot1, g_ddot,
                                               self.ts[i:i + 1],
                                               x, x_dot2, x_dot1, x_ddot,
                                               p, p_dot2, p_dot1, p_ddot,
                                               u, u_dot2, u_dot1, u_ddot)

                # store gradient
                for k in range(0, self.NG):
                    self.ineqc[i + k * self.NTS] = g[k]
                    self.ineqc_dp[i + k * self.NTS, :] = g_dot1[k, :]

                    for l in range(0, self.NU):
                        self.ineqc_dpdq[i + k * self.NTS, :, j + l * \
                                        self.NTS] = g_ddot[k, :, l]
                        self.ineqc_dq[i + k * self.NTS, j + l * \
                                      self.NTS] = g_dot2[k, l]

    # =========================================================================

    def calc_eqc(self):

        """
        Calculate the equality constraints.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        # allocate memory
        x = np.zeros((self.NX,))
        h = np.zeros((self.NH,))
        u = np.zeros((self.NU,))

        # loop through all time steps
        for i in range(0, self.NTS):

            # state and controls for this time step
            x = self.xs[i, :]

            for k in range(0, self.NU):
                u[k] = q[i + k * self.NTS]

            # call fortran backend
            self.backend_fortran.hfcn(h, self.ts[i:i + 1], x, p, u)

            # build constraints
            for k in range(0, self.NH):
                self.eqc[i + k * self.NTS] = h[k]

    # =========================================================================

    def calc_eqc_ds(self):

        """
        Calculate the first order derivatives of the equality constraints
        w.r.t. the shooting variables.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        x = np.zeros((self.NX,))
        x_dot = np.zeros((self.NX, self.NX))
        h = np.zeros((self.NH,))
        h_dot = np.zeros((self.NH, self.NX))
        p_dot = np.zeros((self.NP, self.NX))
        u = np.zeros((self.NU,))
        u_dot = np.zeros((self.NU, self.NX))

        # loop through all time step
        for i in range(0, self.NTS):

            # state and controls for this time step
            x     = self.xs[i, :]
            x_dot = np.reshape(self.xs_ds[i, :, :self.NX], x_dot.shape)

            for l in range(0, self.NU):
                u[l] = q[i + l * self.NTS]

            # call fortran backend
            self.backend_fortran.hfcn_dot(h, h_dot, self.ts[i:i + 1],
                                          x, x_dot, p, p_dot, u, u_dot)

            # store gradient
            for k in range(0, self.NH):

                self.eqc_ds[i + k * self.NTS, :self.NX] = h_dot[k, :]
                self.eqc[i + k * self.NTS] = h[k]

    # =========================================================================

    def calc_eqc_dp(self):

        """
        Calculate the first order derivatives of the equality constraints
        w.r.t. the parameters.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        x = np.zeros((self.NX,))
        x_dot = np.zeros((self.NX, self.NP))
        h = np.zeros((self.NH,))
        h_dot = np.zeros((self.NH, self.NP))
        p_dot = np.eye(self.NP)
        u = np.zeros((self.NU,))
        u_dot = np.zeros((self.NU, self.NP))

        # loop through all time steps
        for i in range(0, self.NTS):

            # state and controls for this time step
            x  = self.xs[i, :]
            x_dot = np.reshape(self.xs_dp[i, :, :], x_dot.shape)

            for l in range(0, self.NU):
                u[l] = q[i + l * self.NTS]

            # call fortran backend
            self.backend_fortran.hfcn_dot(h, h_dot, self.ts[i:i + 1],
                                          x, x_dot, p, p_dot, u, u_dot)

            # store gradient
            for k in range(0, self.NH):
                self.eqc[i + k * self.NTS] = h[k]
                self.eqc_dp[i + k * self.NTS, :] = h_dot[k, :]

    # =========================================================================

    def calc_eqc_dq(self):

        """
        Calculate the first order derivatives of the equality constraints
        w.r.t. the controls.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        x = np.zeros((self.NX,))
        x_dot = np.zeros((self.NX, self.NU))
        h = np.zeros((self.NH,))
        h_dot = np.zeros((self.NH, self.NU))
        p_dot = np.zeros((self.NP, self.NU))
        u  = np.zeros((self.NU,))
        u_dot = np.zeros((self.NU, self.NU))

        # loop through all time steps
        for i in range(0, self.NTS):

            # loop through all previous controls plus the current one
            for j in range(0, i + 1):

                # state and controls for this time step
                x = self.xs[i, :]
                x_dot = np.reshape(self.xs_dq[i, :, j], x_dot.shape)

                for k in range(0, self.NU):
                    u[k] = q[i + k * self.NTS]

                if i == j:
                    u_dot = np.eye(self.NU)
                else:
                    u_dot = np.zeros((self.NU, self.NU))

                # call fortran backend
                self.backend_fortran.hfcn_dot(h, h_dot, self.ts[i:i + 1],
                                              x, x_dot, p, p_dot, u, u_dot)

                # store gradient
                for k in range(0, self.NH):
                    self.eqc[i + k * self.NTS] = h[k]

                    for l in range(0, self.NU):
                        self.eqc_dq[i + k * self.NTS,
                            j + l * self.NTS] = h_dot[k, l]

    # =========================================================================

    def calc_eqc_dsds(self):

        """
        Calculate the second order derivatives of the equality constraints
        w.r.t. the shooting variables and the shooting variables.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        x = np.zeros((self.NX,))
        x_dot = np.zeros((self.NX, self.NX))
        x_ddot = np.zeros(x_dot.shape + (self.NX,))
        h = np.zeros((self.NH,))
        h_dot1 = np.zeros((self.NH, self.NX))
        h_dot2 = np.zeros((self.NH, self.NX))
        h_ddot = np.zeros(h_dot1.shape + (self.NX,))
        p_dot = np.zeros((self.NP, self.NX))
        p_ddot = np.zeros(p_dot.shape + (self.NX,))
        u = np.zeros((self.NU,))
        u_dot = np.zeros((self.NU, self.NX))
        u_ddot = np.zeros(u_dot.shape + (self.NX,))

        # loop through all time step
        for i in range(0, self.NTS):

            # state and controls for this time step
            x = self.xs[i, :]
            x_dot = np.reshape(self.xs_ds[i, :, :self.NX], x_dot.shape)
            x_ddot = np.reshape(self.xs_dsds[i, :, :self.NX, :self.NX],
                                x_ddot.shape)

            for l in range(0, self.NU):
                u[l] = q[i + l * self.NTS]

            # call fortran backend
            self.backend_fortran.hfcn_ddot(h, h_dot2, h_dot1, h_ddot,
                                           self.ts[i:i + 1],
                                           x, x_dot, x_dot, x_ddot,
                                           p, p_dot, p_dot, p_ddot,
                                           u, u_dot, u_dot, u_ddot)

            # store gradient
            for k in range(0, self.NH):
                self.eqc_dsds[i + k * self.NTS, :self.NX, :self.NX] = \
                    h_ddot[k, :, :]
                self.eqc_ds[i + k * self.NTS, :self.NX] = h_dot1[k, :]
                self.eqc[i + k * self.NTS] = h[k]


    # =========================================================================

    def calc_eqc_dpdp(self):

        """
        Calculate the second order derivatives of the equality constraints
        w.r.t. the parameters and the parameters.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        x = np.zeros((self.NX,))
        x_dot = np.zeros((self.NX, self.NP))
        x_ddot = np.zeros(x_dot.shape + (self.NP,))
        h = np.zeros((self.NH,))
        h_dot1 = np.zeros((self.NH, self.NP))
        h_dot2 = np.zeros((self.NH, self.NP))
        h_ddot = np.zeros(h_dot1.shape + (self.NP,))
        p_dot = np.eye(self.NP)
        p_ddot = np.zeros(p_dot.shape + (self.NP,))
        u = np.zeros((self.NU,))
        u_dot = np.zeros((self.NU, self.NP))
        u_ddot = np.zeros(u_dot.shape + (self.NP,))

        # loop through all time step
        for i in range(0, self.NTS):

            # state and controls for this time step
            x = self.xs[i, :]
            x_dot = np.reshape(self.xs_dp[i, :, :], x_dot.shape)
            x_ddot = np.reshape(self.xs_dpdp[i, :, :, :], x_ddot.shape)

            for l in range(0, self.NU):
                u[l] = q[i + l * self.NTS]

            # call fortran backend
            self.backend_fortran.hfcn_ddot(h, h_dot2, h_dot1, h_ddot,
                                           self.ts[i:i + 1],
                                           x, x_dot, x_dot, x_ddot,
                                           p, p_dot, p_dot, p_ddot,
                                           u, u_dot, u_dot, u_ddot)

            # store gradient
            for k in range(0, self.NH):
                self.eqc_dpdp[i + k * self.NTS, :, :] = h_ddot[k, :, :]
                self.eqc_dp[i + k * self.NTS, :] = h_dot1[k, :]
                self.eqc[i + k * self.NTS] = h[k]

    # =========================================================================

    def calc_eqc_dqdq(self):

        """
        Calculate the second order derivatives of the equality constraints
        w.r.t. the controls and the controls.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        x = np.zeros((self.NX,))
        x_dot1 = np.zeros((self.NX, self.NU))
        x_dot2 = np.zeros((self.NX, self.NU))
        x_ddot = np.zeros(x_dot1.shape + (self.NU,))
        h = np.zeros((self.NH,))
        h_dot1 = np.zeros((self.NH, self.NU))
        h_dot2 = np.zeros((self.NH, self.NU))
        h_ddot = np.zeros(h_dot1.shape + (self.NU,))
        p_dot = np.zeros((self.NP, self.NU))
        p_ddot = np.zeros(p_dot.shape + (self.NU,))
        u = np.zeros((self.NU,))
        u_dot1 = np.zeros((self.NU, self.NU))
        u_dot2 = np.zeros((self.NU, self.NU))
        u_ddot = np.zeros(u_dot1.shape + (self.NU,))

        # loop through all time steps three times
        for i in range(0, self.NTS):
            for j in range(0, i + 1):
                for m in range(0, i + 1):

                    # state and controls for this time step
                    x = self.xs[i, :]
                    x_dot1 = np.reshape(self.xs_dq[i, :, j], x_dot1.shape)
                    x_dot2 = np.reshape(self.xs_dq[i, :, m], x_dot2.shape)
                    x_ddot = np.reshape(self.xs_dqdq[i, :, j, m], x_ddot.shape)

                    for k in range(0, self.NU):
                        u[k] = q[i + k * self.NTS]

                    if i == j:
                        u_dot1 = np.eye(self.NU)
                    else:
                        u_dot1 = np.zeros((self.NU, self.NU))

                    if i == m:
                        u_dot2 = np.eye(self.NU)
                    else:
                        u_dot2 = np.zeros((self.NU, self.NU))

                    # call fortran backend
                    self.backend_fortran.hfcn_ddot(h, h_dot2, h_dot1, h_ddot,
                                                   self.ts[i:i + 1],
                                                   x, x_dot2, x_dot1, x_ddot,
                                                   p, p_dot, p_dot, p_ddot,
                                                   u, u_dot2, u_dot1, u_ddot)

                    # store gradient
                    for k in range(0, self.NH):
                        self.eqc[i + k * self.NTS] = h[k]

                        for l in range(0, self.NU):
                            for b in range(0, self.NU):
                                self.eqc_dq[i + k * self.NTS,
                                    j + l * self.NTS] = h_dot1[k, l]
                                self.eqc_dqdq[i + k * self.NTS,
                                    j + l * self.NTS, m + b * self.NTS] = \
                                        h_ddot[k, l, b]

    # =========================================================================

    def calc_eqc_dsdp(self):

        """
        Calculate the second order derivatives of the equality constraints
        w.r.t. the shooting variables and the parameters.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        x = np.zeros((self.NX,))
        x_dot1 = np.zeros((self.NX, self.NX))
        x_dot2 = np.zeros((self.NX, self.NP))
        x_ddot = np.zeros(x_dot1.shape + (self.NP,))
        h = np.zeros((self.NH,))
        h_dot1 = np.zeros((self.NH, self.NX))
        h_dot2 = np.zeros((self.NH, self.NP))
        h_ddot = np.zeros(h_dot1.shape + (self.NP,))
        p_dot1 = np.zeros((self.NP, self.NX))
        p_dot2 = np.eye(self.NP)
        p_ddot = np.zeros(p_dot1.shape + (self.NP,))
        u = np.zeros((self.NU,))
        u_dot = np.zeros((self.NU, self.NX))
        u_ddot = np.zeros(u_dot.shape + (self.NP,))

        # loop through all time step
        for i in range(0, self.NTS):

            # state and controls for this time step
            x = self.xs[i, :]
            x_dot1 = np.reshape(self.xs_ds[i, :, :self.NX], x_dot1.shape)
            x_dot2 = np.reshape(self.xs_dp[i, :, :], x_dot2.shape)
            x_ddot = np.reshape(self.xs_dsdp[i, :, :self.NX, :], x_ddot.shape)

            for l in range(0, self.NU):
                u[l] = q[i + l * self.NTS]

            # call fortran backend
            self.backend_fortran.hfcn_ddot(h, h_dot2, h_dot1, h_ddot,
                                           self.ts[i:i + 1],
                                           x, x_dot2, x_dot1, x_ddot,
                                           p, p_dot2, p_dot1, p_ddot,
                                           u, u_dot, u_dot, u_ddot)

            # store gradient
            for k in range(0, self.NH):
                self.eqc_dsdp[i + k * self.NTS, :self.NX:, :] = \
                    h_ddot[k, :, :]
                self.eqc_ds[i + k * self.NTS, :self.NX] = h_dot1[k, :]
                self.eqc_dp[i + k * self.NTS, :] = h_dot2[k, :]
                self.eqc[i + k * self.NTS] = h[k]

    # =========================================================================

    def calc_eqc_dsdq(self):

        """
        Calculate the second order derivatives of the equality constraints
        w.r.t. the shooting variables and the controls.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        x = np.zeros((self.NX,))
        x_dot1 = np.zeros((self.NX, self.NX))
        x_dot2 = np.zeros((self.NX, self.NU))
        x_ddot = np.zeros(x_dot1.shape + (self.NU,))
        h = np.zeros((self.NH,))
        h_dot1 = np.zeros((self.NH, self.NX))
        h_dot2 = np.zeros((self.NH, self.NU))
        h_ddot = np.zeros(h_dot1.shape + (self.NU,))
        p_dot  = np.zeros((self.NP, self.NX))
        p_ddot = np.zeros(p_dot.shape + (self.NU,))
        u = np.zeros((self.NU,))
        u_dot1 = np.zeros((self.NU, self.NX))
        u_dot2 = np.zeros((self.NU, self.NU))
        u_ddot = np.zeros(u_dot1.shape + (self.NU,))

        # loop through all time steps
        for i in range(0, self.NTS):

            # loop through all time steps including the current one
            for j in range(0, i + 1):

                # state and controls for this time step
                x = self.xs[i, :]
                x_dot1 = np.reshape(self.xs_ds[i, :, :self.NX], x_dot1.shape)
                x_dot2 = np.reshape(self.xs_dq[i, :, j], x_dot2.shape)
                x_ddot = np.reshape(self.xs_dsdq[i, :, :self.NX, j],
                                    x_ddot.shape)

                for l in range(0, self.NU):
                    u[l] = q[i + l * self.NTS]

                if i == j:
                    u_dot2 = np.eye(self.NU)
                else:
                    u_dot2 = np.zeros((self.NU, self.NU))

                # call fortran backend
                self.backend_fortran.hfcn_ddot(h, h_dot2, h_dot1, h_ddot,
                                               self.ts[i:i + 1],
                                               x, x_dot2, x_dot1, x_ddot,
                                               p, p_dot, p_dot, p_ddot,
                                               u, u_dot2, u_dot1, u_ddot)

                # store gradient
                for k in range(0, self.NH):
                    self.eqc[i + k * self.NTS] = h[k]
                    self.eqc_ds[i + k * self.NTS, :self.NX] = h_dot1[k, :]

                    for l in range(0, self.NU):
                        self.eqc_dsdq[i + k * self.NTS, :self.NX,
                            j + l * self.NTS] = h_ddot[k, :, l]
                        self.eqc_dq[i + k * self.NTS, j + l * self.NTS] = \
                            h_dot2[k, l]

    # =========================================================================

    def calc_eqc_dpdq(self):

        """
        Calculate the second order derivatives of the equality constraints
        w.r.t. the parameters and the controls.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        x = np.zeros((self.NX,))
        x_dot1 = np.zeros((self.NX, self.NP))
        x_dot2 = np.zeros((self.NX, self.NU))
        x_ddot = np.zeros(x_dot1.shape + (self.NU,))
        h = np.zeros((self.NH,))
        h_dot1 = np.zeros((self.NH, self.NP))
        h_dot2 = np.zeros((self.NH, self.NU))
        h_ddot = np.zeros(h_dot1.shape + (self.NU,))
        p_dot1 = np.eye(self.NP)
        p_dot2 = np.zeros((self.NP, self.NU))
        p_ddot = np.zeros(p_dot1.shape + (self.NU,))
        u = np.zeros((self.NU,))
        u_dot1 = np.zeros((self.NU, self.NP))
        u_dot2 = np.zeros((self.NU, self.NU))
        u_ddot = np.zeros(u_dot1.shape + (self.NU,))

        # loop through all time step
        for i in range(0, self.NTS):

            # loop through all time steps including the current one
            for j in range(0, i + 1):

                # state and controls for this time step
                x = self.xs[i, :]
                x_dot1 = np.reshape(self.xs_dp[i, :, :], x_dot1.shape)
                x_dot2 = np.reshape(self.xs_dq[i, :, j], x_dot2.shape)
                x_ddot = np.reshape(self.xs_dpdq[i, :, :, j], x_ddot.shape)

                for l in range(0, self.NU):
                    u[l] = q[i + l * self.NTS]

                if i == j:
                    u_dot2 = np.eye(self.NU)
                else:
                    u_dot2 = np.zeros((self.NU, self.NU))

                # call fortran backend
                self.backend_fortran.hfcn_ddot(h, h_dot2, h_dot1, h_ddot,
                                               self.ts[i:i + 1],
                                               x, x_dot2, x_dot1, x_ddot,
                                               p, p_dot2, p_dot1, p_ddot,
                                               u, u_dot2, u_dot1, u_ddot)

                # store gradient
                for k in range(0, self.NH):
                    self.eqc[i + k * self.NTS] = h[k]
                    self.eqc_dp[i + k * self.NTS, :] = h_dot1[k, :]

                    for l in range(0, self.NU):
                        self.eqc_dpdq[i + k * self.NTS, :, j + l * \
                                      self.NTS] = h_ddot[k, :, l]
                        self.eqc_dq[i + k * self.NTS, j + l * \
                                    self.NTS] = h_dot2[k, l]

    # =========================================================================

    def calc_mc(self):

        """
        Calculate the matching conditions.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        # evaluate matching conditions
        s = self.s2array()
        self.mc[:] = self.xs[-1, :] - s[-1, :]

    # =========================================================================

    def calc_mc_ds(self):

        """
        Calculate the first order derivatives of the matching conditions
        w.r.t. the shooting variables.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        # evaluate matching conditions
        self.mc_ds[:] = self.xs_ds[-1, :, :]
        self.mc_ds[:, self.NX:] = -np.eye(self.NX)

    # =========================================================================

    def calc_mc_dp(self):

        """
        Calculate the first order derivatives of the matching conditions
        w.r.t. the parameters.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        # evaluate matching conditions
        self.mc_dp[:] = self.xs_dp[-1, :, :]

    # =========================================================================

    def calc_mc_dq(self):

        """
        Calculate the first order derivatives of the matching conditions
        w.r.t. the controls.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        # evaluate matching conditions
        self.mc_dq[:] = self.xs_dq[-1, :, :]

    # =========================================================================

    def calc_mc_dsds(self):

        """
        Calculate the second order derivatives of the matching conditions
        w.r.t. the shooting variables and the shooting variables.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        # evaluate matching conditions
        self.mc_dsds[:] = self.xs_dsds[-1, :, :, :]

    # =========================================================================

    def calc_mc_dpdp(self):

        """
        Calculate the second order derivatives of the matching conditions
        w.r.t. the parameters and the parameters.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        # evaluate matching conditions
        self.mc_dpdp[:] = self.xs_dpdp[-1, :, :, :]

    # =========================================================================

    def calc_mc_dqdq(self):

        """
        Calculate the second order derivatives of the matching conditions
        w.r.t. the controls and the controls.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        # evaluate matching conditions
        self.mc_dqdq[:] = self.xs_dqdq[-1, :, :, :]

    # =========================================================================

    def calc_mc_dsdp(self):

        """
        Calculate the second order derivatives of the matching conditions
        w.r.t. the shooting variables and the parameters.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        # evaluate matching conditions
        self.mc_dsdp[:] = self.xs_dsdp[-1, :, :, :]

    # =========================================================================

    def calc_mc_dsdq(self):

        """
        Calculate the second order derivatives of the matching conditions
        w.r.t. the shooting variables and the controls.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        # evaluate matching conditions
        self.mc_dsdq[:] = self.xs_dsdq[-1, :, :, :]

    # =========================================================================

    def calc_mc_dpdq(self):

        """
        Calculate the second order derivatives of the matching conditions
        w.r.t. the parameters and the controls.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        # evaluate matching conditions
        self.mc_dpdq[:] = self.xs_dpdq[-1, :, :, :]

    # =========================================================================

    def calc_bcq(self):

        """
        Calculate the box constraints.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        # set the lower bnds for the controls
        for i in range(0, self.NU):
            self.bcq[i * self.NTS:(i + 1) * self.NTS] = (
                self.bnds[i, 0] - q[i * self.NTS: (i + 1) * self.NTS])

        # set the upper bnds for the controls
        for i in range(0, self.NU):
            self.bcq[self.NQ + i * self.NTS:self.NQ + (i + 1) * self.NTS] = (
                q[i * self.NTS:(i + 1) * self.NTS] - self.bnds[i, 1])

    # =========================================================================

    def calc_bcq_ds(self):

        """
        Calculate the first order derivatives of the box constraints
        w.r.t. the shooting variables.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        self.bcq_ds = np.zeros((2 * self.NQ, self.NS))

    # =========================================================================

    def calc_bcq_dp(self):

        """
        Calculate the first order derivatives of the box constraints
        w.r.t. the parameters.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        self.bcq_dp = np.zeros((2 * self.NQ, self.NP))

    # =========================================================================

    def calc_bcq_dq(self):

        """
        Calculate the first order derivatives of the box constraints
        w.r.t. the controls.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        # set derivatives
        self.bcq_dq[:self.NQ, :] = -np.eye(self.NQ)
        self.bcq_dq[self.NQ:, :] = np.eye(self.NQ)

    # =========================================================================

    def calc_bcq_dsds(self):

        """
        Calculate the second order derivatives of the box constraints
        w.r.t. the shooting variables and the shooting variables.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        self.bcq_dsds = np.zeros((2 * self.NQ, self.NS, self.NS))

    # =========================================================================

    def calc_bcq_dpdp(self):

        """
        Calculate the second order derivatives of the box constraints
        w.r.t. the parameters and the parameters.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        self.bcq_dpdp = np.zeros((2 * self.NQ, self.NP, self.NP))

    # =========================================================================

    def calc_bcq_dqdq(self):

        """
        Calculate the second order derivatives of the box constraints
        w.r.t. the controls and the controls.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        self.bcq_dqdq = np.zeros((2 * self.NQ, self.NQ, self.NQ))

    # =========================================================================

    def calc_bcq_dsdp(self):

        """
        Calculate the second order derivatives of the box constraints
        w.r.t. the shooting variables and the parameters.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        self.bcq_dsdp = np.zeros((2 * self.NQ, self.NS, self.NP))

    # =========================================================================

    def calc_bcq_dsdq(self):

        """
        Calculate the second order derivatives of the box constraints
        w.r.t. the shooting variables and the controls.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        self.bcq_dsdq = np.zeros((2 * self.NQ, self.NS, self.NQ))

    # =========================================================================

    def calc_bcq_dpdq(self):

        """
        Calculate the second order derivatives of the box constraints
        w.r.t. the parameters and the controls.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        self.bcq_dpdq = np.zeros((2 * self.NQ, self.NP, self.NQ))

    # =========================================================================

    def calc_bcs(self):

        """
        Calculate the constraints on the shooting nodes.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        self.bcs = -1e6 * np.ones((self.NS,))

        # evaluate the equality constraints for s
        for i in range(0, self.NX):

            if self.x0[i] is not None:
                self.bcs[i] = self.x0[i] - s[i]

            if self.xend[i] is not None:
                self.bcs[self.NX + i] = self.xend[i] - s[self.NX + i]

    # =========================================================================

    def calc_bcs_ds(self):

        """
        Calculate the first order derivatives of constraints on the
        shooting nodes w.r.t. the shooting variables.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        self.bcs_ds = np.zeros((self.NS, self.NS))

        # evaluate the derivatives
        for i in range(0, self.NX):

            if self.x0[i] is not None:
                self.bcs_ds[i, i] = -1

            if self.xend[i] is not None:
                self.bcs_ds[self.NX + i, self.NX + i] = -1

    # =========================================================================

    def calc_bcs_dp(self):

        """
        Calculate the first order derivatives of the constraints on the
        shooting nodes w.r.t. the parameters.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        self.bcs_dp = np.zeros((self.NS, self.NP))

    # =========================================================================

    def calc_bcs_dq(self):

        """
        Calculate the first order derivatives of the constraints on the
        shooting nodes w.r.t. the controls.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        self.bcs_dq = np.zeros((self.NS, self.NQ))

    # =========================================================================

    def calc_bcs_dsds(self):

        """
        Calculate the second order derivatives of the constraints on the
        shooting nodes w.r.t. the shooting variables and the shooting
        variables.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        self.bcs_dsds = np.zeros((self.NS, self.NS, self.NS))

    # =========================================================================

    def calc_bcs_dpdp(self):

        """
        Calculate the second order derivatives of the constraints on the
        shooting nodes w.r.t. the parameters and the parameters.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        self.bcs_dpdp = np.zeros((self.NS, self.NP, self.NP))

    # =========================================================================

    def calc_bcs_dqdq(self):

        """
        Calculate the second order derivatives of the constraints on the
        shooting nodes w.r.t. the controls and the controls.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        self.bcs_dqdq = np.zeros((self.NS, self.NQ, self.NQ))

    # =========================================================================

    def calc_bcs_dsdp(self):

        """
        Calculate the second order derivatives of the constraints on the
        shooting nodes w.r.t. the shooting variables and the parameters.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        self.bcs_dsdp = np.zeros((self.NS, self.NS, self.NP))

    # =========================================================================

    def calc_bcs_dsdq(self):

        """
        Calculate the second order derivatives of the constraints on the
        shooting nodes w.r.t. the shooting variables and the controls.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        self.bcs_dsdq = np.zeros((self.NS, self.NS, self.NQ))

    # =========================================================================

    def calc_bcs_dpdq(self):

        """
        Calculate the second order derivatives of the constraints on the
        shooting nodes w.r.t. the parameters and the controls.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        self.bcs_dpdq = np.zeros((self.NS, self.NP, self.NQ))

    # =========================================================================

    def calc_obj(self):

        """
        Calculate the objective function.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        self.obj = self.sign * self.xs[-1, -1]

    # =========================================================================

    def calc_obj_ds(self):

        """
        Calculate the first order derivatives of the objective function
        w.r.t. the shooting variables.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        self.obj_ds = self.sign * self.xs_ds[-1, -1, :]

    # =========================================================================

    def calc_obj_dp(self):

        """
        Calculate the first order derivatives of the objective function
        w.r.t. the parameters.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        self.obj_dp = self.sign * self.xs_dp[-1, -1, :]

    # =========================================================================

    def calc_obj_dq(self):

        """
        Calculate the first order derivatives of the objective function
        w.r.t. the controls.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        self.obj_dq = self.sign * self.xs_dq[-1, -1, :]

    # =========================================================================

    def calc_obj_dsds(self):

        """
        Calculate the second order derivatives of the objective function
        w.r.t. the shooting variables and the shooting variables.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        self.obj_dsds = self.sign * self.xs_dsds[-1, -1, :]

    # =========================================================================

    def calc_obj_dpdp(self):

        """
        Calculate the second order derivatives of the objective function
        w.r.t. the parameters and the parameters.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        self.obj_dpdp = self.sign * self.xs_dpdp[-1, -1, :]

    # =========================================================================

    def calc_obj_dqdq(self):

        """
        Calculate the second order derivatives of the objective function
        w.r.t. the controls and the controls.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        self.obj_dqdq = self.sign * self.xs_dqdq[-1, -1, :]

    # =========================================================================

    def calc_obj_dsdp(self):

        """
        Calculate the second order derivatives of the objective function
        w.r.t. the shooting variables and the parameters.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        self.obj_dsdp = self.sign * self.xs_dsdp[-1, -1, :]

    # =========================================================================

    def calc_obj_dsdq(self):

        """
        Calculate the second order derivatives of the objective function
        w.r.t. the shooting variables and the controls.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        self.obj_dsdq = self.sign * self.xs_dsdq[-1, -1, :]

    # =========================================================================

    def calc_obj_dpdq(self):

        """
        Calculate the second order derivatives of the objective function
        w.r.t. the parameters and the controls.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        p = self.p
        q = self.q
        s = self.s

        self.obj_dpdq = self.sign * self.xs_dpdq[-1, -1, :]

    # =========================================================================

    def solve(self):

        """
        Solve the OCP by calling one of the available NLP solvers.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        if self.solver == "snopt":
            self.snopt()

        if self.solver == "scipy":
            self.scipy()

    # =========================================================================

    def snopt(self):

        """
        Solve the OCP with SNOPT.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        # set initival values
        q0 = self.q
        s0 = self.s

        # =====================================================================

        def setup(inform, Prob, neF, n, iAfun, jAvar, lenA, neA, A,
                  iGfun, jGvar, lenG, neG, ObjAdd, ObjRow, xlow, xupp, Flow,
                  Fupp, x, xstate, Fmul):

            # give the problem a name.
            Prob[:3] = list('ocp')

            # assign the dimensions of the constraint Jacobian
            neF[0] = 1 + self.NC + self.NMC
            n[0] = self.NQ + self.NS

            # set the objective row
            ObjRow[0] = 1
            ObjAdd[0] = 0
            Flow[0] = -1e6
            Fupp[0] = 1e6

            # set the upper and lower bounds for the inequality constraints
            Flow[1:1 + self.NCG] = -1e6
            Fupp[1:1 + self.NCG] = 0

            # set the upper and lower bounds for the equality constraints
            Flow[1 + self.NCG:1 + self.NC] = 0
            Fupp[1 + self.NCG:1 + self.NC] = 0

            # set the upper and lower bounds for the matching conditions
            Flow[1 + self.NC:] = 0
            Fupp[1 + self.NC:] = 0

            # set the upper and lower bounds for the controls q
            for i in range(0, self.NU):
                xlow[i * self.NTS:(i + 1) * self.NTS] = self.bnds[i, 0]
                xupp[i * self.NTS:(i + 1) * self.NTS] = self.bnds[i, 1]

            # set the upper and lower bounds for the shooting variables s
            xlow[self.NQ:] = -1e6
            xupp[self.NQ:] = 1e6

            # fix the shooting variables s at the boundaries if necessary
            for i in range(0, self.NX):

                if self.x0[i] is not None:
                    xlow[self.NQ + i] = self.x0[i]
                    xupp[self.NQ + i] = self.x0[i]

                if self.xend[i] is not None:
                    xlow[self.NQ + self.NS - self.NX + i] = self.xend[i]
                    xupp[self.NQ + self.NS - self.NX + i] = self.xend[i]

            # set xstate
            xstate[:] = 0

            # set up pattern for the jacobian
            neG[0] = (self.NQ + self.NS) * (1 + self.NC + self.NMC)

            l = 0
            for i in range(0, self.NC + self.NMC + 1):
                for j in range(0, self.NQ + self.NS):
                    iGfun[l + j] = i + 1
                    jGvar[l + j] = j + 1

                l = l + self.NQ + self.NS

        # =====================================================================

        def evaluate(status, x, needF, neF, F, needG, neG, G, cu, iu, ru):

            # set controls and shooting variables
            self.q = x[:self.NQ]
            self.s = x[self.NQ:]

            if needF[0] != 0:

                # integrate and evaluate
                self.integrate()

                self.calc_obj()
                F[0] = self.obj

                if self.NG > 0:
                    self.calc_ineqc()
                    F[1:self.NCG + 1] = self.ineqc

                if self.NH > 0:
                    self.calc_eqc()
                    F[self.NCG + 1:self.NC + 1] = self.eqc

                self.calc_mc()
                F[self.NC + 1:] = self.mc

            if needG[0] != 0:

                # integrate and evaluate
                self.integrate_dq()
                self.integrate_ds()

                self.calc_obj_dq()
                self.calc_obj_ds()

                # set derviatives of objective
                G[:self.NQ] = self.obj_dq
                G[self.NQ:self.NQ + self.NS] = self.obj_ds
                l = self.NQ + self.NS

                # set derviatives of constraints
                if self.NG > 0:
                    self.calc_ineqc_dq()
                    self.calc_ineqc_ds()

                    for i in range(0, self.NCG):
                        G[l:l + self.NQ] = self.ineqc_dq[i, :]
                        G[l + self.NQ:l + self.NQ + self.NS] = \
                            self.ineqc_ds[i, :]
                        l = l + self.NQ + self.NS

                if self.NH > 0:
                    self.calc_eqc_dq()
                    self.calc_eqc_ds()

                    for i in range(0, self.NCH):
                        G[l:l + self.NQ] = self.eqc_dq[i, :]
                        G[l + self.NQ:l + self.NQ + self.NS] = \
                            self.eqc_ds[i, :]
                        l = l + self.NQ + self.NS

                self.calc_mc_dq()
                self.calc_mc_ds()

                for i in range(0, self.NMC):
                    G[l:l + self.NQ] = self.mc_dq[i, :]
                    G[l + self.NQ:l + self.NQ + self.NS] = self.mc_ds[i, :]
                    l = l + self.NQ + self.NS

        # =====================================================================

        snopt.check_memory_compatibility()
        minrw = np.zeros((1), dtype=np.int32)
        miniw = np.zeros((1), dtype=np.int32)
        mincw = np.zeros((1), dtype=np.int32)

        rw = np.zeros((1000000,), dtype=np.float64)
        iw = np.zeros((1000000,), dtype=np.int32)
        cw = np.zeros((10000,), dtype=np.character)

        Cold = np.array([0], dtype=np.int32)
        Basis = np.array([1], dtype=np.int32)
        Warm = np.array([2], dtype=np.int32)

        x = np.append(q0, s0)
        xlow = np.zeros((self.NQ + self.NS,), dtype=np.float64)
        xupp = np.zeros((self.NQ + self.NS,), dtype=np.float64)
        xmul = np.zeros((self.NQ + self.NS,), dtype=np.float64)
        F = np.zeros((1 + self.NC + self.NMC,), dtype=np.float64)
        Flow = np.zeros((1 + self.NC + self.NMC,), dtype=np.float64)
        Fupp = np.zeros((1 + self.NC + self.NMC,), dtype=np.float64)
        Fmul = np.zeros((1 + self.NC + self.NMC,), dtype=np.float64)

        ObjAdd = np.zeros((1,), dtype=np.float64)

        xstate = np.zeros((self.NQ + self.NS,), dtype=np.int32)
        Fstate = np.zeros((1 + self.NC + self.NMC,), dtype=np.int32)

        INFO = np.zeros((1,), dtype=np.int32)
        ObjRow = np.zeros((1,), dtype=np.int32)
        n = np.zeros((1,), dtype=np.int32)
        neF = np.zeros((1,), dtype=np.int32)

        lenA = np.zeros((1,), dtype=np.int32)
        lenA[0] = (self.NQ + self.NS) * (1 + self.NC + self.NMC)

        iAfun = np.zeros((lenA[0],), dtype=np.int32)
        jAvar = np.zeros((lenA[0],), dtype=np.int32)

        A = np.zeros((lenA[0],), dtype=np.float64)

        lenG = np.zeros((1,), dtype=np.int32)
        lenG[0] = (self.NQ + self.NS) * (1 + self.NC + self.NMC)

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
        Prob = np.zeros((200 * 8,), dtype=np.character)

        iSpecs = np.zeros((1,), dtype=np.int32)
        iSumm = np.zeros((1,), dtype=np.int32)
        iPrint = np.zeros((1,), dtype=np.int32)

        iSpecs[0] = 4
        iSumm[0] = 6
        iPrint[0] = 9

        printname = np.zeros((200 * 8,), dtype=np.character)
        specname  = np.zeros((200 * 8,), dtype=np.character)

        nS = np.zeros((1,), dtype=np.int32)
        nInf = np.zeros((1,), dtype=np.int32)
        sInf = np.zeros((1,), dtype=np.float64)

        # open output files using snfilewrappers.[ch] */
        specn = self.path + "/snopt.spc"
        printn = self.path + "/output/" + \
                 datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + \
                 "-snopt.out"
        specname[:len(specn)] = list(specn)
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
            print(("Warning: Trouble reading specs file %s \n" % (specname)))

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
        self.q = x[:self.NQ]
        self.s = x[self.NQ:]
        self.integrate_dq()
        self.integrate_ds()
        self.integrate()
        self.calc_obj()
        if self.NG > 0: self.calc_ineqc
        if self.NH > 0: self.calc_eqc
        self.calc_mc()
        self.calc_bcq()
        self.calc_bcs()

        # print results
        print("\n")
        print(("q_opt:", self.q, "\n"))
        print(("s_opt:", self.s, "\n"))
        print(("obj_opt:", self.obj, "\n"))
        if self.NG > 0: print(("ineqc_opt:", self.ineqc, "\n"))
        if self.NH > 0: print(("eqc_opt:", self.eqc, "\n"))
        print(("mc_opt:", self.mc, "\n"))
        print(("bcq_opt:", self.bcq, "\n"))
        print(("bcs_opt:", self.bcs, "\n"))

        self.ineqc_mul = Fmul[1:self.NCG + 1]
        self.eqc_mul = Fmul[self.NCG + 1:self.NCG + self.NCH + 1]
        self.mc_mul = Fmul[self.NCG + self.NCH + 1:]
        print(("ineqc_mul", self.ineqc_mul, "\n"))
        print(("eqc_mul", self.eqc_mul, "\n"))
        print(("mc_mul", self.mc_mul, "\n"))

    # =========================================================================

    def scipy(self):

        """
        Solve the OCP with SLSQP from scipy.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        # set bounds
        bnds = []

        # set the upper and lower bounds for the controls q
        for j in range(0, self.NU):
            for k in range(0, self.NTS):
                bnds.append((self.bnds[j, 0], self.bnds[j, 1]))

        # fix the shooting variables s at the boundaries if necessary
        for i in range(0, self.NX):
            if self.x0[i] is not None:
                bnds.append((self.x0[i], self.x0[i]))

            else:
                bnds.append((-1e6, 1e6))

        for i in range(0, self.NS - 2 * self.NX):
            bnds.append((-1e6, 1e6))

        for i in range(0, self.NX):
            if self.xend[i] is not None:
                bnds.append((self.xend[i], self.xend[i]))

            else:
                bnds.append((-1e6, 1e6))

        # =====================================================================

        def obj(x):

            # set controls and shooting variables
            self.q = x[:self.NQ]
            self.s = x[self.NQ:]

            # integrate and evaluate
            self.integrate_dq()
            self.integrate_ds()

            self.calc_obj()
            self.calc_obj_dq()
            self.calc_obj_ds()

            # allocate memory
            jac = np.zeros((self.NQ + self.NS,))

            # build jacobian
            jac[:self.NQ] = self.obj_dq
            jac[self.NQ:] = self.obj_ds

            return self.obj, jac

        # =====================================================================

        def ineqc(x):

            # set controls and shooting variables
            self.q = x[:self.NQ]
            self.s = x[self.NQ:]

            # integrate and evaluate
            self.integrate()
            self.calc_ineqc()

            return -self.ineqc

        # =====================================================================

        def ineqc_jac(x):

            # separate controls and shooting variables for readability
            q = x[:self.NQ]
            s = x[self.NQ:]

            # integrate and evaluate
            self.integrate_dq()
            self.integrate_ds()

            self.calc_ineqc_dq()
            self.calc_ineqc_ds()

            # allocate memory
            jac = np.zeros((self.NCG, self.NQ + self.NS))

            # build jacobian
            jac[:, :self.NQ] = self.ineqc_dq
            jac[:, self.NQ:] = self.ineqc_ds

            return -jac

        # =====================================================================

        def eqc(x):

            # set controls and shooting variables
            self.q = x[:self.NQ]
            self.s = x[self.NQ:]

            # integrate and evaluate
            self.integrate()
            self.calc_eqc()

            return self.eqc

        # =====================================================================

        def eqc_jac(x):

            # set controls and shooting variables
            self.q = x[:self.NQ]
            self.s = x[self.NQ:]

            # integrate and evaluate
            self.integrate_dq()
            self.integrate_ds()

            self.calc_eqc_dq()
            self.calc_eqc_ds()

            # allocate memory
            jac = np.zeros((self.NCH, self.NQ + self.NS))

            # build jacobian
            jac[:, :self.NQ] = self.eqc_dq
            jac[:, self.NQ:] = self.eqc_ds

            return jac

        # =====================================================================

        def mc(x):

            # set controls and shooting variables
            self.q = x[:self.NQ]
            self.s = x[self.NQ:]

            # integrate and evaluate
            self.integrate()
            self.calc_mc()

            return self.mc

        # =====================================================================

        def mc_jac(x):

            # set controls and shooting variables
            self.q = x[:self.NQ]
            self.s = x[self.NQ:]

            # integrate and evaluate
            self.integrate_dq()
            self.integrate_ds()

            self.calc_mc_dq()
            self.calc_mc_ds()

            # allocate memory
            jac = np.zeros((self.NMC, self.NQ + self.NS))

            # build jacobian
            jac[:, :self.NQ] = self.mc_dq
            jac[:, self.NQ:] = self.mc_ds

            return jac

        # =====================================================================

        # set initial values
        x = np.append(self.q, self.s)

        # inequality constraints only
        if self.NG > 0 and self.NH == 0:
            res = opt.minimize(obj, x, args=(), method="SLSQP", jac=True,
                               bounds=bnds, constraints=(
                               {"type":"ineq", "fun":ineqc, "jac":ineqc_jac},
                               {"type":"eq", "fun":mc, "jac":mc_jac}),
                               options={"disp":True, "iprint":2, "ftol":1e-9})

        # equality constraints only
        elif self.NG == 0 and self.NH > 0:
            res = opt.minimize(obj, x, args=(), method="SLSQP", jac=True,
                               bounds=bnds, constraints=(
                               {"type":"eq", "fun":eqc, "jac":eqc_jac},
                               {"type":"eq", "fun":mc, "jac":mc_jac}),
                               options={"disp":True, "iprint":2, "ftol":1e-9})

        # inequality and equality constraints
        elif self.NG > 0 and self.NH > 0:
            res = opt.minimize(obj, x, args=(), method="SLSQP", jac=True,
                               bounds=bnds, constraints=(
                               {"type":"ineq", "fun":ineqc, "jac":ineqc_jac},
                               {"type":"eq", "fun":eqc, "jac":eqc_jac},
                               {"type":"eq", "fun":mc, "jac":mc_jac}),
                               options={"disp":True, "iprint":2, "ftol":1e-9})

        # no additional constraints
        else:
            res = opt.minimize(obj, x, args=(), method="SLSQP", jac=True,
                               bounds=bnds, constraints=(
                               {"type":"eq", "fun":mc, "jac":mc_jac}),
                               options={"disp":True, "iprint":2, "ftol":1e-9})

        # detailed output
        print(res)

        # save results
        self.q = res.x[:self.NQ]
        self.s = res.x[self.NQ:]
        self.integrate()
        self.calc_obj()
        if self.NG > 0: self.calc_ineqc
        if self.NH > 0: self.calc_eqc
        self.calc_mc()
        self.calc_bcq()
        self.calc_bcs()

        # print results
        print("\n")
        print(("q_opt:", self.q, "\n"))
        print(("s_opt:", self.s, "\n"))
        print(("obj_opt:", self.obj, "\n"))
        if self.NG > 0: print(("ineqc_opt:", self.ineqc, "\n"))
        if self.NH > 0: print(("eqc_opt:", self.eqc, "\n"))
        print(("mc_opt:", self.mc, "\n"))
        print(("bcq_opt", self.bcq, "\n"))
        print(("bcs_opt:", self.bcs, "\n"))

    # =========================================================================

    def calc_multipliers(self):

        """
        Calculate the Lagrange multipliers post-optimally.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

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
        for i in range(0, self.NCG):
            if self.ineqc[i] >= -1e-6:
                self.ineqca.append(i)
        # self.ineqca = []
        self.NINEQCA = len(self.ineqca)

        # evaluate the active matching conditions
        for i in range(0, self.NMC):
            if self.xend[i] is not None:
                self.mca.append(i)
        self.NMCA = len(self.mca)

        # evaluate the active box constraints for q
        for i in range(0, 2 * self.NQ):
            if self.bcq[i] >= -1e-6:
                self.bcqa.append(i)
        self.NBCQA = len(self.bcqa)

        # evaluate the active constraints for s
        for i in range(0, self.NS):
            if self.bcs[i] >= -1e-6:
                self.bcsa.append(i)
        self.NBCSA = len(self.bcsa)

        self.NCA = (self.NINEQCA + self.NCH +
                    self.NMCA + self.NBCQA + self.NBCSA)

        if self.NCA > 0:

            # allocate memory for jacobian of active constraints
            J_ca = np.zeros((self.NCA, self.NQ + self.NS))

            # calculate dq and ds of all active constraints
            self.integrate_dq
            self.integrate_ds

            if self.NINEQCA > 0:

                self.calc_ineqc_dq()
                self.calc_ineqc_ds()

                J_ca[:self.NINEQCA, :self.NQ] = self.ineqc_dq[self.ineqca, :]
                J_ca[:self.NINEQCA, self.NQ:] = self.ineqc_ds[self.ineqca, :]

            l = self.NINEQCA
            if self.NH > 0:

                self.calc_eqc_ds()
                self.calc_eqc_dq()

                J_ca[l:l + self.NCH, :self.NQ] = self.eqc_dq[:, :]
                J_ca[l:l + self.NCH, self.NQ:] = self.eqc_ds[:, :]

            l = l + self.NCH
            if self.NMCA > 0:

                self.calc_mc_dq()
                self.calc_mc_ds()

                J_ca[l:l + self.NMCA, :self.NQ] = self.mc_dq[self.mca, :]
                J_ca[l:l + self.NMCA, self.NQ:] = self.mc_ds[self.mca, :]

            l = l + self.NMCA
            if self.NBCQA > 0:

                self.calc_bcq_dq()
                self.calc_bcq_ds()

                J_ca[l:l + self.NBCQA, :self.NQ] = self.bcq_dq[self.bcqa, :]
                J_ca[l:l + self.NBCQA, self.NQ:] = self.bcq_ds[self.bcqa, :]

            l = l + self.NBCQA
            if self.NBCSA > 0:

                self.calc_bcs_dq()
                self.calc_bcs_ds()

                J_ca[l:l + self.NBCSA, :self.NQ] = self.bcs_dq[self.bcsa, :]
                J_ca[l:l + self.NBCSA, self.NQ:] = self.bcs_ds[self.bcsa, :]

            # calculate jacobian of objective
            self.calc_obj_dq()
            self.calc_obj_ds()

            J_obj = np.zeros((1, self.NQ + self.NS))
            J_obj[0, :self.NQ] = self.obj_dq
            J_obj[0, self.NQ:] = self.obj_ds

            # calculate multipliers post-optimally
            R, Q = rq(J_ca)
            try:
                self.mula = (-J_obj.dot((Q.T).dot(np.linalg.inv(R)))).T
            except:
                sys.exit("ERROR: Post-optimal calculation of multipliers " +
                         "not possible. Does the OCP have redundant " +
                         "constraints?")

            # print detailed results
            print(("ineqca:", self.ineqca, "\n"))
            print(("mca:", self.mca, "\n"))
            print(("bcqa:", self.bcqa, "\n"))
            print(("bcsa:", self.bcsa, "\n"))
            print(("mula:", self.mula, "\n"))

    # =========================================================================

    def calc_sensitivities(self):

        """
        Calculate the sensitivites of the optimal controls, optimal shooting
        variables and the optimal value w.r.t. the parameters.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        # integrate
        self.integrate_dsds()
        self.integrate_dpdp()
        self.integrate_dqdq()
        self.integrate_dsdp()
        self.integrate_dsdq()
        self.integrate_dpdq()

        # evaluate deriviatives of objective function
        self.calc_obj_ds()
        self.calc_obj_dp()
        self.calc_obj_dq()
        self.calc_obj_dsds()
        self.calc_obj_dpdp()
        self.calc_obj_dqdq()
        self.calc_obj_dsdp()
        self.calc_obj_dsdq()
        self.calc_obj_dpdq()

        # evaluate derivatives of inequality constraints
        if self.NG > 0:
            self.calc_ineqc_dsds()
            self.calc_ineqc_dpdp()
            self.calc_ineqc_dqdq()
            self.calc_ineqc_dsdp()
            self.calc_ineqc_dsdq()
            self.calc_ineqc_dpdq()

        # evaluate derivatives of equality constraints
        if self.NH > 0:
            self.calc_eqc_dsds()
            self.calc_eqc_dpdp()
            self.calc_eqc_dqdq()
            self.calc_eqc_dsdp()
            self.calc_eqc_dsdq()
            self.calc_eqc_dpdq()

        # evaluate derivatives of matching conditions
        self.calc_mc_ds()
        self.calc_mc_dp()
        self.calc_mc_dq()
        self.calc_mc_dsds()
        self.calc_mc_dpdp()
        self.calc_mc_dqdq()
        self.calc_mc_dsdp()
        self.calc_mc_dsdq()
        self.calc_mc_dpdq()

        # evaluate derivatives of box constraints on q
        self.calc_bcq_ds()
        self.calc_bcq_dp()
        self.calc_bcq_dq()
        self.calc_bcq_dsds()
        self.calc_bcq_dpdp()
        self.calc_bcq_dqdq()
        self.calc_bcq_dsdp()
        self.calc_bcq_dsdq()
        self.calc_bcq_dpdq()

        # evaluate derivatives of equality constraints on s
        self.calc_bcs_ds()
        self.calc_bcs_dp()
        self.calc_bcs_dq()
        self.calc_bcs_dsds()
        self.calc_bcs_dpdp()
        self.calc_bcs_dqdq()
        self.calc_bcs_dsdp()
        self.calc_bcs_dsdq()
        self.calc_bcs_dpdq()

        # allocate memory
        self.lagrange_ds = np.zeros((self.NS))
        self.lagrange_dp = np.zeros((self.NP))
        self.lagrange_dq = np.zeros((self.NQ))
        self.lagrange_dsds = np.zeros((self.NS, self.NS))
        self.lagrange_dpdp = np.zeros((self.NP, self.NP))
        self.lagrange_dqdq = np.zeros((self.NQ, self.NQ))
        self.lagrange_dsdp = np.zeros((self.NS, self.NP))
        self.lagrange_dsdq = np.zeros((self.NS, self.NQ))
        self.lagrange_dpdq = np.zeros((self.NP, self.NQ))

        self.ca_ds = np.zeros((self.NCA, self.NS))
        self.ca_dp = np.zeros((self.NCA, self.NP))
        self.ca_dq = np.zeros((self.NCA, self.NQ))
        self.ca_dsds = np.zeros((self.NCA, self.NS, self.NS))
        self.ca_dpdp = np.zeros((self.NCA, self.NP, self.NP))
        self.ca_dqdq = np.zeros((self.NCA, self.NQ, self.NQ))
        self.ca_dsdp = np.zeros((self.NCA, self.NS, self.NP))
        self.ca_dsdq = np.zeros((self.NCA, self.NS, self.NQ))
        self.ca_dpdq = np.zeros((self.NCA, self.NP, self.NQ))

        self.kkt = np.zeros((self.NQ + self.NS + self.NCA,
                             self.NQ + self.NS + self.NCA))
        self.rhs = np.zeros((self.NQ + self.NS + self.NCA, self.NP))

        # concatenate derivatives of active contraints
        l = 0
        for i in self.ineqca:
            self.ca_ds[l, :] = self.ineqc_ds[i, :]
            self.ca_dp[l, :] = self.ineqc_dp[i, :]
            self.ca_dq[l, :] = self.ineqc_dq[i, :]
            self.ca_dsds[l, :, :] = self.ineqc_dsds[i, :, :]
            self.ca_dpdp[l, :, :] = self.ineqc_dpdp[i, :, :]
            self.ca_dqdq[l, :, :] = self.ineqc_dqdq[i, :, :]
            self.ca_dsdp[l, :, :] = self.ineqc_dsdp[i, :, :]
            self.ca_dsdq[l, :, :] = self.ineqc_dsdq[i, :, :]
            self.ca_dpdq[l, :, :] = self.ineqc_dpdq[i, :, :]
            l = l + 1

        for i in range(0, self.NCH):
            self.ca_ds[l, :] = self.eqc_ds[i, :]
            self.ca_dp[l, :] = self.eqc_dp[i, :]
            self.ca_dq[l, :] = self.eqc_dq[i, :]
            self.ca_dsds[l, :, :] = self.eqc_dsds[i, :, :]
            self.ca_dpdp[l, :, :] = self.eqc_dpdp[i, :, :]
            self.ca_dqdq[l, :, :] = self.eqc_dqdq[i, :, :]
            self.ca_dsdp[l, :, :] = self.eqc_dsdp[i, :, :]
            self.ca_dsdq[l, :, :] = self.eqc_dsdq[i, :, :]
            self.ca_dpdq[l, :, :] = self.eqc_dpdq[i, :, :]
            l = l + 1

        for i in self.mca:
            self.ca_ds[l, :] = self.mc_ds[i, :]
            self.ca_dp[l, :] = self.mc_dp[i, :]
            self.ca_dq[l, :] = self.mc_dq[i, :]
            self.ca_dsds[l, :, :] = self.mc_dsds[i, :, :]
            self.ca_dpdp[l, :, :] = self.mc_dpdp[i, :, :]
            self.ca_dqdq[l, :, :] = self.mc_dqdq[i, :, :]
            self.ca_dsdp[l, :, :] = self.mc_dsdp[i, :, :]
            self.ca_dsdq[l, :, :] = self.mc_dsdq[i, :, :]
            self.ca_dpdq[l, :, :] = self.mc_dpdq[i, :, :]
            l = l + 1

        for i in self.bcqa:
            self.ca_ds[l, :] = self.bcq_ds[i, :]
            self.ca_dp[l, :] = self.bcq_dp[i, :]
            self.ca_dq[l, :] = self.bcq_dq[i, :]
            self.ca_dsds[l, :, :] = self.bcq_dsds[i, :, :]
            self.ca_dpdp[l, :, :] = self.bcq_dpdp[i, :, :]
            self.ca_dqdq[l, :, :] = self.bcq_dqdq[i, :, :]
            self.ca_dsdp[l, :, :] = self.bcq_dsdp[i, :, :]
            self.ca_dsdq[l, :, :] = self.bcq_dsdq[i, :, :]
            self.ca_dpdq[l, :, :] = self.bcq_dpdq[i, :, :]
            l = l + 1

        for i in self.bcsa:
            self.ca_ds[l, :] = self.bcs_ds[i, :]
            self.ca_dp[l, :] = self.bcs_dp[i, :]
            self.ca_dq[l, :] = self.bcs_dq[i, :]
            self.ca_dsds[l, :, :] = self.bcs_dsds[i, :, :]
            self.ca_dpdp[l, :, :] = self.bcs_dpdp[i, :, :]
            self.ca_dqdq[l, :, :] = self.bcs_dqdq[i, :, :]
            self.ca_dsdp[l, :, :] = self.bcs_dsdp[i, :, :]
            self.ca_dsdq[l, :, :] = self.bcs_dsdq[i, :, :]
            self.ca_dpdq[l, :, :] = self.bcs_dpdq[i, :, :]
            l = l + 1

        # build up derivatives of the lagrange function
        self.lagrange_ds[:] = self.obj_ds
        self.lagrange_dp[:] = self.obj_dp
        self.lagrange_dq[:] = self.obj_dq
        self.lagrange_dsds[:] = self.obj_dsds
        self.lagrange_dpdp[:] = self.obj_dpdp
        self.lagrange_dqdq[:] = self.obj_dqdq
        self.lagrange_dsdp[:] = self.obj_dsdp
        self.lagrange_dsdq[:] = self.obj_dsdq
        self.lagrange_dpdq[:] = self.obj_dpdq

        for i in range(0, self.NCA):
            self.lagrange_ds = self.lagrange_ds + self.mula[i] * \
                               self.ca_ds[i, :]
            self.lagrange_dp = self.lagrange_dp + self.mula[i] * \
                               self.ca_dp[i, :]
            self.lagrange_dq = self.lagrange_dq + self.mula[i] * \
                               self.ca_dq[i, :]
            self.lagrange_dsds = self.lagrange_dsds + self.mula[i] * \
                                 self.ca_dsds[i, :, :]
            self.lagrange_dpdp = self.lagrange_dpdp + self.mula[i] * \
                                 self.ca_dpdp[i, :, :]
            self.lagrange_dqdq = self.lagrange_dqdq + self.mula[i] * \
                                 self.ca_dqdq[i, :, :]
            self.lagrange_dsdp = self.lagrange_dsdp + self.mula[i] * \
                                 self.ca_dsdp[i, :, :]
            self.lagrange_dsdq = self.lagrange_dsdq + self.mula[i] * \
                                 self.ca_dsdq[i, :, :]
            self.lagrange_dpdq = self.lagrange_dpdq + self.mula[i] * \
                                 self.ca_dpdq[i, :, :]

        # build KKT-matrix
        self.kkt[:self.NQ, :self.NQ] = self.lagrange_dqdq
        self.kkt[self.NQ:self.NQ + self.NS, :self.NQ] = self.lagrange_dsdq
        self.kkt[:self.NQ, self.NQ:self.NQ + self.NS] = \
            self.lagrange_dsdq.T
        self.kkt[self.NQ:self.NQ + self.NS, self.NQ:self.NQ + self.NS] = \
            self.lagrange_dsds

        self.kkt[:self.NQ, self.NQ + self.NS:] = self.ca_dq.T
        self.kkt[self.NQ:self.NQ + self.NS, self.NQ + self.NS:] = \
            self.ca_ds.T

        self.kkt[self.NQ + self.NS:self.NQ + self.NS + self.NCA, \
            :self.NQ] = self.ca_dq
        self.kkt[self.NQ + self.NS:self.NQ + self.NS + self.NCA, \
            self.NQ:self.NQ + self.NS] = self.ca_ds

        # build rhs
        self.rhs[:self.NQ, :] = self.lagrange_dpdq.T
        self.rhs[self.NQ:self.NQ + self.NS, :] = self.lagrange_dsdp
        self.rhs[self.NQ + self.NS:, :] = self.ca_dp

        # calculate first order sensitivites of q and s
        try:
            combined_dp = -np.linalg.solve(self.kkt, self.rhs)
        except:
            print(("WARNING: KKT matrix is singular! " + \
                  "Using least squares solution. \n"))
            combined_dp = -np.linalg.lstsq(self.kkt, self.rhs)[0]

        self.q_dp = combined_dp[:self.NQ, :]
        self.s_dp = combined_dp[self.NQ:self.NQ + self.NS, :]
        self.mul_dp = combined_dp[self.NQ + self.NS:, :]

        # calculate first and second order sensitivites of the optimal value
        self.obj_dp = self.lagrange_dp
        tmp1 = (combined_dp[:self.NQ + self.NS, :].T). \
               dot(self.kkt[:self.NQ + self.NS, :self.NQ + self.NS])
        tmp2 = np.hstack((self.lagrange_dpdq, self.lagrange_dsdp.T))
        self.obj_dpdp = tmp1.dot(combined_dp[:self.NQ + self.NS, :]) + \
                        self.lagrange_dpdp + \
                        2 * (tmp2.dot(combined_dp[:self.NQ + self.NS, :])).T

        print(("q_dp:", self.q_dp, "\n"))
        print(("s_dp:", self.s_dp, "\n"))
        print(("mul_dp:", self.mul_dp, "\n"))
        print(("obj_dp:", self.obj_dp, "\n"))
        print(("obj_dpdp:", self.obj_dpdp, "\n"))

    # =========================================================================

    def calc_approximations(self):

        """
        Give approximations for the optimal controls, optimal shooting
        variables and the optimal value for a new parameters.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        # use taylor approximations to estimate the new values
        if self.NCA > 0:
            self.mul_approx = self.mula + self.mul_dp.dot(self.p_new - self.p)
        else:
            self.mul_approx = []

        self.q_approx = self.q + self.q_dp.dot(self.p_new - self.p)
        self.s_approx = self.s + self.s_dp.dot(self.p_new - self.p)
        self.obj_approx_fo = self.obj + self.obj_dp.dot(self.p_new - self.p)
        self.obj_approx_so = self.obj + self.obj_dp.dot((self.p_new - self.p)) \
                             + 0.5 * ((self.p_new - self.p).T. \
                             dot(self.obj_dpdp)).dot(self.p_new - self.p)

        print(("mul_approx:", self.mul_approx, "\n"))
        print(("q_approx:", self.q_approx, "\n"))
        print(("s_approx:", self.s_approx, "\n"))
        print(("obj_approx_fo:", self.obj_approx_fo, "\n"))
        print(("obj_approx_so:", self.obj_approx_so, "\n"))

    # =========================================================================

    def restore_feasibility(self):

        """
        Restore the feasibility of the approximated optimal controls and
        optimal shooting using SNOPT.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        # set initival values
        q0 = self.q[:]
        s0 = self.s[:]

        # =====================================================================

        def setup(inform, Prob, neF, n, iAfun, jAvar, lenA, neA, A,
                  iGfun, jGvar, lenG, neG, ObjAdd, ObjRow, xlow, xupp, Flow,
                  Fupp, x, xstate, Fmul):

            # give the problem a name.
            Prob[:3] = list('ocp')

            # assign the dimensions of the constraint Jacobian
            neF[0] = 1 + self.NC + self.NMC
            n[0] = self.NQ + self.NS

            # set the objective row
            ObjRow[0] = 1
            ObjAdd[0] = 0
            Flow[0] = -1e6
            Fupp[0] = 1e6

            # set the upper and lower bounds for the inequality constraints
            Flow[1:1 + self.NCG] = -1e6
            Fupp[1:1 + self.NCG] = 0

            # set the upper and lower bounds for the equality constraints
            Flow[1 + self.NCG:1 + self.NC] = 0
            Fupp[1 + self.NCG:1 + self.NC] = 0

            # set the upper and lower bounds for the matching conditions
            Flow[1 + self.NC:] = 0
            Fupp[1 + self.NC:] = 0

            # set the upper and lower bounds for the controls q
            for i in range(0, self.NU):
                xlow[i * self.NTS:(i + 1) * self.NTS] = self.bnds[i, 0]
                xupp[i * self.NTS:(i + 1) * self.NTS] = self.bnds[i, 1]

            # set the upper and lower bounds for the shooting variables s
            xlow[self.NQ:] = -1e6
            xupp[self.NQ:] = 1e6

            # fix the shooting variables s at the boundaries if necessary
            for i in range(0, self.NX):

                if self.x0[i] is not None:
                    xlow[self.NQ + i] = self.x0[i]
                    xupp[self.NQ + i] = self.x0[i]

                if self.xend[i] is not None:
                    xlow[self.NQ + self.NS - self.NX + i] = self.xend[i]
                    xupp[self.NQ + self.NS - self.NX + i] = self.xend[i]

            # set xstate
            xstate[:] = 0

            # set up pattern for the jacobian
            neG[0] = (self.NQ + self.NS) * (1 + self.NC + self.NMC)

            l = 0
            for i in range(0, self.NC + self.NMC + 1):
                for j in range(0, self.NQ + self.NS):
                    iGfun[l + j] = i + 1
                    jGvar[l + j] = j + 1

                l = l + self.NQ + self.NS

        # =====================================================================

        def evaluate(status, x, needF, neF, F, needG, neG, G, cu, iu, ru):

            # set controls and shooting variables
            self.q = x[:self.NQ]
            self.s = x[self.NQ:]

            if needF[0] != 0:

                # integrate and evaluate
                self.integrate()

                F[0] = 0.5 * (np.linalg.norm(self.q - q0) ** 2 +
                       np.linalg.norm(self.s - s0) ** 2)

                if self.NG > 0:
                    self.calc_ineqc()
                    F[1:self.NCG + 1] = self.ineqc

                if self.NH > 0:
                    self.calc_eqc()
                    F[self.NCG + 1:self.NC + 1] = self.eqc

                self.calc_mc()
                F[self.NC + 1:] = self.mc

            if needG[0] != 0:

                # integrate and evaluate
                self.integrate_dq()
                self.integrate_ds()

                # set derviatives of objective
                G[:self.NQ] = self.q - q0
                G[self.NQ:self.NQ + self.NS] = self.s - s0
                l = self.NQ + self.NS

                # set derviatives of constraints
                if self.NG > 0:
                    self.calc_ineqc_dq()
                    self.calc_ineqc_ds()

                    for i in range(0, self.NCG):
                        G[l:l + self.NQ] = self.ineqc_dq[i, :]
                        G[l + self.NQ:l + self.NQ + self.NS] = \
                            self.ineqc_ds[i, :]
                        l = l + self.NQ + self.NS

                if self.NH > 0:
                    self.calc_eqc_dq()
                    self.calc_eqc_ds()

                    for i in range(0, self.NCH):
                        G[l:l + self.NQ] = self.eqc_dq[i, :]
                        G[l + self.NQ:l + self.NQ + self.NS] = \
                            self.eqc_ds[i, :]
                        l = l + self.NQ + self.NS

                self.calc_mc_dq()
                self.calc_mc_ds()

                for i in range(0, self.NMC):
                    G[l:l + self.NQ] = self.mc_dq[i, :]
                    G[l + self.NQ:l + self.NQ + self.NS] = self.mc_ds[i, :]
                    l = l + self.NQ + self.NS

        # =====================================================================

        snopt.check_memory_compatibility()
        minrw = np.zeros((1), dtype=np.int32)
        miniw = np.zeros((1), dtype=np.int32)
        mincw = np.zeros((1), dtype=np.int32)

        rw = np.zeros((1000000,), dtype=np.float64)
        iw = np.zeros((1000000,), dtype=np.int32)
        cw = np.zeros((10000,), dtype=np.character)

        Cold = np.array([0], dtype=np.int32)
        Basis = np.array([1], dtype=np.int32)
        Warm = np.array([2], dtype=np.int32)

        x = np.append(q0, s0)
        xlow = np.zeros((self.NQ + self.NS,), dtype=np.float64)
        xupp = np.zeros((self.NQ + self.NS,), dtype=np.float64)
        xmul = np.zeros((self.NQ + self.NS,), dtype=np.float64)
        F = np.zeros((1 + self.NC + self.NMC,), dtype=np.float64)
        Flow = np.zeros((1 + self.NC + self.NMC,), dtype=np.float64)
        Fupp = np.zeros((1 + self.NC + self.NMC,), dtype=np.float64)
        Fmul = np.zeros((1 + self.NC + self.NMC,), dtype=np.float64)

        ObjAdd = np.zeros((1,), dtype=np.float64)

        xstate = np.zeros((self.NQ + self.NS,), dtype=np.int32)
        Fstate = np.zeros((1 + self.NC + self.NMC,), dtype=np.int32)

        INFO = np.zeros((1,), dtype=np.int32)
        ObjRow = np.zeros((1,), dtype=np.int32)
        n = np.zeros((1,), dtype=np.int32)
        neF = np.zeros((1,), dtype=np.int32)

        lenA = np.zeros((1,), dtype=np.int32)
        lenA[0] = (self.NQ + self.NS) * (1 + self.NC + self.NMC)

        iAfun = np.zeros((lenA[0],), dtype=np.int32)
        jAvar = np.zeros((lenA[0],), dtype=np.int32)

        A = np.zeros((lenA[0],), dtype=np.float64)

        lenG = np.zeros((1,), dtype=np.int32)
        lenG[0] = (self.NQ + self.NS) * (1 + self.NC + self.NMC)

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
        Prob = np.zeros((200 * 8,), dtype=np.character)

        iSpecs = np.zeros((1,), dtype=np.int32)
        iSumm = np.zeros((1,), dtype=np.int32)
        iPrint = np.zeros((1,), dtype=np.int32)

        iSpecs[0] = 4
        iSumm[0] = 6
        iPrint[0] = 9

        printname = np.zeros((200 * 8,), dtype=np.character)
        specname  = np.zeros((200 * 8,), dtype=np.character)

        nS = np.zeros((1,), dtype=np.int32)
        nInf = np.zeros((1,), dtype=np.int32)
        sInf = np.zeros((1,), dtype=np.float64)

        # open output files using snfilewrappers.[ch] */
        specn = self.path + "/snopt.spc"
        printn = self.path + "/output/" + \
                 datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + \
                 "-snopt.out"
        specname[:len(specn)] = list(specn)
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
            print(("Warning: Trouble reading specs file %s \n" % (specname)))

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
        self.q = x[:self.NQ]
        self.s = x[self.NQ:]
        self.integrate_dq()
        self.integrate_ds()
        self.integrate()
        self.calc_obj()
        if self.NG > 0: self.calc_ineqc
        if self.NH > 0: self.calc_eqc
        self.calc_mc()
        self.calc_bcq()
        self.calc_bcs()

        # print results
        print("\n")
        print(("q_opt:", self.q, "\n"))
        print(("s_opt:", self.s, "\n"))
        print(("obj_opt:", self.obj, "\n"))
        if self.NG > 0: print(("ineqc_opt:", self.ineqc, "\n"))
        if self.NH > 0: print(("eqc_opt:", self.eqc, "\n"))
        print(("mc_opt:", self.mc, "\n"))
        print(("bcq_opt", self.bcq, "\n"))
        print(("bcs_opt:", self.bcs, "\n"))

        self.ineqc_mul = Fmul[1:self.NCG + 1]
        self.eqc_mul = Fmul[self.NCG + 1:self.NCG + self.NCH + 1]
        self.mc_mul = Fmul[self.NCG + self.NCH + 1:]
        print(("ineqc_mul", self.ineqc_mul, "\n"))
        print(("eqc_mul", self.eqc_mul, "\n"))
        print(("mc_mul", self.mc_mul, "\n"))

    # =========================================================================

    def calc_sensitivity_domain(self):

        """
        Give an approximation of the validity of the approxmations for the
        optimal controls and optimal shooting variables.

        CAUTION: All states or derivatives necessary for computation have
                 to be calculated beforehand.

        TODO: Validate algorithm.

        Args:
            None

        Returns:
            None

        Raises:
            None
        """

        # allocate memory
        NC = self.NCG + self.NCH + self.NMC + 2 * self.NQ + self.NS
        total = np.zeros((NC, self.NP))
        perturbations = np.zeros((self.NP, NC))

        # calculate total derivative of all constraints
        l = 0
        if self.NG > 0:
            total[l:self.NCG, :] = self.ineqc_dq.dot(self.q_dp) + self.ineqc_dp
        l = self.NCG
        if self.NH > 0:
            total[l:l + self.NCH, :] = self.eqc_dq.dot(self.q_dp) + self.eqc_dp
        l = l + self.NCH
        total[l:l + self.NMC, :] = self.mc_dq.dot(self.q_dp) + self.mc_dp
        l = l + self.NMC
        total[l:l + 2 * self.NQ, :] = self.bcq_dq.dot(self.q_dp) + self.bcq_dp
        l = l + 2 * self.NQ
        total[l:l + self.NS, :] = self.bcs_dq.dot(self.q_dp) + self.bcs_dp

        # iterate through parameters
        for j in range(0, self.NP):

            # save positions in constraints and active constraints
            l = 0
            m = 0

            # evaluate the inequality constraints
            for i in range(0, self.NCG):

                if i in self.ineqca:
                    if self.mul_dp[m, j] != 0:
                        perturbations[j, l + i] = self.p[j] - self.mula[m] / \
                            self.mul_dp[m, j]
                    else:
                        perturbations[j, l + i] = -np.sign(self.mula[m]) \
                            * np.inf
                            # / np.sign(self.mul_dp[m, j]) * np.inf
                    m = m + 1

                elif i not in self.ineqca:
                    if total[l + i, j] != 0:
                        perturbations[j, l + i] = self.p[j] - self.ineqc[i] / \
                            total[l + i, j]
                    else:
                        perturbations[j, l + i] = -np.sign(self.ineqc[i]) \
                            * np.inf
                            # / np.sign(total[l + i, j]) * np.inf

            l = self.NCG

            # evaluate the equality constraints
            for i in range(0, self.NCH):

                if self.mul_dp[m, j] != 0:
                    perturbations[j, l + i] = self.p[j] - self.mula[m] / \
                        self.mul_dp[m, j]
                else:
                    perturbations[j, l + i] = -np.sign(self.mula[m]) \
                            * np.inf
                            # / np.sign(self.mul_dp[m, j]) * np.inf
                m = m + 1

            l = self.NCG + self.NCH

            # evaluate the matching conditions
            for i in range(0, self.NMC):

                if i in self.mca:
                    if self.mul_dp[m, j] != 0:
                        perturbations[j, l + i] = self.p[j] - self.mula[m] / \
                            self.mul_dp[m, j]
                    else:
                        perturbations[j, l + i] = -np.sign(self.mula[m]) \
                            * np.inf
                            # / np.sign(self.mul_dp[m, j]) * np.inf
                    m = m + 1

                elif i not in self.mca:
                    perturbations[j, l + i] = np.inf

            l = self.NCG + self.NCH + self.NMC

            # evaluate the box constraints for q
            for i in range(0, 2 * self.NQ):

                if i in self.bcqa:
                    if self.mul_dp[m, j] != 0:
                        perturbations[j, l + i] = self.p[j] - self.mula[m] / \
                            self.mul_dp[m, j]
                    else:
                        perturbations[j, l + i] = -np.sign(self.mula[m]) \
                            * np.inf
                            # / np.sign(self.mul_dp[m, j]) * np.inf
                    m = m + 1

                elif i not in self.bcqa:
                    if total[l + i, j] != 0:
                        perturbations[j, l + i] = self.p[j] - self.bcq[i] / \
                            total[l + i, j]
                    else:
                        perturbations[j, l + i] = -np.sign(self.bcq[i]) \
                            * np.inf
                            # / np.sign(total[l + i, j]) * np.inf

            l = self.NCG + self.NCH + self.NMC + 2 * self.NQ

            # evaluate the constraints for s
            for i in range(0, self.NS):

                if i in self.bcsa:
                    if self.mul_dp[m, j] != 0:
                        perturbations[j, l + i] = self.p[j] - self.mula[m] / \
                            self.mul_dp[m, j]
                    else:
                        perturbations[j, l + i] = -np.sign(self.mula[m]) \
                            * np.inf
                            # / np.sign(self.mul_dp[m, j]) * np.inf
                    m = m + 1

                elif i not in self.bcsa:
                    if total[l + i, j] != 0:
                        perturbations[j, l + i] = self.p[j] - self.bcs[i] / \
                            total[l + i, j]
                    else:
                        perturbations[j, l + i] = -np.sign(self.bcs[i]) \
                            * np.inf
                            # / np.sign(total[l + i, j]) * np.inf

        # find maximal perturbations in each line
        self.domain = np.zeros((self.NP, 2))

        for i in range(0, self.NP):
            self.domain[i, 0] = np.min(perturbations[i, :])
            self.domain[i, 1] = np.max(perturbations[i, :])

        print(("sensitivity domain:", self.domain, "\n"))

    # =========================================================================

    def check_fo_derivatives(self):

        """
        Check first order AD-derivatives of the OCP with FD.

        CAUTION: Does not work for very large numbers, e.g. box constraints
                 with value 1e6.

        Args:
            None.

        Returns:
            None.

        Raises:
            Assertion Error.
        """

        # set h for finite differences
        h = 1.0e-8

        # set desired functions and derivatives
        functions = [["obj", 1],
                     ["mc", self.NMC],
                     ["bcs", self.NS]]
        if self.NG > 0: functions.append(["ineqc", self.NCG])
        if self.NH > 0: functions.append(["eqc", self.NCH])

        derivatives = [["s", self.NS],
                       ["p", self.NP],
                       ["q", self.NQ]]

        # first order derivatives
        for f in functions:
            for d in derivatives:

                print(("Checking " + f[0] + "_d" + d[0] + ": ..."))

                getattr(self, "integrate_d" + d[0])()
                getattr(self, "calc_" + f[0])()
                getattr(self, "calc_" + f[0] + "_d" + d[0])()
                val = getattr(self, f[0]).copy()
                ad = getattr(self, f[0] + "_d" + d[0]).copy()

                # allocate memory
                if f[1] == 1:
                    fd = np.zeros((d[1],))
                else:
                    fd = np.zeros((f[1], d[1]))

                # loop through all variables
                for i in range(0, d[1]):

                    # set directions and calculate function values
                    dot = np.zeros((d[1],))
                    dot[i] = 1

                    setattr(self, d[0], (getattr(self, d[0]) + h * dot))
                    getattr(self, "integrate_d" + d[0])()
                    getattr(self, "calc_" + f[0])()
                    comp = getattr(self, f[0]).copy()
                    setattr(self, d[0], (getattr(self, d[0]) - h * dot))

                    # approximate derivatives by finite differences
                    if f[1] == 1:
                        fd[i] = (comp - val) / h
                    else:
                        fd[:, i] = (comp - val) / h

                # compare ad and fd derivatives
                np.testing.assert_almost_equal(fd, ad, decimal=5)
                print("OK!")

 # =========================================================================

    def check_so_derivatives(self):

        """
        Check second order AD-derivatives of the OCP with FD.

        CAUTION: Requires correct first order derivatives.
        CAUTION: Does not work for very large numbers, e.g. box constraints
                 with value 1e6.

        Args:
            None.

        Returns:
            None.

        Raises:
            Assertion Error.
        """

        # set h for finite differences
        h = 1.0e-8

        # set desired functions and derivatives
        functions = [["obj", 1],
                     ["mc", self.NMC],
                     ["bcs", self.NS]]
        if self.NG > 0: functions.append(["ineqc", self.NCG])
        if self.NH > 0: functions.append(["eqc", self.NCH])

        derivatives = [["s", self.NS, "s", self.NS],
                       ["s", self.NS, "p", self.NP],
                       ["s", self.NS, "q", self.NQ],
                       ["p", self.NP, "p", self.NP],
                       ["p", self.NP, "q", self.NQ],
                       ["q", self.NQ, "q", self.NQ]]

        # second order derivatives
        for f in functions:
            for d in derivatives:

                print(("Checking " + f[0] + "_d" + d[0] + "d" + d[2] + ": ..."))

                getattr(self, "integrate_d" + d[0] + "d" + d[2])()
                getattr(self, "calc_" + f[0] + "_d" + d[0])()
                getattr(self, "calc_" + f[0] + "_d" + d[0] + "d" + d[2])()
                val = getattr(self, f[0] + "_d" + d[0]).copy()
                ad = getattr(self, f[0] + "_d" + d[0] + "d" + d[2]).copy()

                # allocate memory
                fd = np.zeros((f[1], d[1], d[3]))
                ad = ad.reshape(fd.shape)

                # loop through all variables
                for i in range(0, d[3]):

                    # set directions and calculate function values
                    dot = np.zeros((d[3],))
                    dot[i] = 1

                    setattr(self, d[2], (getattr(self, d[2]) + h * dot))
                    getattr(self, "integrate_d" + d[0] + "d" + d[2])()
                    getattr(self, "calc_" + f[0] + "_d" + d[0])()
                    comp = getattr(self, f[0] + "_d" + d[0]).copy()
                    setattr(self, d[2], (getattr(self, d[2]) - h * dot))

                    # approximate derivatives by finite differences
                    res = (comp - val) / h
                    fd[:, :, i] = res

                # compare ad and fd derivatives
                np.testing.assert_almost_equal(fd, ad, decimal=5)
                print("OK!")

# =============================================================================