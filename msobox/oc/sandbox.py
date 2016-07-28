#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Academic example of a rocket car for optimal control with single shooting.

Find a time optimal acceleration and deceleration profile of a point mass to
start in a point x(0) = x_0 and reach a final destination x(T) = x_f.
"""
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec as gridspec

from collections import (OrderedDict,)

from msobox.mf.model import (Model,)
from msobox.mf.tapenade import (Differentiator,)
from msobox.ind.rc_rk4classic import (RcRK4Classic,)
from msobox.ind.rc_explicit_euler import (RcExplicitEuler,)
from msobox.nlp.snopt_wrapper import (SNOPT,)

import warnings

# make it look beautiful
matplotlib.style.use('ggplot')

# setting print options to print all array elements
np.set_printoptions(threshold=np.nan, precision=4, linewidth=200)


# ------------------------------------------------------------------------------
class Plotter(object):

    """Brief plotting class."""

    def __init__(self, NY, NU, ind, mf, show_canvas=True):
        """Initialize plotting structure."""
        # assign values
        self.ind = ind
        self.mf = mf
        self.NY = NY
        self.NU = NU

        # plot options
        self.show_canvas = show_canvas

        # PLOTTING
        # plot integration
        plt.ion()
        self.fig = plt.figure()
        (rows, cols) = (4, 1)
        self.gs = gridspec.GridSpec(rows, cols)

        # plotting objectives
        row = 0
        col = 0
        self.ax_objctv = self.fig.add_subplot(self.gs[row, col])
        pl_lfcn, = self.ax_objctv.plot([], [], label="$L$")
        pl_mfcn, = self.ax_objctv.plot([], [], ls="", ms=10, marker="+", label="$E$")
        self.pl_objctv = [pl_lfcn, pl_mfcn]
        self.ax_objctv.legend(loc='best')
        self.ax_objctv.grid()

        # plotting nonlinear constraints
        row += 1
        col = 0
        self.ax_cnstrn = self.fig.add_subplot(self.gs[row, col])
        pl_r0, = self.ax_cnstrn.plot([], [], ls="", ms=10, marker="+", label="$r_{0}$")
        pl_rf, = self.ax_cnstrn.plot([], [], ls="", ms=10, marker="+", label="$r_{f}$")
        self.pl_cnstrn = [pl_r0, pl_rf]
        self.ax_cnstrn.legend(loc='best')
        self.ax_cnstrn.grid()

        # plotting states
        row += 1
        col = 0
        self.ax_states = self.fig.add_subplot(self.gs[row, col])
        self.ind.ts_plt = []
        self.ind.ys_plt = []
        self.pl_states = []
        for iy in range(self.NY):
            line, = self.ax_states.plot(self.ind.ts_plt, self.ind.ys_plt, label="$x_{}$".format(iy))
            self.pl_states.append(line)
        self.ax_states.legend(loc='best')
        self.ax_states.grid()

        # plotting controls
        row += 1
        col = 0
        self.ax_cntrls = self.fig.add_subplot(self.gs[row, col])
        self.pl_cntrls = []
        for iu in range(self.NU):
            step, = self.ax_cntrls.step([], [], marker="x", where='post', label="$u_{}$".format(iu))
            self.pl_cntrls.append(step)

        self.ax_cntrls.legend(loc='best')
        self.ax_cntrls.grid()

        self.fig.tight_layout()

        if self.show_canvas:
            self.fig.show()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plt.pause(1e-8)

    def update(self, F, x):
        """Update plots in structure."""
        # unpack variables
        y0 = x[:NY]  # initial values
        q = x[NY:-NP].reshape(NMS, NU)  # pwc. controls
        p = x[-NP:]  # parameters

        # update objective
        [pl_lfcn, pl_mfcn] = self.pl_objctv

        pl_lfcn.set_xdata([])
        pl_lfcn.set_ydata([])

        pl_mfcn.set_xdata(ts[-1])
        m = F[0]
        pl_mfcn.set_ydata(m)

        self.ax_objctv.set_xlim([ts[0] - 0.1, ts[-1] + 0.1])
        self.ax_objctv.set_ylim([0.0 - 0.1, m + 0.1])
        # self.ax_objctv.relim()
        # self.ax_objctv.autoscale_view(True, True, True)
        self.ax_objctv.grid()

        # update constraints
        [pl_r0, pl_rf] = self.pl_cnstrn

        r0 = F[1:1 + NY]
        pl_r0.set_xdata([ts[0]]*r0.size)
        pl_r0.set_ydata(r0)

        rf = F[1 + NY:1 + 2*NY]
        pl_rf.set_xdata([ts[-1]]*rf.size)
        pl_rf.set_ydata(rf)

        self.ax_cnstrn.set_xlim([ts[0] - 0.1, ts[-1] + 0.1])
        ymin = np.min([r0.min(), rf.min()])
        ymax = np.max([r0.max(), rf.max()])
        self.ax_cnstrn.set_ylim([ymin - 0.1, ymax + 0.1])
        # ax_cnstrn.relim()
        # ax_cnstrn.autoscale_view(True, True, True)
        self.ax_cnstrn.grid()

        # update states
        ind_ts_plt = np.asarray(self.ind.ts_plt)
        ind_ys_plt = np.asarray(self.ind.ys_plt)
        for iy in range(self.NY):
            line = self.pl_states[iy]
            line.set_xdata(ind_ts_plt)
            line.set_ydata(ind_ys_plt[:, iy])

        self.ax_states.set_xlim([ts[0] - 0.1, ts[-1] + 0.1])
        ymin = np.min(ind_ys_plt)
        ymax = np.max(ind_ys_plt)
        self.ax_states.set_ylim([ymin - 0.1, ymax + 0.1])
        # self.ax_states.relim()
        # self.ax_states.autoscale_view(True, True, True)
        self.ax_states.grid()

        # update controls
        for iu in range(self.NU):
            step = self.pl_cntrls[iu]
            step.set_xdata(ts[:-1])
            step.set_ydata(q)

        self.ax_cntrls.set_xlim([ts[0] - 0.1, ts[-1] + 0.1])
        ymin = np.min(q)
        ymax = np.max(q)
        self.ax_cntrls.set_ylim([ymin - 0.1, ymax + 0.1])
        # ax_cntrls.relim()
        # ax_cntrls.autoscale_view(True, True, True)
        self.ax_cntrls.grid()

        # enable tight layout
        self.fig.tight_layout()

        # update plots
        if self.show_canvas:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plt.pause(1e-8)

    def savefig(self, **kwargs):
        _d = {
            "fname": "ss_final",
            "fmt": "pdf",
            "dpi": 200,
        }
        _d.update(kwargs)
        self.fig.savefig('{fname}.{fmt}'.format(**_d), **_d)


# ------------------------------------------------------------------------------
class MF(object):

    """Model function definitions of the rocket car example."""

    def lfcn(self, l, t, x, u):
        """
        Lagrange objective of the rocket car example, which is one possible
        formulation of the minimal time optimal control problem, in the form
        of::

            min int_0^T L(t, x, u) dt = int_0^T 1 dt = T
             T

        .. NOTE::
            add an additional state to efficiently solve the objective.

        .. NOTE::
            this has to be rescaled as well

        """
        l[0] = 1.0

    def lfcn_dot(self, l, l_dot, t, x, x_dot, u, u_dot):
        """Implement first-order forward derivative of lfcn."""
        l_dot[0, :] = 0.0
        l[0] = 1.0

    def mfcn(self, m, t, x, p):
        """
        Mayer objective of the rocket car example, which is one possible
        formulation of the minimal time optimal control problem, in the form
        of::

            min E(T, x(T), p) = T
             T

        .. NOTE::
            this has to be rescaled as well

        """
        m[0] = p[0]

    def mfcn_dot(self, m, m_dot, t, x, x_dot, p, p_dot):
        """Implement first-order forward derivative of mfcn."""
        m_dot[0, :] = p_dot[0, :]
        m[0] = p[0]

    def ffcn(self, f, t, x, u):
        """
        Model of the rocket car with time transformation on unit interval.

        The reduced order model of the point mass with rocket engine given by
        the control -1 <= u(t) <= 1 for all t in [0, T] is give by::

            x_dot[0] = x[1],
            x_dot[1] = u[0],

        """
        f[0] = x[1]
        f[1] = u[0]

    def ffcn_dot(self, f, f_dot, t, x, x_dot, u, u_dot):
        """
        Implement first-order forward derivative.
        """
        f_dot[0, :] = x_dot[1, :]
        f_dot[1, :] = u_dot[0, :]

        f[0] = x[1]
        f[1] = u[0]

    def ffcnp(self, f, t, x, u, p):
        """
        Model of the rocket car with time transformation on unit interval.

        The reduced order model of the point mass with rocket engine given by
        the control -1 <= u(t) <= 1 for all t in [0, T] is give by::

            x_dot[0] = x[1],
            x_dot[1] = u[0],

        """
        f[0] = x[1]*p[0]
        f[1] = u[0]*p[0]

    def ffcnp_dot(self, f, f_d, t, x, x_d, u, u_d, p, p_d):
        """
        Model of the rocket car with time transformation on unit interval.

        The reduced order model of the point mass with rocket engine given by
        the control -1 <= u(t) <= 1 for all t in [0, T] is give by::

            x_dot[0] = x[1],
            x_dot[1] = u[0],

        """
        f_d[0, :] = x_d[1]*p[0] + x[1]*p_d[0]
        f_d[1, :] = u_d[0]*p[0] + u[0]*p_d[0]

        f[0] = x[1]*p[0]
        f[1] = u[0]*p[0]

    def rfcn_0(self, r, t, x):
        """
        Endpoint constraint ensuring the reaching of a fixed end position and
        velocity, given by::
        """
        r[0] = x[0]
        r[1] = x[1]

    def rfcn_0_dot(self, r, r_dot, t, x, x_dot):
        r_dot[0, :] = x_dot[0, :]
        r_dot[1, :] = x_dot[1, :]

        r[0] = x[0]
        r[1] = x[1]

    def rfcn_f(self, r, t, x):
        """
        Endpoint constraint ensuring the reaching of a fixed end position and
        velocity, given by::
        """
        r[0] = x[0] - 1.0
        r[1] = x[1]

        return None

    def rfcn_f_dot(self, r, r_dot, t, x, x_dot):
        r_dot[0, :] = x_dot[0, :]
        r_dot[1, :] = x_dot[1, :]

        r[0] = x[0] - 1.0
        r[1] = x[1]

        return None


# ------------------------------------------------------------------------------
def eval_F(F, x, ind, mf):
    # empty plot lists
    ind.ts_plt = []
    ind.ys_plt = []

    # unpack variables
    y0 = x[:NY]  # initial values
    q = x[NY:-NP].reshape(NMS, NU)  # pwc. controls
    p = x[-NP:]  # parameters

    # evaluate start point constraint
    r0 = F[1:1+NY]
    mf.rfcn_0(r0, ts[0:1], y0)

    # integration loop from shooting node to shooting node
    for ci in range(NMS):
        tis = np.linspace(ts[ci], ts[ci+1], NIS, endpoint=True)
        ind.init_zo_forward(ts=tis)

        # reverse communication loop
        while True:
            # print "STATE: ", ind.STATE, "(", ind.j, "/", ind.NTS,")"

            if ind.STATE == 'provide_x0':
                if ci == 0:
                    ind.x = y0
                else:
                    # NOTE initial variable is used
                    pass

            if ind.STATE == 'provide_f':
                # FIXME use lfcn formulation
                # mf.lfcn(ind.f[:1], ind.t, ind.x, p, q[ci])
                # mf.ffcn(ind.f, ind.t, ind.x, q[ci])
                mf.ffcnp(ind.f, ind.t, ind.x, q[ci], p)
                #NOTE: rescale to unit interval [0, 1] by parameter p::
                # ind.f *= p[0]

            if ind.STATE == 'plot':
                # NOTE list contains pointer to memory therefore copy arrays
                ind.ts_plt.append(ind.t.copy())
                ind.ys_plt.append(ind.x.copy())

            if ind.STATE == 'finished':
                # print 'done'
                break

            ind.step_zo_forward()
    else:  # end of for loop
        pass

    # evaluate Mayer-type objective
    m = F[0:1]
    mf.mfcn(m, ts[-1:], ind.x, p)

    # evaluate end point constraint
    rf = F[1+NY:1+2*NY]
    mf.rfcn_f(rf, ts[-1:], ind.x)

    # print "m:  ", m
    # print "r0: ", r0
    # print "rf: ", rf

    return 0


# ------------------------------------------------------------------------------
def eval_G(F, G, x, x_d, ind, mf):
    # empty plot lists
    ind.ts_plt = []
    ind.ys_plt = []

    # unpack variables
    y0 = x[:NY]  # initial values
    q = x[NY:-NP].reshape(NMS, NU)  # pwc. controls
    p = x[-NP:]  # parameters

    # unpack directions
    P = x_d.shape[1]
    y0_d = x_d[:NY, :]  # initial values
    q_d = x_d[NY:-NP, :].reshape(NMS, NU, P)  # pwc. controls
    p_d = x_d[-NP:, :]  # parameters

    # reshape G
    if G.size == F.size * x.size:
        G = G.reshape([F.size, x.size])
    else:
        return 0

    if not hasattr(ind, "f_d"):
        ind.f_d = np.zeros(ind._f.shape + (P,), dtype=float)

    if not hasattr(ind, "x_d"):
        ind.x_d = np.zeros(ind._x.shape + (P,), dtype=float)

    if not hasattr(ind, "_x_d"):
        ind._x_d = np.zeros(ind._x.shape + (P,), dtype=float)

    # evaluate start point constraint
    r0_d = G[1:1+NY, :]
    r0 = F[1:1+NY]
    mf.rfcn_0_dot(r0, r0_d, ts[0:1], y0, y0_d)
    tmp = r0.copy()
    mf.rfcn_0(r0, ts[0:1], y0)
    # # print "r0 - r0: ", tmp - r0
    # print "r0_d:\n", r0_d

    # integration loop from shooting node to shooting node
    for ci in range(NMS):
        tis = np.linspace(ts[ci], ts[ci+1], NIS, endpoint=True)
        ind.init_fo_forward(ts=tis)

        # reverse communication loop
        while True:
            # print "STATE: ", ind.STATE, "(", ind.j, "/", ind.NTS,")"

            if ind.STATE == 'provide_x0':
                if ci == 0:
                    ind.x_d = y0_d.copy()
                    ind.x = y0
                else:
                    # NOTE initial variable is used
                    pass

            if ind.STATE == 'provide_f_dot':
                # FIXME use lfcn formulation
                # mf.lfcn(ind.f, ind.t, ind.x, p, q[ci])
                # mf.ffcn(ind.f, ind.t, ind.x, q[ci])
                # mf.ffcn_dot(
                #     ind.f, ind.f_d, ind.t, ind.x, ind.x_d, q[ci], q_d[ci]
                # )
                mf.ffcnp_dot(
                    ind.f, ind.f_d, ind.t, ind.x, ind.x_d, q[ci], q_d[ci], p, p_d
                )
                #NOTE: rescale to unit interval [0, 1] by parameter p::
                # ind.f_d = ind.f[:, np.newaxis] * p_d[0] + ind.f_d * p[0]
                # ind.f *= p[0]

            if ind.STATE == 'plot':
                # NOTE list contains pointer to memory therefore copy arrays
                ind.ts_plt.append(ind.t.copy())
                ind.ys_plt.append(ind.x.copy())

            if ind.STATE == 'finished':
                # print 'done'
                break

            ind.step_fo_forward()
    else:  # end of for loop
        pass

    # evaluate Mayer-type objective
    m_d = G[0:1, :]
    m = F[0:1]
    mf.mfcn_dot(m, m_d, ts[-1:], ind.x, ind.x_d, p, p_d)
    tmp = m.copy()
    mf.mfcn(m, ts[-1:], ind.x, p)
    # print "m - m: ", tmp - m
    # print "m_d:\n", m_d

    # evaluate end point constraint
    rf_d = G[1+NY:1+2*NY, :]
    rf = F[1+NY:1+2*NY]
    mf.rfcn_f_dot(rf, rf_d, ts[-1:], ind.x, ind.x_d)
    tmp = rf.copy()
    mf.rfcn_f(rf, ts[-1:], ind.x)
    # print "rf - rf: ", tmp - rf
    # print "rf_d:\n", rf_d

    # print "m:  ", m
    # print "r0: ", r0
    # print "rf: ", rf

    # print "m_d:  \n", m_d
    # print "r0_d: \n", r0_d
    # print "rf_d: \n", rf_d

    return 0


# ------------------------------------------------------------------------------
if __name__ == "__main__":

    # define single shooting grid for control discretization
    NTS = 11
    NIS = 100  # take NIS intermediate steps during integration
    ts = np.linspace(0, 1, NTS, endpoint=True)

    # initial values, controls and parameters
    NMS = NTS - 1
    NY = 2
    NU = 1
    NP = 1

    # number of shooting variables
    # variables
    # V = [x0, u0, u1, ..., u_NMS, p]
    NV = NY + NMS*NU + NP
    NC = NY + NY  # for start and endpoint constraints

    # setup NLP problem using SNOPT
    sn = SNOPT(NV=NV, NC=NC)
    # F = np.zeros([1 + NC])
    # G = np.zeros([1 + NC, NV])

    # SETUP OF SHOOTING NLP
    # retrieve variables from NLP solver
    # x = np.zeros(NV)
    x = sn.x
    y0 = x[:NY]  # initial values
    q = x[NY:-NP].reshape(NMS, NU)  # pwc. controls
    p  = x[-NP:]  # parameters

    y0[...] = [0.0, 0.0]  # define initial values

    # initialize with bang bang solution
    q[:NTS/2, :] = 1.0
    q[NTS/2:, :] = -1.0

    # time scaling of one
    p[...] = [1.5]

    print "y0: \n", y0
    print "q: \n", q
    print "p: \n", p

    # Gradient calculation
    # setup directions for directional derivatives
    P = NV
    x_d = np.eye(NV)
    y0_d = x_d[:NY, :]  # initial values
    q_d = x_d[NY:-NP, :].reshape(NMS, NU, P)  # pwc. controls
    p_d = x_d[-NP:, :]  # parameters

    print "y0_d: \n", y0_d
    print "q_d: \n", q_d
    print "p_d: \n", p_d

    # define Gradient in sparse coordinate form (i, j, val)
    k = 0
    for i in range(NC + 1):
        for j in range(NV):
            sn.iGfun[k] = i  # row coordinate of G[k]
            sn.jGvar[k] = j  # col coordinate of G[k]
            k += 1

    sn.neG[0] = k

    # print "iGfun:\n", sn.iGfun
    # print "jGvar:\n", sn.jGvar

    # set bounds on variables
    v_lo = sn.xlow
    y0_lo = v_lo[:NY]  # initial values
    q_lo = v_lo[NY:-NP].reshape(NMS, NU)  # pwc. controls
    p_lo = v_lo[-NP:]  # parameters

    v_up = sn.xupp
    y0_up = v_up[:NY]  # initial values
    q_up = v_up[NY:-NP].reshape(NMS, NU)  # pwc. controls
    p_up = v_up[-NP:]  # parameters

    y0_lo[...] = 0.0
    y0_up[...] = 0.0

    q_lo[...] = -1.0
    q_up[...] = 1.0

    p_lo[...] = 0.1
    p_up[...] = 5.0

    # set bounds on nonlinear constraints
    sn.Flow[1:1+NY] = [0.0, 0.0]
    sn.Fupp[1:1+NY] = [0.0, 0.0]

    sn.Flow[1+NY:1+2*NY] = [0.0, 0.0]
    sn.Fupp[1+NY:1+2*NY] = [0.0, 0.0]

    # EVALUATION OF SHOOTING NLP
    # define model functions
    mf = MF()

    # instantiate integrator
    # ind = RcExplicitEuler(x0=y0.copy())
    ind = RcRK4Classic(x0=y0.copy())

    # setup plotting routines
    plot = Plotter(NY=NY, NU=NU, ind=ind, mf=mf)

    # add clobal counter for number of calls
    cnt = 0
    # -------------------------------------------------------------------------
    # define shooting problem in an SNOPT compatible way
    def evaluate(status, x, needF, nF, F, needG, neG, G, cu, iu, ru):
        """Function to implement that is used by SNOPT to solve the problem."""
        global cnt
        # print "iteration: ", cnt
        F_evaluated = False

        # set status flag
        status[0] = 0

        # erase current plotting data
        ind.ts_plt = []
        ind.ys_plt = []

        # unpack variables
        y0 = x[:NY]  # initial values
        q = x[NY:-NP].reshape(NMS, NU)  # pwc. controls
        p = x[-NP:]  # parameters

        # evaluate gradient of problem
        if needG[0] != 0:
            # print "In need G!"
            ret = eval_G(F, G, x, x_d, ind, mf)
            if ret == 0:
                # print "F evaluated!"
                F_evaluated = True

        # evaluate F only when explicitly needed
        if needF[0] != 0:
            # print "In need F!"
            eval_F(F, x, ind, mf)

        cnt += 1

        return None
    # -------------------------------------------------------------------------

    # run SNOPT
    sn.set_derivative_mode(1)  # mode 0 means use finite differences
    if sn.deropt[0] == 0:
        print "estimate jacobian structure"
        print "-"*30
        sn.calc_jacobian(evaluate)
        print "iGfun = \n", sn.iGfun
        print "jGvar = \n", sn.jGvar
        print ""

    # calculate derivative with finite differences
    F = np.zeros([1 + NC])
    G = np.zeros([1 + NC, NV])
    # 1) nominal evaluation
    eval_F(F, x, ind, mf)
    F_tmp = F.copy()
    for j in range(P):
        eval_F(G[:, j], x + 1e-8*x_d[:, j], ind, mf)
        G[:, j] = (G[:, j] - F_tmp)/1e-8
    G_fd = G.copy()

    print "G_fd = \n", G
    eval_G(F, G, x, x_d, ind, mf)
    G_ad = G.copy()
    print "G_ad = \n", G_ad
    print "err = \n", G_ad - G_fd

    print "solve SQP problem"
    print "-"*30
    sn.sqp_step(evaluate)
    print ""

    print "evaluate F for plotting"
    F = sn.F
    x = sn.x

    y0 = x[:NY]  # initial values
    q = x[NY:-NP].reshape(NMS, NU)  # pwc. controls
    p  = x[-NP:]  # parameters

    print "y0: \n", y0
    print "q: \n", q
    print "p: \n", p

    eval_F(F, x, ind, mf)
    plot.update(F, x)
    plot.savefig(fname="final_ss")
    print ""


# ------------------------------------------------------------------------------
