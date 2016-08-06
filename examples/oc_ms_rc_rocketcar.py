#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rocket car example implemented using reverse communication."""

# system imports
import os
import sys
import json
import pprint
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec as gridspec

from collections import (OrderedDict,)

from msobox.mf.model import (Model,)
from msobox.mf.tapenade import (Differentiator,)
from msobox.ind.rc_explicit_euler import (RcExplicitEuler,)
from msobox.nlp.snopt_wrapper import (SNOPT,)

import warnings

# make it look beautiful
matplotlib.style.use('ggplot')

# setting print options to print all array elements
np.set_printoptions(threshold=np.nan, precision=2, linewidth=200)


# ------------------------------------------------------------------------------
def get_msobox_examples_path():
    """Get root directory of msobox examples."""
    DIR = os.path.dirname(os.path.abspath(__file__))
    DIR = os.path.dirname(DIR)
    DIR = os.path.join(DIR, "examples")
    return DIR


# ------------------------------------------------------------------------------
def load_fortran_example_model_by_name(p_name, generate_so=True):
    """Compile and load fortran model by name form msobox examples."""
    # retrieve msobox examples path
    DIR = get_msobox_examples_path()
    DIR = os.path.join(DIR, "fortran")

    # load json model description
    p_path = os.path.join(DIR, p_name)
    ds = "{}"
    with open(os.path.join(p_path, "ds.json"), "r") as f:
        ds = json.load(f)

    # differentiate model functions
    if generate_so:
        Differentiator(p_path, ds=ds)

    # set up path to shared library
    so_path = os.path.join(p_path, "gen", "libproblem.so")

    return so_path, ds


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
        pl_mfcn, = self.ax_objctv.plot(
            [], [], ls="", ms=10, marker="+", label="$E$"
        )
        self.pl_objctv = [pl_lfcn, pl_mfcn]
        self.ax_objctv.legend(loc='best')
        self.ax_objctv.grid()

        # plotting nonlinear constraints
        row += 1
        col = 0
        self.ax_cnstrn = self.fig.add_subplot(self.gs[row, col])
        pl_r0, = self.ax_cnstrn.plot(
            [], [], ls="", ms=10, marker="+", label="$r_{0}$"
        )
        pl_rf, = self.ax_cnstrn.plot(
            [], [], ls="", ms=10, marker="+", label="$r_{f}$"
        )
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
            line, = self.ax_states.plot(
                self.ind.ts_plt, self.ind.ys_plt, label="$x_{}$".format(iy)
            )
            self.pl_states.append(line)
        self.ax_states.legend(loc='best')
        self.ax_states.grid()

        # plotting controls
        row += 1
        col = 0
        self.ax_cntrls = self.fig.add_subplot(self.gs[row, col])
        self.pl_cntrls = []
        for iu in range(self.NU):
            step, = self.ax_cntrls.step(
                [], [], marker="x", where='post', label="$u_{}$".format(iu)
            )
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
        y = x[:NTS*NY].reshape(NTS, NY)  # shooting variables
        q = x[NTS*NY:-NP].reshape(NMS, NU)  # pwc. controls
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
        """Save current canvas as file."""
        _d = {
            "fname": "ss_final",
            "fmt": "pdf",
            "dpi": 200,
        }
        _d.update(kwargs)
        self.fig.savefig('{fname}.{fmt}'.format(**_d), **_d)

    def __del__(self):
        plt.close(self.fig)


# ------------------------------------------------------------------------------
def eval_F(F, x, ind, mf):
    """Evaluate current optimal control problem."""
    # empty plot lists
    ind.ts_plt = []
    ind.ys_plt = []

    # unpack variables
    y = x[:NTS*NY].reshape(NTS, NY)  # shooting variables
    q = x[NTS*NY:-NP].reshape(NMS, NU)  # pwc. controls
    p = x[-NP:]  # parameters

    # evaluate start point constraint
    r0 = F[1:1+NY]
    mf.rfcn_0(r0, ts[0:1], y[0, :])

    r_mc = F[1+NY:-NY].reshape(NMS, NY)
    # integration loop from shooting node to shooting node
    for ci in range(NMS):
        tis = np.linspace(ts[ci], ts[ci+1], NIS, endpoint=True)
        ind.init_zo_forward(ts=tis)

        # reverse communication loop
        while True:
            if ind.STATE == 'provide_x0':
                ind.x = y[ci, :]

            if ind.STATE == 'provide_f':
                # evaluate ffcn with current time, state and control to provide f
                mf.ffcn(ind.f, ind.t, ind.x, q[ci])
                #NOTE: rescale to unit interval [0, 1] by parameter p::
                ind.f *= p[0]

            if ind.STATE == 'plot':
                # NOTE list contains pointer to memory therefore copy arrays
                ind.ts_plt.append(ind.t.copy())
                ind.ys_plt.append(ind.x.copy())

            if ind.STATE == 'finished':
                # print 'done'
                break

            ind.step_zo_forward()
        # END WHILE LOOP

        # evaluate multiple shooting matching conditions
        r_mc[ci, :] = y[ci+1, :] - ind.x

    else:  # end of for loop
        pass

    # evaluate Mayer-type objective
    m = F[0:1]
    mf.mfcn(m, ts[-1:], y[-1, :], p)

    # evaluate end point constraint
    rf = F[-NY:]
    mf.rfcn_f(rf, ts[-1:], y[-1, :])

    return 0


# ------------------------------------------------------------------------------
def eval_G(F, G, x, x_d, ind, mf):
    """Evaluate current optimal control problem and gradient."""
    # empty plot lists
    ind.ts_plt = []
    ind.ys_plt = []

    # unpack variables
    y = x[:NTS*NY].reshape(NTS, NY)  # shooting variables
    q = x[NTS*NY:-NP].reshape(NMS, NU)  # pwc. controls
    p = x[-NP:]  # parameters

    # unpack directions
    P = x_d.shape[1]
    y_d = x_d[:NTS*NY, :].reshape(NTS, NY, P)  # initial values
    q_d = x_d[NTS*NY:-NP, :].reshape(NMS, NU, P)  # pwc. controls
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
    mf.rfcn_0_dot(r0, r0_d, ts[0:1], y[0, :], y_d[0, :])

    r_mc_d = G[1+NY:-NY, :].reshape(NMS, NY, P)
    r_mc = F[1+NY:-NY].reshape(NMS, NY)
    # integration loop from shooting node to shooting node
    for ci in range(NMS):
        tis = np.linspace(ts[ci], ts[ci+1], NIS, endpoint=True)
        ind.init_fo_forward(ts=tis)

        # reverse communication loop
        while True:
            # print "STATE: ", ind.STATE, "(", ind.j, "/", ind.NTS,")"

            if ind.STATE == 'provide_x0':
                ind.x_d = y_d[ci, :].copy()
                ind.x = y[ci, :]

            if ind.STATE == 'provide_f_dot':
                mf.ffcn_dot(
                    ind.f, ind.f_d,
                    ind.t,
                    ind.x, ind.x_d,
                    q[ci], q_d[ci],
                )
                #NOTE: rescale to unit interval [0, 1] by parameter p::
                ind.f_d = ind.f[:, np.newaxis] * p_d[0] + ind.f_d * p[0]
                ind.f *= p[0]

            if ind.STATE == 'plot':
                # NOTE list contains pointer to memory therefore copy arrays
                ind.ts_plt.append(ind.t.copy())
                ind.ys_plt.append(ind.x.copy())

            if ind.STATE == 'finished':
                # print 'done'
                break

            ind.step_fo_forward()
        # END WHILE LOOP

        # evaluate multiple shooting matching conditions
        # TODO check gradient matrix visually
        r_mc_d[ci, ...] = 0.0
        r_mc_d[ci, :, :] = -ind.x_d
        r_mc_d[ci, :, (ci+1)*NY:(ci+2)*NY] = np.eye(NY)
        r_mc[ci, :] = y[ci+1, :] - ind.x
    else:  # end of for loop
        pass

    # evaluate Mayer-type objective
    m_d = G[0:1, :]
    m = F[0:1]
    mf.mfcn_dot(m, m_d, ts[-1:], y[-1, :], y_d[-1, :], p, p_d)

    # evaluate end point constraint
    rf_d = G[-NY:, :]
    rf = F[-NY:]
    mf.rfcn_f_dot(rf, rf_d, ts[-1:], y[-1, :], y_d[-1, :])

    return 0


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    The work flow for setting up a single-shooting optimal control problem with
    SNOPT7 is given by:

    1) retrieve optimization variables from SNOPT7 and specify
        - initial values x0
        - time discretization for controls t0, t1, ..., tn
        - controls on the time grid q0, q1, ..., qn
        - free parameters p
    2) load model functions lfcn, mfcn, ffcn, rfcn, etc.
    3) set up nominal evaluation F(...) of the OCP using
        - the model function interface for respective nominal function calls
        - using an respective integrator from the IND module
    4) set up gradient evaluation G(...) following the structure of 3), i.e.
        - the model function interface for the respective calls to nominal and
          first-order forward function evaluation
        - using an respective integrator from the IND module to propagate the
          sensitivity information
    5) solve the NLP by using the SNOPT7 interface
    *) plot the final solution using matplotlib
        - for real-time plotting instantiate plotter instance right after 1)
    """
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
    # V = [x0, x1, ..., xNTS, u0, u1, ..., u_NMS, p]
    NV = NTS*NY + NMS*NU + NP
    NC = (
        NMS*NY  # for matching conditions on shooting intervals
        +
        NY + NY  # for start and endpoint constraints
    )

    # setup NLP problem using SNOPT
    sn = SNOPT(NV=NV, NC=NC)

    # SETUP OF SHOOTING NLP
    # retrieve variables from NLP solver
    x = sn.x
    y = x[:NTS*NY].reshape(NTS, NY)  # shooting variables
    q = x[NTS*NY:-NP].reshape(NMS, NU)  # pwc. controls
    p = x[-NP:]  # parameters
    assert x.size == y.size + q.size + p.size

    # define initial positions
    # y[:, 0] = np.linspace(0, 1, NTS, endpoint=True)  # define initial values
    # y[:, 1] = np.linspace(0, 1, NTS, endpoint=True)  # define initial values

    # define initial velocities
    # y[:NTS/2, 1] = np.linspace(0, 1, NTS/2, endpoint=True)
    # y[NTS/2:, 1] = np.linspace(1, 0, NTS - NTS/2, endpoint=True)

    # initialize with bang bang solution
    q[:NTS/2, :] = 1.0
    q[NTS/2:, :] = -1.0

    # time scaling of one
    p[...] = [1.5]

    print "y: \n", y
    print "q: \n", q
    print "p: \n", p
    sn.xstate[...] = sn.x

    # Gradient calculation
    # setup directions for directional derivatives
    P = NV
    x_d = np.eye(NV)
    y_d = x_d[:NTS*NY, :].reshape(NTS, NY, P)  # initial values
    q_d = x_d[NTS*NY:-NP, :].reshape(NMS, NU, P)  # pwc. controls
    p_d = x_d[-NP:, :]  # parameters
    assert x_d.size == y_d.size + q_d.size + p_d.size

    # define Gradient in sparse coordinate form (i, j, val)
    # TODO use sparsity in SNOPT7
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
    y_lo = v_lo[:NTS*NY].reshape(NTS, NY)  # shooting variables
    q_lo = v_lo[NTS*NY:-NP].reshape(NMS, NU)  # pwc. controls
    p_lo = v_lo[-NP:]  # parameters
    assert v_lo.size == y_lo.size + q_lo.size + p_lo.size

    v_up = sn.xupp
    y_up = v_up[:NTS*NY].reshape(NTS, NY)  # shooting variables
    q_up = v_up[NTS*NY:-NP].reshape(NMS, NU)  # pwc. controls
    p_up = v_up[-NP:]  # parameters
    assert v_up.size == y_up.size + q_up.size + p_up.size

    q_lo[...] = -1.0
    q_up[...] = 1.0

    p_lo[...] = 0.1
    p_up[...] = 5.0

    # set bounds on nonlinear constraints
    # start point constraints
    r0_lo = sn.Flow[1:1+NY]
    r0_up = sn.Fupp[1:1+NY]

    # matching conditions
    r_mc_lo = sn.Flow[1+NY:-NY].reshape(NMS, NY)
    r_mc_up = sn.Fupp[1+NY:-NY].reshape(NMS, NY)

    # end point constraints
    rf_lo = sn.Flow[-NY:]
    rf_up = sn.Fupp[-NY:]
    assert sn.Fupp.size == 1 + r0_up.size + r_mc_up.size + rf_up.size
    assert sn.Flow.size == 1 + r0_lo.size + r_mc_lo.size + rf_lo.size

    r0_lo[:] = [0.0, 0.0]
    r0_up[:] = [0.0, 0.0]

    r_mc_lo[:, :] = 0.0
    r_mc_up[:, :] = 0.0

    rf_lo[:] = [0.0, 0.0]
    rf_up[:] = [0.0, 0.0]

    # load model function interface
    generate_so = False
    # generate_so = True
    so_path, ds = load_fortran_example_model_by_name("rocketcar", generate_so)
    # define model functions
    mf = Model(model_functions=so_path, model_definitions=ds, verbose=True)

    # instantiate integrator
    ind = RcExplicitEuler(x0=y[0, :].copy())

    # setup plotting routines
    show_canvas = False
    plot = Plotter(NY=NY, NU=NU, ind=ind, mf=mf, show_canvas=show_canvas)

    # -------------------------------------------------------------------------
    # EVALUATION OF SHOOTING NLP
    # define shooting problem in an SNOPT compatible way
    def evaluate(status, x, needF, nF, F, needG, neG, G, cu, iu, ru):
        """Function to implement that is used by SNOPT to solve the problem."""
        F_evaluated = False

        # set status flag
        status[0] = 0

        # erase current plotting data
        ind.ts_plt = []
        ind.ys_plt = []

        # evaluate gradient of problem
        if needG[0] != 0:
            ret = eval_G(F, G, x, x_d, ind, mf)
            if ret == 0:
                F_evaluated = True

        # evaluate F only when explicitly needed
        if needF[0] != 0:
            eval_F(F, x, ind, mf)

        return None
    # -------------------------------------------------------------------------

    # set derivative mode of SNOPT and evaluate initial Jacobian
    sn.set_derivative_mode(1)  # mode 0 means use finite differences
    if sn.deropt[0] == 0:
        print "estimate jacobian structure"
        print "-"*30
        sn.calc_jacobian(evaluate)
        print ""

    print "solve SQP problem"
    print "-"*30
    sn.sqp_step(evaluate)
    print ""

    print "evaluate F for plotting"
    print "-"*30
    print ""
    F = sn.F
    x = sn.x

    y = x[:NTS*NY].reshape(NTS, NY)  # shooting variables
    q = x[NTS*NY:-NP].reshape(NMS, NU)  # pwc. controls
    p = x[-NP:]  # parameters

    print "Results"
    print "-"*30
    print "y: \n", y
    print "q: \n", q
    print "p: \n", p

    eval_F(F, x, ind, mf)
    plot.update(F, x)
    plot.savefig(fname="oc_ms_rc_rocketcar")
    print ""


# ------------------------------------------------------------------------------
