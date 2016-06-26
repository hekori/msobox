# -*- coding: utf-8 -*-
"""."""
import os
import sys
import numpy as np
import pytest

from msobox.ind.rkf45swt import RKF45SWT


# ------------------------------------------------------------------------------
# LOCAL FIXTURES
G = 9.81

def ref_ffcn(f, t, x, p):
    f[0] = x[1]
    f[1] = -G


def ref_sfcn(s, t, x, p):
    s[0] = x[0]


def ref_dfcn(d, t, x, p):
    d[0] = x[0]
    d[1] = -p[0]*x[1]


def ref_x(x, t, p, t0, x0):
    x[0] = -0.5*G*t**2 + t*x0[1] + x0[0]
    x[1] = -G*t + x0[1]


def ref_tswt(tswt, p, t0, x0):
    tswt0 = (x0[1] + G*t0 - np.sqrt(2.0*x0[0]*G + x0[1]**2))/G
    tswt1 = (x0[1] + G*t0 + np.sqrt(2.0*x0[0]*G + x0[1]**2))/G
    tswt[0] = max(tswt0, tswt1)


def ref_xs(xs, ts, p, t0, x0):
    xs[:, 0] = -0.5*G*(ts - t0)**2 + (ts - t0)*x0[1] + x0[0]
    xs[:, 1] = -G*(ts - t0) + x0[1]


def ref_solution(xs, ts, p, t0, x0):
    t0 = np.asarray(t0.copy())
    tf = np.asarray(ts[-1].copy())
    if not t0.shape:
        t0 = np.array([t0]).flatten()
    if not tf.shape:
        tf = np.array([tf]).flatten()

    x0 = np.asarray(x0)

    mask = np.zeros(ts.shape, dtype=bool)

    tswt = np.array([0.0])
    xswt = np.zeros([1, x0.size])

    tswts = []
    xswts = []
    cnt = 0
    while tswt[0] < tf[0]:
        # evaluate next switching point
        ref_tswt(tswt, p, t0, x0)
        ref_xs(xswt, tswt, p, t0, x0)

        if tswt[0] <= tf[0]:
            # retrieve range of values
            tmin = np.min(ts[np.where(t0[0] <= ts)])
            tmin = np.where(ts == tmin)[0][0]

            tmax = np.max(ts[np.where(ts <= tswt[0])])
            tmax = np.where(ts == tmax)[0][0]

            # define slice to reuse array
            s = slice(tmin, tmax+1)

            # integrate in between switches
            ref_xs(xs[s, :], ts[s], p, t0, x0)

            # copy states
            t0[...] = tswt
            x0[...] = xswt[0, :]

            # append current states
            # NOTE: append time points and states twice due to discontinuity
            tswts.append(t0.copy())
            xswts.append(x0.copy())

            # apply update function
            ref_dfcn(x0, tswt, xswt[0, :], p)

            # append current states
            # NOTE: append time points and states twice due to discontinuity
            tswts.append(t0.copy())
            xswts.append(x0.copy())

            cnt += 1
        else:
            # evaluate last time point
            tswt[0] = tf
            xswt[...] = 0.0

            # retrieve range of values
            tmin = np.min(ts[np.where(t0[0] <= ts)])
            tmin = np.where(ts == tmin)[0][0]

            tmax = np.max(ts[np.where(ts <= tswt[0])])
            tmax = np.where(ts == tmax)[0][0]

            # define slice to reuse array
            s = slice(tmin, tmax+1)

            # integrate in between switches
            ref_xs(xs[s, :], ts[s], p, t0, x0)
            break

    tswts = np.asarray(tswts).flatten()
    xswts = np.asarray(xswts)

    return tswts, xswts


# ------------------------------------------------------------------------------
# ACTUAL TESTS

# ------------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    Nts = 101
    NX = 2
    NP = 1

    # states and parameters
    x = np.zeros(NX)
    p = np.zeros(NP)
    p[0] = 1.0  # fully elastic collision

    # initial values
    t0 = np.array([0.0])
    tf = np.array([10.0])
    x0 = x.copy()
    x0[0] = 1.0  # initial height
    x0[1] = 0.0  # initial velocity

    # setup figure
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # plot solution
    ts = np.linspace(t0, tf, Nts, endpoint=True)
    xs = np.zeros([Nts, NX])

    # evaluate reference solution + switching points
    tswts, xswts = ref_solution(xs, ts, p, t0, x0)

    ax.plot(ts, xs, label="ref")
    ax.plot(tswts, xswts, ls="", marker="x", label="swt")

    tnew = np.zeros([ts.shape[0] + tswts.shape[0]])
    xnew = np.zeros([tnew.shape[0], xs.shape[1]])

    # sort in values
    _a = 0  # col counter
    _iswt = 0
    _, unq_idx = np.unique(tswts, return_index=True)
    for i, iswt in enumerate(unq_idx):
        print i
        # insert integration values
        if _iswt == 0:
            idx = np.where(ts <= tswts[iswt])
        else:
            idx = np.where((tswts[_iswt] <= ts) & (ts <= tswts[iswt]))

        _ts = ts[idx]
        _xs = xs[idx]

        _b = _a + _ts.shape[0]
        print "s = ", _a, _b, "(", _b - _a,")"
        print "_ts.shape: ", _ts.shape
        print "_xs.shape: ", _xs.shape
        print "tnew[_a:_b].shape:", tnew[_a:_b].shape
        print "xnew[_a:_b].shape:", xnew[_a:_b].shape
        tnew[_a:_b] = _ts
        xnew[_a:_b] = _xs

        _a = _b
        _b = _b + 2
        print "s = ", _a, _b, "(", _b - _a,")"
        tnew[_a:_b] = tswts[iswt:iswt+2]
        xnew[_a:_b] = xswts[iswt:iswt+2]
        print "tswts[iswt:iswt+1]: ", tswts[iswt:iswt+2]
        print "xswts[iswt:iswt+1]: ", xswts[iswt:iswt+2]
        print "tnew[_a:_b].shape:", tnew[_a:_b].shape
        print "xnew[_a:_b].shape:", xnew[_a:_b].shape

        _a = _b
        _iswt = iswt
        print ""

    print tswts
    print tnew
    # print xnew

    # beautify axes
    ax.legend(loc="best")
    ax.relim()
    ax.autoscale_view(True, True, False)

    # save figure
    _options = {
        "fmt": "pdf",
        "dpi": 300,
    }
    path = os.path.join("temp", "ball_swt_simulation.{fmt}".format(**_options))
    fig.savefig(path, **_options)
