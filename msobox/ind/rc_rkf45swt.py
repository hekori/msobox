# -*- coding: utf-8 -*-
"""
A continuous Runge-Kutta-Fehlberg integrator with implict switch detection.

Features
--------
* automatic detection of implicit switches in the states & right-hand side,
* treats variational differential equations (not varied trajectories)

based on C implementation
(c) Christian Kirches, 2006

based on rkfXXadj.cpp
(c) Leonard Wirsching, 2004

partially based on FORTRAN implementation rkf45s.F
(c) Andreas Schaefer, 1996

References
----------
*  Fehlberg, E.: Klassische Runge-Kutta-Formeln vierter
   und niedrigerer Ordnung. Computing 6, 1970, pp. 61-70
*  Stoer, J., and R. Bulirsch: Introduction to Numerical
   Analysis. Springer, New York, 1993.
   (formula for step size estimate)
*  Bock, H.-G.: Numerical treatment of inverse problems in
   chemical reaction kinetics. In K.H. Ebert, P. Deuflhard, and
   W. Jaeger (eds.): Modelling of Chemical Reaction Systems
   (Springer Series in Chemical Physics 18). Springer, Heidelberg, 1981.
   (IND principle)
*  Shampine, L. F.: Numerical Solution of Ordinary Differential
   Equations. Chapman & Hall, New York, 1994.
   (initial step size selection)
*  Mombaur, K. D.: Stability Optimization of Open Loop Controlled
   Walking Robots, PhD thesis, University of Heidelberg, 2001.
   (1st and 2nd order sensitivity updates for sensitivity matrices)
*  Enright, W., Jackson, K.R., Norsett, S.P., Thomsen, P.G.: Interpolants
   for Runge-Kutta Formulas. ACM Transactions on Mathematical Software
   (TOMS), Vol. 12, No. 3, September 1986, pp. 193-218 (interpolants)
"""

import numpy

from msobox.ind.polynomials import NevilleInterpolationPolynomial

class RcRKF45SWT(object):

    """
    A continuous Runge-Kutta-Fehlberg integrator.

    """

    # orders
    _orders = numpy.array([4, 5])
    # stage counts
    _stages = numpy.array([5, 6])

    # butcher-tableau of the Runge-Kutta method
    #  0 |
    # a0 | b00
    # a1 | b10 b11
    # a2 | b20 b21 b22
    # .. | ... ... ...
    # as | bs0 bs1 bs2 ... bss
    # ------------------------
    #    | c00 c01 c02 ... c0s
    #    | c10 c11 c12 ... c1s

    # coefficients alpha (times)
    _As = numpy.array([
        0.0, 1.0/4.0, 3.0/8.0, 12.0/13.0, 1.0, 1.0/2.0
    ])
    # triangular coefficient matrix beta
    _Bs = numpy.array([
        [       1.0/4.0,            0.0,            0.0,           0.0,        0.0],
        [      3.0/32.0,       9.0/32.0,            0.0,           0.0,        0.0],
        [ 1932.0/2197.0, -7200.0/2197.0,  7296.0/2197.0,           0.0,        0.0],
        [   439.0/216.0,           -8.0,   3680.0/513.0, -845.0/4104.0,        0.0],
        [     -8.0/27.0,            2.0, -3544.0/2565.0, 1859.0/4104.0, -11.0/40.0],
    ])
    _Cs = numpy.array([
        # coefficients c of lower-order-method
        [25.0/216.0, 0.0, 1408.0/2565.0,  2197.0/4104.0,  -1.0/5.0,       0.0],
        # coefficients c of higher-order method
        [-1.0/360.0, 0.0,  128.0/4275.0, 2197.0/75240.0, -1.0/50.0, -2.0/55.0],
    ])

    # coefficients for Horn's locally sixth-order approximation in tau=0.6
    _Ws = numpy.array([
              1559.0/  12500.0,
                           0.0,
            153856.0/ 296875.0,
             68107.0/2612500.0,
              -243.0/  31250.0,
             -2106.0/  34375.0
    ])

    # --------------------------------------------------------------------------
    def _get_STATE(self):
        return self._STATE

    STATE = property(
        _get_STATE, None, None,
        "Current state of the reverse communication interface."
    )

    # --------------------------------------------------------------------------
    def _get_f(self):
        return self._f

    def _set_f(self, value):
        self._f[:] = value

    f = property(
        _get_f, _set_f, None,
        "Current right hand side f evaluation."
    )

    # --------------------------------------------------------------------------
    x = property(
        None, None, None,
        "Current state of the ordinary differential equation."
    )

    # --------------------------------------------------------------------------
    def _get_y(self):
        return self._y

    def _set_y(self, value):
        self._y[...] = value

    y = property(
        _get_y, _set_y, None,
        "Current state of the ordinary differential equation."
    )

    # --------------------------------------------------------------------------
    def _get_z(self):
        return self._z

    def _set_z(self, value):
        self._z[...] = value

    z = property(
        _get_z, _set_z, None,
        "Current algebraic state from interpolation."
    )

    # --------------------------------------------------------------------------
    def _get_tswts(self):
        return numpy.asarray(self._tswts)

    tswts = property(
        _get_tswts, None, None,
        "Current switching times."
    )

    # --------------------------------------------------------------------------
    def _get_yswts(self):
        return numpy.asarray(self._yswts)

    yswts = property(
        _get_yswts, None, None,
        "Current switching differential states."
    )

    # --------------------------------------------------------------------------
    def _get_zswts(self):
        return numpy.asarray(self._zswts)

    zswts = property(
        _get_zswts, None, None,
        "Current switching algebraic states."
    )

    # --------------------------------------------------------------------------
    def __init__(self, y0, z0=None, NS=0):
        """Initialize integrator."""

        self.j = 0
        self.t = numpy.zeros(1)

        self.NTS = 0
        self.ts = numpy.zeros([self.NTS], dtype=float)

        # assign differential states
        self.NX = y0.size
        self._f = y0.copy()
        self._y = y0.copy()
        self._y_tmp = y0.copy()
        #   2nd procedure
        self._y1 = y0.copy()

        #   algebraic states for interpolation
        self._z = z0
        if z0:
            self._z = z0.copy()

        # prepare placeholder for switching structure
        self._tswts = []
        self._yswts = []
        self._zswts = []

        # Runge-Kutta intermediate step evaluation
        # NOTE: _Ks[-1, :] is used to store Horn's locally sixth-order
        #       approximation in tau=0.6 for error control interpolation
        self._ts = numpy.zeros((self._stages[-1]+1,))
        self._Ks = numpy.zeros((self._stages[-1]+1, self.NX))
        self.h = None

        # go into first state
        self._STATE = 'provide_y0'
        self._K_CNT = 0

    def init_zo_forward(self, ts):
        """Check dimensions and allocate memory."""
        self.NTS = ts.size
        self.ts = ts

        self.j = 0

        self._f.fill(0.0)
        self._y.fill(0.0)
        self._y_tmp.fill(0.0)
        self._y1.fill(0.0)
        if self._z:
            self._z.fill(0.0)

        # interpolation polynomial
        self.p = None

        # prepare placeholder for switching structure
        self._tswts = []
        self._yswts = []
        self._zswts = []

        # go into first state
        self._STATE = 'provide_y0'
        self._K_CNT = 0

    # --------------------------------------------------------------------------
    def step_zo_forward(self):
        """
        Solve nominal differential equation using an Runge-Kutta scheme.
        """
        # rename for convenience
        ts = self.ts
        As = self._As
        Bs = self._Bs
        Cs = self._Cs
        Ks = self._Ks
        Ws = self._Ws

        # try get time step
        try:
            self.h = self.ts[self.j+1] - self.ts[self.j]
        except IndexError:
            # NOTE: fails on last time node and is then not needed
            self.h = None

        if self.j == self.NTS - 1:
            self._STATE = 'finished'

        elif self.STATE == 'provide_y0':
            # NOTE: temporarily save initial value
            self._y_tmp[:] = self._y
            self._STATE = 'plot'

        elif self.STATE == 'plot':
            self._STATE = 'prepare_K'

        elif self.STATE == 'prepare_K':
            # TODO evaluate f only when x0 was provided, else recycle last evaluation
            if self._K_CNT == 0:
                self.t[0] = ts[self.j]
                self._ts[0] = self.t[0]  # NOTE: save time node for evaluation
                self._y[:] = self._y_tmp  # As[0] = 0.0

            elif self._K_CNT == 1:
                self.t[0] = self.ts[self.j] + As[1]*self.h
                self._ts[1] = self.t[0]  # NOTE: save time node for evaluation
                self._y[:] = self._y_tmp + Bs[0, :1]*(Ks[0])

            elif self._K_CNT == 2:
                self.t[0] = self.ts[self.j] + As[2]*self.h
                self._ts[2] = self.t[0]  # NOTE: save time node for evaluation
                self._y[:] = self._y_tmp + Bs[1, 0]*Ks[0] + Bs[1, 1]*Ks[1]

            elif self._K_CNT == 3:
                self.t[0] = ts[self.j] + As[3]*self.h
                self._ts[3] = self.t[0]  # NOTE: save time node for evaluation
                self._y[:] = self._y_tmp \
                    + Bs[2, 0]*Ks[0] + Bs[2, 1]*Ks[1] + Bs[2, 2]*Ks[2]

            elif self._K_CNT == 4:
                self.t[0] = self.ts[self.j] + As[4]*self.h
                self._ts[4] = self.t[0]  # NOTE: save time node for evaluation
                self._y[:] = self._y_tmp \
                    + Bs[3, 0]*Ks[0] + Bs[3, 1]*Ks[1] + Bs[3, 2]*Ks[2] \
                    + Bs[3, 3]*Ks[3]

            elif self._K_CNT == 5:
                self.t[0] = self.ts[self.j] + As[5]*self.h
                self._ts[5] = self.t[0]  # NOTE: save time node for evaluation
                self._y[:] = self._y_tmp \
                    + Bs[4, 0]*Ks[0] + Bs[4, 1]*Ks[1] + Bs[4, 2]*Ks[2] \
                    + Bs[4, 3]*Ks[3] + Bs[4, 4]*Ks[4]

            else:
                err_str = "Ups, something went wrong!"
                raise IndexError(err_str)

            # compute right hand side evaluation
            self._STATE = 'provide_f'

        elif self.STATE == 'provide_f':
            self._STATE = 'compute_K'

        elif self.STATE == 'compute_K':
            Ks[self._K_CNT] = self.h*self._f
            self._K_CNT += 1
            self._K_CNT = self._K_CNT % self._stages[-1]
            if self._K_CNT == 0:
                self._STATE = 'advance'
            else:
                self._STATE = 'prepare_K'

        elif self.STATE == 'advance':
            self._y[:] = self._y_tmp \
                + Cs[0, 0]*Ks[0] + Cs[0, 1]*Ks[1] + Cs[0, 2]*Ks[2] \
                + Cs[0, 3]*Ks[3] + Cs[0, 4]*Ks[4]
            self._y1[:] = self._y_tmp \
                + Cs[1, 0]*Ks[0] + Cs[1, 1]*Ks[1] + Cs[1, 2]*Ks[2] \
                + Cs[1, 3]*Ks[3] + Cs[1, 4]*Ks[4] + Cs[1, 5]*Ks[5]

            # TODO check if this is correct
            # compute Horn's locally sixth-order approximation in tau=0.6
            self._ts[6] = self.ts[self.j] + 0.6*self.h
            self._Ks[6] = Ws[0]*Ks[0] + Ws[2]*Ks[2] + Ws[3]*Ks[3] \
                + Ws[4]*Ks[4] + Ws[5]*Ks[5]

            self._Ks[...] = self._y_tmp + self._Ks

            # TODO check which procedure is best used to predict next step
            self._y_tmp[:] = self._y[:]
            # self._y_tmp[:] = self._y1[:]

            self._STATE = 'interpolate'

        elif self._STATE == 'interpolate':
            self.p = self._get_interpolation_polynomial(self._ts, Ks)
            self._STATE = 'provide_sigma'

        elif self._STATE == 'provide_sigma':
            self._STATE = 'provide_delta'

        elif self._STATE == 'provide_delta':

            self.j += 1
            self._STATE = 'plot'

        else:
            err_str = "ERROR: STATE {} not defined!".format(self.STATE)
            raise AttributeError(err_str)


    # --------------------------------------------------------------------------
    def _get_interpolation_polynomial(self, ts, ys):
        """
        Get interpolation polynomial on intervall [x0, x1].

        Polynomial interpolations using Atkin-Neville algorithm. Callable
        evaluates 6th order Hermite-Birkhoff polynomial recycling evaluation of
        the embedded, continuous Runge-Kutta-Fehlberg method.

        Parameters
        ----------
        ts : array-like (NTS,)
            time nodes of function evaluations

        ys : array-like (NTS, NY)
            function evaluation at given time nodes

        Returns
        -------
        res : Functor
            Cubic Hermite polynomial evaluation using Atkin-Neville's algorithm
        """
        # return interpolation polynomial
        return NevilleInterpolationPolynomial(ts, ys)


# ------------------------------------------------------------------------------
if __name__ == '__main__':
    import sys

    def ffcn(f, t, x, p, u):
        f[0] = x[1]
        f[1] = -10.0

        # f[0] = - p[0]*x[0]
        # f[1] = - p[1]*x[1]

    def ref(xs, ts, p, u, x0):
        xs[:, 0] = (numpy.exp(-p[0]*ts)*x0[0])
        xs[:, 1] = (numpy.exp(-p[1]*ts)*x0[1])

    NTS = 11
    t0 = 0.0
    tf = 10.
    ts = numpy.linspace(t0, tf, NTS, endpoint=True)

    NX = 2
    NP = 2
    NU = 1
    x = numpy.array([2.0, 0.0])
    p = numpy.array([1.0, 0.7])
    u = numpy.array([0.0])

    xs = numpy.zeros([NTS, NX])
    rs = xs.copy()
    ref(rs, ts, p, u, x)

    ind = RcRKF45SWT(x)
    ind.init_zo_forward(ts=ts)

    # reverse communication loop
    while True:
        # TODO remove debug out
        print(("STATE: ", ind._STATE))
        print(("K_CNT: ", ind._K_CNT))
        print(("j: ", ind.j, " / ", ind.NTS))

        if ind.STATE == 'provide_y0':
            ind.y = x

        if ind.STATE == 'provide_f':
            u[0] = 1.
            print("evaluate f")
            ffcn(ind.f, ind.t, ind.y, p, u)

        if ind.STATE == 'plot':
            xs[ind.j, :] = ind.y

        print(("ts[", ind.j, "] = \n", ts[ind.j]))
        print(("rs[", ind.j, "] = \n", rs[ind.j]))
        print("-")
        print(("ind.y = \n", ind.y))
        print(("ind._y_tmp = \n", ind._y_tmp))
        print(("ind._f = \n", ind._f))
        print(("ind._Ks = \n", ind._Ks.T))
        print("")

        if ind.STATE == 'finished':
            print('done')
            break

        ind.step_zo_forward()

    import matplotlib.pyplot as plt
    plt.plot(ts, xs, '-k', label="rk45swt")
    plt.plot(ts, rs, ':r', label="ref")
    plt.legend(loc="best")
    plt.show()
