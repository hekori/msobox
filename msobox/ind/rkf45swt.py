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


class RKF45SWT(object):

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
    _ws = numpy.array([
              1559.0/  12500.0,
                           0.0,
            153856.0/ 296875.0,
             68107.0/2612500.0,
              -243.0/  31250.0,
             -2106.0/  34375.0
    ])

    def _get_interpolation_polynomial(self, x0, x1, ks, h, res):
        """
        Get interpolation polynomial on intervall [x0, x1].

        Polynomial interpolations using Atkin-Neville algorithm. Callable
        evaluates 6th order Hermite-Birkhoff polynomial recycling evaluation of
        the embedded, continuous Runge-Kutta-Fehlberg method.

        Parameters
        ----------
        x0 : array-like of shape (NX,)
            Solution y(t)
        x1 : array-like of shape (NX,)
            Solution y(t+h)
        ks : array-like of shape (7, NX)
            Matrix of approximate derivatives, we need y'(t), y'(t+h)
        h : scalar
            Step size h

        Returns
        -------
        res : Functor
            Cubic Hermite polynomial evaluation using Atkin-Neville's algorithm
        """
        # check dimensions of derivatives
        assert ks.shape == (7, self.NX)

        # allocate memory
        Xs = numpy.zeros(8, )
        Ys = numpy.zeros(8, self.NX)

        # assign function evaluations from integration
        Xs[:-1,]   = 0.0
        Ys[:-1, :] = ks[:, :]

        # compute Horn's locally sixth-order approximation in tau=0.6
        Ys[:, -1] = x0 + h*(
            w[0]*ks[:, 0] + w[2]*ks[:, 2] + w[3]*ks[:, 3] +
            w[4]*ks[:, 4] + w[5]*ks[:, 5]
        )

        # return interpolation polynomial
        return NevilleInterpolationPolynomial(Xs, Ys)

    def __init__(self, backend):
        """Initialize integrator."""
        self.printlevel = 0

        self.NY  = 0           # number of differential variables y
        self.NZ  = 0           # number of algebraic variables z
        self.NX  = 0           # number of variables x = (y,z)
        self.NP  = 0           # number of parameters
        self.NU  = 0           # number of control functions
        self.NQI = 0           # number of q in one control interval

        self.backend = backend

    def update_u(self, i):
        self.u[:] = self.q[:, i, 0]

    def update_u_dot(self, i):
        self.u[:] = self.q[:, i, 0]
        self.u_dot[:, :] = self.q_dot[:, i, 0, :]

    def update_u_bar(self, i):
        self.q_bar[:, i, 0] += self.u_bar[:]
        self.u_bar[:] = 0.

    def update_u_ddot(self, i):
        self.u[:] = self.q[:, i, 0]
        self.u_dot1[:, :] = self.q_dot1[:, i, 0, :]
        self.u_dot2[:, :] = self.q_dot2[:, i, 0, :]
        self.u_ddot[:, :] = self.q_ddot[:, i, 0, :, :]

    def zo_check(self, ts, x0, p, q):
        """Check dimensions and allocate memory."""
        # set dimensions
        self.M   = ts.size              # number of time steps
        self.NQ  = self.NU*self.M*2     # number of control variables

        # assert that the dimensions match
        self.NX = x0.size
        self.NP = p.size

        self.NU = q.shape[0]
        self.M  = q.shape[1]

        # assign variables
        self.ts = ts
        self.x0 = x0
        self.p  = p
        self.q  = q

        # allocate memory
        self.xs   = numpy.zeros((self.M, self.NX))
        self._x1s = numpy.zeros((self.M, self.NX)) # 2nd procedure
        self._Ks  = numpy.zeros((self._stages[-1], self.NX))
        self.f    = numpy.zeros((self.NX,))
        self.u    = numpy.zeros((self.NU,))

    def zo_forward(self, ts, x0, p, q):
        # check if dimensions are correct
        self.zo_check(ts, x0, p, q)

        # rename for convenience
        As = self._As
        Bs = self._Bs
        Cs = self._Cs
        Ks = self._Ks
        xs = self.xs
        x1s = self._x1s
        p = self.p
        u = self.u

        # set initial value
        xs[0, :]  = x0
        x1s[0, :] = x0 # 2nd procedure

        # initialize intermediate values
        t = numpy.zeros(1)
        y = numpy.zeros(self.f.shape)

        for interval in range(self.M-1):
            self.update_u(interval)

            # get time step
            h = self.ts[interval+1] - self.ts[interval]

            # on first interval evaluate ffcn
            #if interval == 0:  # is first time interval?
            # forward K1 = h*f[t, y, p, u]
            t[0] = ts[interval]
            y[:] = xs[interval, :]  # As[0] = 0.0
            self.backend.ffcn(Ks[0], t, y, p, u)
            Ks[0] *= h

            t[0] = ts[interval] + As[1]*h
            y[:] = xs[interval, :] + Bs[0, :1].dot(Ks[:1])
            self.backend.ffcn(Ks[1], t, y, p, u)
            Ks[1] *= h

            t[0] = ts[interval] + As[2]*h
            y[:] = xs[interval, :] + Bs[1, 0]*Ks[0] + Bs[1, 1]*Ks[1]
            self.backend.ffcn(Ks[2], t, y, p, u)
            Ks[2] *= h

            t[0] = ts[interval] + As[3]*h
            y[:] = xs[interval, :] + Bs[2, 0]*Ks[0] + Bs[2, 1]*Ks[1] + Bs[2, 2]*Ks[2]
            self.backend.ffcn(Ks[3], t, y, p, u)
            Ks[3] *= h

            t[0] = ts[interval] + As[4]*h
            y[:] = xs[interval, :] + Bs[3, 0]*Ks[0] + Bs[3, 1]*Ks[1] + Bs[3, 2]*Ks[2] + Bs[3, 3]*Ks[3]
            self.backend.ffcn(Ks[4], t, y, p, u)
            Ks[4] *= h

            t[0] = ts[interval] + As[5]*h
            y[:] = xs[interval, :] + Bs[4, 0]*Ks[0] + Bs[4, 1]*Ks[1] + Bs[4, 2]*Ks[2] + Bs[4, 3]*Ks[3] + Bs[4, 4]*Ks[4]
            self.backend.ffcn(Ks[5], t, y, p, u)
            Ks[5] *= h

            xs [interval + 1, :] =  xs[interval, :] + Cs[0, 0]*Ks[0] + Cs[0, 1]*Ks[1] + Cs[0, 2]*Ks[2] + Cs[0, 3]*Ks[3] + Cs[0, 4]*Ks[4] + Cs[0, 5]*Ks[5]
            x1s[interval + 1, :] = x1s[interval, :] + Cs[1, 0]*Ks[0] + Cs[1, 1]*Ks[1] + Cs[1, 2]*Ks[2] + Cs[1, 3]*Ks[3] + Cs[1, 4]*Ks[4] + Cs[1, 5]*Ks[5]
