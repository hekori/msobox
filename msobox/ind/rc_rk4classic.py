# -*- coding: utf-8 -*-
"""
Classic Runge-Kutta Scheme of Order 4 with reverse communication interface.
"""
import numpy


# ------------------------------------------------------------------------------
class RcRK4Classic(object):

    """Classic Runge-Kutta Scheme of Order 4."""

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
        self._f[...] = value

    f = property(
        _get_f, _set_f, None,
        "Current right hand side f evaluation."
    )

    # --------------------------------------------------------------------------
    def _get_x(self):
        return self._x

    def _set_x(self, value):
        self._x[...] = value

    x = property(
        _get_x, _set_x, None,
        "Current state of the ordinary differential equation."
    )

    # --------------------------------------------------------------------------
    def __init__(self, NX=0):
        """Classic Runge-Kutta scheme with reverse communication interface."""
        self._STATE = 'provide_x0'

        self.j = 0
        self.t = numpy.zeros(1)

        self.NTS = 0
        self.ts = numpy.zeros([self.NTS], dtype=float)

        self.NX = NX
        self._f = numpy.zeros([self.NX], dtype=float)
        self._x = numpy.zeros([self.NX], dtype=float)
        self._y = numpy.zeros([self.NX], dtype=float)

        # Runge-Kutta intermediate steps
        self._K1 = numpy.zeros([self.NX])
        self._K2 = numpy.zeros([self.NX])
        self._K3 = numpy.zeros([self.NX])
        self._K4 = numpy.zeros([self.NX])

        self.h = 0.0
        self.h2 = 0.0

    def init_zo_forward(self, ts, NX=None):
        """
        Initialize memory for zero order forward integration.

        Parameters
        ----------
        ts : array-like
            time grid for integration
        NX : int
            number of differential states
        """
        if NX and NX != self.NX:
            self.NX = NX
            self._f.resize([NX])
            self._x.resize([NX])
            self._y .resize([NX])

            self._K1.resize([NX])
            self._K2.resize([NX])
            self._K3.resize([NX])
            self._K4.resize([NX])

        self.NTS = ts.size
        self.ts = ts
        self.j = 0
        self._STATE = 'provide_x0'
        self._K_CNT = 0

    def step_zo_forward(self):
        """
        Solve nominal differential equation using an Runge-Kutta scheme.
        """
        try:
            self.h = self.ts[self.j+1] - self.ts[self.j]
            self.h2 = self.h/2.0
        except IndexError:
            self.h = 0.0
            self.h2 = self.h/2.0

        if self.j == self.NTS - 1:
            self._STATE = 'finished'

        elif self.STATE == 'provide_x0':
            # NOTE: temporarily save initial value
            self._y[:] = self._x
            self._STATE = 'plot'

        elif self.STATE == 'plot':
            self._STATE = 'prepare_K1'

        elif self.STATE == 'prepare_K1':
            # K1 = h*f(t, y, p, u)
            self.t[0] = self.ts[self.j]
            self._x[:] = self._y
            self._STATE = 'provide_f'
            self._K_CNT = 1

        elif self.STATE == 'provide_f' and self._K_CNT == 1:
            self._STATE = 'compute_K1'

        elif self.STATE == 'compute_K1':
            self._K1[:] = self.h*self._f
            self._STATE = 'prepare_K2'

        elif self.STATE == 'prepare_K2':
            # K2 = h*f(t + h2, y + 0.5*K1, ...)
            self.t[0] = self.ts[self.j] + self.h2
            self._x[:] = self._y + 0.5*self._K1
            self._STATE = 'provide_f'
            self._K_CNT = 2

        elif self.STATE == 'provide_f' and self._K_CNT == 2:
            self._STATE = 'compute_K2'

        elif self.STATE == 'compute_K2':
            self._K2[:] = self.h*self._f
            self._STATE = 'prepare_K3'

        elif self.STATE == 'prepare_K3':
            # K3 = h*f(t + h2, y + 0.5*K2, ...)
            self.t[0] = self.ts[self.j] + self.h2
            self._x[:] = self._y + 0.5*self._K2
            self._STATE = 'provide_f'
            self._K_CNT = 3

        elif self.STATE == 'provide_f' and self._K_CNT == 3:
            self._STATE = 'compute_K3'

        elif self.STATE == 'provide_f3':
            self._STATE = 'compute_K3'

        elif self.STATE == 'compute_K3':
            self._K3[:] = self.h*self._f
            self._STATE = 'prepare_K4'

        elif self.STATE == 'prepare_K4':
            # K4 = h*f(t + h, y + K3, ...)
            self.t[0] = self.ts[self.j] + self.h
            self._x[:] = self._y + self._K3
            self._STATE = 'provide_f'
            self._K_CNT = 4

        elif self.STATE == 'provide_f' and self._K_CNT == 4:
            self._STATE = 'compute_K4'
            self._K_CNT = 0

        elif self.STATE == 'compute_K4':
            self._K4[:] = self.h*self._f
            self._STATE = 'advance'

        elif self.STATE == 'advance':
            self._x[:] = self._y[:] \
                + (self._K1 + 2*self._K2 + 2*self._K3 + self._K4) / 6.0
            self._y[:] = self._x[:]
            self.j += 1
            self._STATE = 'plot'

        else:
            err_str = "ERROR: STATE {} not defined!".format(self.STATE)
            raise AttributeError(err_str)

    def init_fo_forward(self, ts, NX=None, P=1):
        """
        Initialize memory for first order forward integration.

        Parameters
        ----------
        ts : array-like
            time grid for integration
        NX : int
            number of differential states
        P : int
            number of first-order forward directions
        """
        pass

    def step_fo_forward(self):
        """
        Solve nominal differential equation and evaluate first-order forward
        sensitivities.
        """
        pass


# ------------------------------------------------------------------------------
if __name__ == '__main__':
    import sys

    def ffcn(f, t, x, p, u):
        f[0] = -p[0]*x[0]

    def ref(xs, ts, p, u, x0):
        xs[...] = (numpy.exp(-p[0]*ts)*x0).reshape(xs.shape)

    NTS = 101
    t0 = 0.0
    tf = 10.
    ts = numpy.linspace(t0, tf, NTS, endpoint=True)

    NX = 1
    NP = 1
    NU = 1
    x = numpy.array([2.])
    p = numpy.array([1])
    u = numpy.array([0])

    xs = numpy.zeros([NTS, NX])
    rs = xs.copy()
    ref(rs, ts, p, u, x)

    ind = RcRK4Classic(NX=1)
    ind.init_zo_forward(ts=ts)

    # reverse communication loop
    while True:
        if ind.STATE == 'provide_x0':
            ind.x = x

        if ind.STATE == 'provide_f':
            u[0] = 1.
            ffcn(ind.f, ind.t, ind.x, p, u)

        if ind.STATE == 'plot':
            xs[ind.j, :] = ind.x

        if ind.STATE == 'finished':
            print 'done'
            break

        ind.step_zo_forward()

    import matplotlib.pyplot as plt
    plt.plot(ts, xs, '-k', label="rk4")
    plt.plot(ts, rs, ':r', label="ref")
    plt.legend(loc="best")
    plt.show()
