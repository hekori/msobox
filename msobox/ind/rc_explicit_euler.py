# -*- coding: utf-8 -*-
"""
Implementation of a reverse communication explicit Euler integration scheme.
"""
import numpy


# ------------------------------------------------------------------------------
class RcExplicitEuler(object):

    """Explicit Euler integration scheme with reverse communication interface."""

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
    def _get_f_d(self):
        return self._f_d

    def _set_f_d(self, value):
        if self._f_d is None:
            self._f_d = value

        self._f_d[...] = value

    f_d = property(
        _get_f_d, _set_f_d, None,
        "Current right hand side f first-order forward derivative."
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
    def _get_x_d(self):
        return self._x_d

    def _set_x_d(self, value):
        if self._x_d is None:
            self._x_d = value

        self._x_d[...] = value

    x_d = property(
        _get_x_d, _set_x_d, None,
        "Current state of the ordinary differential equation."
    )

    # --------------------------------------------------------------------------
    def __init__(self, x0):
        """Classic Runge-Kutta scheme with reverse communication interface."""
        self.j = 0
        self.t = numpy.zeros(1)

        self.NTS = 0
        self.ts = numpy.zeros([self.NTS], dtype=float)

        self.NX = x0.size
        self._f = x0.copy()
        self._x = x0.copy()

        self._f_d = None
        self._x_d = None

        self._STATE = 'provide_x0'

    def init_zo_forward(self, ts):
        """
        Initialize memory for zero order forward integration.

        Parameters
        ----------
        ts : array-like
            time grid for integration
        """
        self.NTS = ts.size
        self.ts = ts

        self.j = 0

        self._STATE = 'provide_x0'

    def step_zo_forward(self):
        """
        Solve nominal differential equation using an Runge-Kutta scheme.
        """
        self.t[0] = self.ts[self.j]

        if self.j == self.NTS -1:
            self._STATE = 'finished'

        elif self.STATE == 'provide_x0':
            # NOTE: temporarily save initial value
            self._STATE = 'plot'

        elif self.STATE == 'plot':
            self._STATE = 'provide_f'

        elif self.STATE == 'provide_f':
            self._STATE = 'advance'

        elif self.STATE == 'advance':
            h = self.ts[self.j+1] - self.ts[self.j]
            self.x[:] = self.x + h*self.f
            self._STATE = 'plot'
            self.j += 1

    def init_fo_forward(self, ts):
        """
        Initialize memory for first order forward integration.

        Parameters
        ----------
        ts : array-like
            time grid for integration
        """
        self.NTS = ts.size
        self.ts = ts

        self._STATE = 'provide_x0'
        self.j = 0

    def step_fo_forward(self):
        """
        Solve nominal differential equation using an explicit Euler.
        """
        self.t[0] = self.ts[self.j]

        if self.j == self.NTS -1:
            self._STATE = 'finished'

        elif self.STATE == 'provide_x0':
            # NOTE: temporarily save initial value
            self._f_d = self._x_d.copy()
            self._STATE = 'plot'

        elif self.STATE == 'plot':
            self._STATE = 'provide_f_dot'

        elif self.STATE == 'provide_f_dot':
            self._STATE = 'advance'

        elif self.STATE == 'advance':
            h = self.ts[self.j+1] - self.ts[self.j]
            self._x_d[:] = self.x_d + h*self.f_d
            self._x[:] = self.x + h*self.f
            self._STATE = 'plot'
            self.j += 1


if __name__ == '__main__':

    def ffcn(f, t, x, p, u):
        f[0] = - p[0]*x[0]
        f[1] = - p[1]*x[1]

    def ref(xs, ts, p, u, x0):
        xs[:, 0] = (numpy.exp(-p[0]*ts)*x0[0])
        xs[:, 1] = (numpy.exp(-p[1]*ts)*x0[1])

    NTS = 101
    t0 = 0.0
    tf = 10.
    ts = numpy.linspace(t0, tf, NTS, endpoint=True)

    NX = 2
    NP = 2
    NU = 1
    x = numpy.array([2.0, 1.5])
    p = numpy.array([1.0, 0.7])
    u = numpy.array([0.0])

    xs = numpy.zeros([NTS, NX])
    rs = xs.copy()
    ref(rs, ts, p, u, x)

    ind = RcExplicitEuler(x)
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
    plt.plot(ts, xs, '-k', label="ee")
    plt.plot(ts, rs, ':r', label="ref")
    plt.legend(loc="best")
    plt.show()
