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

if __name__ == '__main__':

    def ffcn(f, t, x, p, u):
        f[0] = - p[0]*x[0]

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

    ind = RcExplicitEuler(NX=1)
    ind.init_zo_forward(ts=ts)

    # reverse communication loop
    while True:
        print "STATE: ", ind.STATE, "(", ind.j, "/", ind.NTS,")"

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
