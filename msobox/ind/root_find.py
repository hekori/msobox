# -*- coding: utf-8 -*-
"""Efficient root finding algorithms."""

import numpy


# ------------------------------------------------------------------------------
def root_find(s, t, t0, t1, func, tol=1e-08, eps=1e-16):
    """
    Root-finding algorithm due to Brent & Decker [1].

    Searches for root ts of function func in the interval [t0, t1] from a given
    initial guess t up to a given tolerance tol using inverse quadratic
    interpolation safeguarded by *regula falsi* and a bisection strategy.

    Parameters
    ----------
    s : double
        value of at expected root

    t : double
        initial guess of root of function

    t0 : double
        left border of search interval

    t1 : double
        right border of search interval

    func : univariate function with interface f(t)
        function analyze for roots

    tol : double
        root finding accuracy tolerance up to what a root is accepted,
        defaults to single precision 1e-08

    eps : double
        machine precision, defaults to double precision 1e-16

    Returns
    -------
    root : double
        accurate estimate of actual root up to a given tolerance

    value : double
        accurate estimate of actual root up to a given tolerance

    """
    # a, b, c: abscissae
    a = t0
    b = t1
    c = a

    # fa, fb, fc: corresponding function values
    fa = func(a)
    fb = func(b)
    fc = fa

    # Main iteration loop
    n_iter = 0
    while True:
        n_iter += 1
        prev_step = b-a  # Distance from the last but one to the last approx.
        tol_act = 0.0   # Actual tolerance
        new_step = 0.0   # Step at this iteration

        # Interpolation step is calculated in the form p/q
        # division operations is delayed until the last moment
        p = 0.0
        q = 0.0

        if abs(fc) < abs(fb):
            # Swap data for b to be the best approximation
            a = b
            b = c
            c = a
            fa = fb
            fb = fc
            fc = fa

        tol_act = 2.0*eps*abs(fb) + tol/2.0
        new_step = (c - b)/2.0

        # Acceptable approximation found ?
        if abs(new_step) <= tol_act or fb == 0.0:
            root = b
            value = fb
            # print 'finished after {} iterations.'.format(n_iter)
            return (root, value)

        # NOTE: Interpolation may be tried if prev_step was large enough and in
        #       true direction
        if abs(prev_step) >= tol_act and abs(fa) > abs(fb):
            cb = c-b

            if a == c:
                # NOTE: If we have only two distinct points, linear
                #       interpolation can only be applied
                t1 = fb / fa
                p = cb * t1
                q = 1.0 - t1
            else:
                # Inverse quadratic interpolation
                q = fa/fc
                t1 = fb/fc
                t2 = fb/fa
                p = t2 * (cb*q*(q - t1) - (b - a)*(t1 - 1.0))
                q = (q - 1.0) * (t1 - 1.0) * (t2 - 1.0)

            # NOTE: p was calculated with the opposite sign make p positive and
            #       assign possible minus to q
            if p > 0.0:
                q = -q
            else:
                p = -p

            # NOTE: If b+p/q falls in [b,c] and isn't too large, it is accepted
            # NOTE: If p/q is too large then the bisection procedure can reduce
            #       [b,c] range to a larger extent
            if (p < 0.75*cb*q - abs(tol_act*q)/2.0
            and p < abs(prev_step*q/2.0)):
                new_step = p/q

        # Adjust the step to be not less than tolerance
        if abs(new_step) < tol_act:
            if new_step > 0.0:
                new_step = tol_act
            else:
                new_step = -tol_act

        # Save the previous approximate
        a = b
        fa = fb

        # Do step to a new approximation
        b += new_step
        fb = func(b)

        # Adjust c for it to have a sign opposite to that of b
        if (fb > 0 and fc > 0) or (fb < 0 and fc < 0):
            c = a
            fc = fa


# ------------------------------------------------------------------------------
class RootFind(object):
    """
    Root-finding algorithm due to Brent & Decker [1].

    Searches for root ts of function func in the interval [t0, t1] from a given
    initial guess t up to a given tolerance tol using inverse quadratic
    interpolation safeguarded by *regula falsi* and a bisection strategy.
    """

    # --------------------------------------------------------------------------
    def _get_STATE(self):
        return self._STATE

    STATE = property(
        _get_STATE, None, None,
        "Current state of the reverse communication interface."
    )

    # --------------------------------------------------------------------------
    def _get_t0(self):
        return self._t0

    def _set_t0(self, value):
        self._t0[...] = value

    t0 = property(
        _get_t0, _set_t0, None, ""
    )

    # --------------------------------------------------------------------------
    def _get_t1(self):
        return self._t1

    def _set_t1(self, value):
        self._t1[...] = value

    t1 = property(
        _get_t1, _set_t1, None, ""
    )

    # --------------------------------------------------------------------------
    def _get_t(self):
        return self._t

    def _set_t(self, value):
        self._t[...] = value

    t = property(
        _get_t, _set_t, None, ""
    )

    # --------------------------------------------------------------------------
    def _get_f0(self):
        return self._f0

    def _set_f0(self, value):
        self._f0[...] = value

    f0 = property(
        _get_f0, _set_f0, None, ""
    )


    # --------------------------------------------------------------------------
    def _get_f(self):
        return self._f

    def _set_f(self, value):
        self._f[...] = value

    f = property(
        _get_f, _set_f, None, ""
    )


    # --------------------------------------------------------------------------
    def _get_f1(self):
        return self._f1

    def _set_f1(self, value):
        self._f1[...] = value

    f1 = property(
        _get_f1, _set_f1, None, ""
    )

    # --------------------------------------------------------------------------
    def __init__(self, tol=1e-10, eps=1e-16):
        """
        Detect root of given function.
        Parameters
        ----------
        tol : double
            root finding accuracy tolerance up to what a root is accepted,
            defaults to single precision 1e-08

        eps : double
            machine precision, defaults to double precision 1e-16

        """
        self.TOL = tol
        self.EPS = eps

        # t0, t1, t: abscissae
        self._t0 = numpy.zeros(1, dtype=float)
        self._t1 = numpy.zeros(1, dtype=float)
        self._t = numpy.zeros(1, dtype=float)

        # f0, f1, f: corresponding function values
        self._f0 = numpy.zeros(1, dtype=float)
        self._f1 = numpy.zeros(1, dtype=float)
        self._f = numpy.zeros(1, dtype=float)

        self.n_iter = 0
        self._STATE = 'plot'

    def initialize(self, t0, t1, f0, f1):
        """
        Parameters
        ----------
        t0 : double
            left border of search interval

        t1 : double
            right border of search interval

        f0 : double
            function evaluation left border

        f1 : double
            function evaluation right border

        """
        # fa, fb, fc: corresponding function values
        self.f0 = f0
        self.f1 = f1
        self.f = self.f0

        # a, b, c: abscissae
        self.t0 = t0
        self.t1 = t1
        self.t = self.t0

        self.n_iter = 0
        self._STATE = 'plot'

    def root_find(self, func):
        """
        Returns
        -------
        root : double
            accurate estimate of actual root up to a given tolerance

        value : double
            accurate estimate of actual root up to a given tolerance
        """
        self.n_iter += 1
        tol_act = 0.0   # Actual tolerance
        new_step = 0.0   # Step at this iteration

        # Interpolation step is calculated in the form p/q
        # division operations is delayed until the last moment
        p = 0.0
        q = 0.0

        # Distance from the last but one to the last approx.
        h = self.t1 - self.t0

        # NOTE Swap data for b to be the best approximation
        if abs(self.f) < abs(self.f1):
            self.t0 = self.t1
            self.t1 = self.t
            self.t = self.t

            self.t0 = self.f1
            self.f1 = self.f
            self.f = self.f0

        TOL = 2.0*self.EPS*abs(self.f1) + self.TOL/2.0
        h_ = (self.t - self.t1)/2.0

        # Acceptable approximation found ?
        if abs(h_) < TOL or abs(self.f1) < TOL:
            root = self.t1
            value = self.f1
            # print 'finished after {} iterations.'.format(n_iter)
            self._STATE = "finished"
            return (root, value)

        # NOTE: Interpolation may be tried if prev_step was large enough and in
        #       true direction
        if abs(h) >= TOL and abs(self.fa) > abs(self.fb):
            # NOTE common factor compute it once
            cb = self.t - self.t1

            if self.t0 == self.t:
                # NOTE: If we have only two distinct points, only linear
                #       interpolation can be applied
                t1 = self.fb/self.fa
                p = cb * t1
                q = 1.0 - t1
            else:
                # NOTE: this implies three distinct points and inverse
                #       quadratic interpolation is applicable
                q = self.fa/self.fc
                t1 = self.fb/self.fc
                t2 = self.fb/self.fa
                p = t2 * (cb*q*(q - t1) - (self.b - self.a)*(t1 - 1.0))
                q = (q - 1.0) * (t1 - 1.0) * (t2 - 1.0)

            # NOTE: p was calculated with the opposite sign make p positive and
            #       assign possible minus to q
            if p > 0.0:
                q = -q
            else:
                p = -p

            # NOTE: If b+p/q falls in [b,c] and isn't too large, it is accepted
            # NOTE: If p/q is too large then the bisection procedure can reduce
            #       [b,c] range to a larger extent
            if (p < 0.75*cb*q - abs(tol_act*q)/2.0
            and p < abs(prev_step*q/2.0)):
                new_step = p/q

        # Adjust the step to be not less than tolerance
        if abs(new_step) < tol_act:
            if new_step > 0.0:
                new_step = tol_act
            else:
                new_step = -tol_act

        # Save the previous approximate
        self.a = self.b
        self.fa = self.fb

        # Do step to a new approximation
        self.b += new_step
        func(self.fb, self.b)

        # Adjust c for it to have a sign opposite to that of b
        if (self.fb > 0 and self.fc > 0) or (self.fb < 0 and self.fc < 0):
            self.c = self.a
            self.fc = self.fa


    # --------------------------------------------------------------------------
    def retrieve_values(self, As, Bs, Cs, fAs, fBs, fCs):
        """Append current values to lists."""
        As.append(self.t0)
        Bs.append(self.t1)
        Cs.append(self.t)

        fAs.append(self.f0)
        fBs.append(self.f1)
        fCs.append(self.f)


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from time import sleep
    import scipy.linalg as lg
    import matplotlib.pyplot as plt

    def sfcn(s, t):
        """Simple quadratic function with roots in -1 and 0.5."""
        s[...] = (t + 1.0)*(t - 0.5)

    NTS = 1001
    ts = numpy.linspace(-2.0, 2.0, NTS, endpoint=True)
    xs = numpy.zeros(ts.shape)
    sfcn(xs, ts)

    plt.ion()
    fig, ax = plt.subplots()
    ax.plot(ts, xs, "-k", label="f")
    ax.set_title('root find algorithm')

    as_options = {'marker': 'x', 'ms': 10, 'color': 'b', 'ls': '', 'label': 'as'}
    bs_options = {'marker': '.', 'ms': 10, 'color': 'r', 'ls': '', 'label': 'bs'}
    cs_options = {'marker': '+', 'ms': 10, 'color': 'g', 'ls': '', 'label': 'cs'}
    as_plt, = ax.plot([], [], **as_options)
    bs_plt, = ax.plot([], [], **bs_options)
    cs_plt, = ax.plot([], [], **cs_options)

    ax.grid()
    ax.legend(loc="best")
    fig.tight_layout()
    plt.show()
    plt.pause(0.0001)
    fig.savefig("root_find.pdf", dpi=200)

    As = []
    Bs = []
    Cs = []

    fAs = []
    fBs = []
    fCs = []

    t = numpy.zeros(1)
    s = numpy.zeros(1)

    t0 = t.copy()
    t0[0] = 0.0

    t1 = t.copy()
    t1[0] = 1.0

    s0 = s.copy()
    sfcn(s0, t0)

    s1 = s.copy()
    sfcn(s1, t1)

    print((t0, s0))
    print((t1, s1))
    rf = RootFind()
    rf.initialize(t0=t0, f0=s0, t1=t1, f1=s1)

    while True:
        rf.retrieve_values(As, Bs, Cs, fAs, fBs, fCs)

        as_plt.set_xdata(As)
        as_plt.set_ydata(fAs)

        bs_plt.set_xdata(Bs)
        bs_plt.set_ydata(fBs)

        cs_plt.set_xdata(Cs)
        cs_plt.set_ydata(fCs)

        ret = rf.root_find(sfcn)
        fig.canvas.draw()
        plt.pause(0.0001)

        if rf.STATE == 'finished':
            print(("root:  ", ret[0]))
            print(("value: ", ret[1]))
            print('done')
            break

        print("sleeping")
        sleep(1.0)

    print("sleeping")
    sleep(2.0)
    fig.savefig("root_find.pdf", dpi=200)

