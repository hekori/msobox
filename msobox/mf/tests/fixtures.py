"""Collection of INDegrator unit test fixtures."""

import numpy

from numpy.testing import *

from msobox.mf.python import (BackendPython,)
from msobox.mf.mf_algopy import (BackendAlgopy,)
from msobox.mf.tapenade import (BackendTapenade,)
from msobox.mf.findiff import (BackendFiniteDifferences,)


class PureBackendFixture(object):

    """Test evaluation of pure back end using AD identities."""

    TOL = 6

    def test_forward_vs_reverse_evaluation(self):
        """
        Test forward vs reverse evaluation.

        Evaluate the AD-identity:

          y_bar.T * y_dot = x_bar.T * x_dot + p_bar.T * p_dot + u_bar.T * u_dot

        """
        # NOTE: Specify back-end when inheriting from this fixture class
        backend = self.backend

        # NOTE: Specify model dimensions when inheriting from this class
        NX = self.NX
        NP = self.NP
        NU = self.NU

        # define random inputs
        f = numpy.random.random((NX,))
        t = numpy.random.random((1,))
        x = numpy.random.random((NX,))
        p = numpy.random.random((NP,))
        u = numpy.random.random((NU,))

        P = NX + NP + NU
        f_dot = numpy.zeros((NX, P))
        x_dot = numpy.random.random((NX, P))
        p_dot = numpy.random.random((NP, P))
        u_dot = numpy.random.random((NU, P))

        # NOTE: These back-ends support more then one adjoint direction
        if isinstance(backend, (BackendPython,)):
            Q = NX
        # NOTE: These back-ends support *only one* adjoint direction
        elif isinstance(backend, (BackendAlgopy, BackendTapenade)):
            Q = 1
        else:
            err_str = 'This type of back-end does not exist.'
            raise Exception(err_str)

        f_bar = numpy.random.random((NX, Q))
        x_bar = numpy.zeros((NX, Q))
        p_bar = numpy.zeros((NP, Q))
        u_bar = numpy.zeros((NU, Q))

        # forward evaluation
        backend.ffcn_dot(f, f_dot, t, x, x_dot, p, p_dot, u, u_dot)
        f_tmp = f.copy()

        # backward evaluation
        backend.ffcn_bar(f, f_bar, t, x, x_bar, p, p_bar, u, u_bar)

        # check correctness of function evaluation
        assert_array_almost_equal(f, f_tmp, self.TOL)

        # check AD identity
        a = x_bar.T.dot(x_dot) + p_bar.T.dot(p_dot) + u_bar.T.dot(u_dot)
        b = f_bar.T.dot(f_dot)
        assert_array_almost_equal(a, b, self.TOL)


class BackendvsPythonFixture(object):

    """Test evaluation of specified back end vs pure python implementation."""

    TOL = 7

    def test_backend_ffcn_vs_pure_python(self):
        """
        Test nominal function evaluation of specified vs pure python back-end.
        """
        # NOTE: Specify back-end when inheriting from this fixture class
        backend = self.backend
        backend_python = self.backend_python

        # NOTE: Specify model dimensions when inheriting from this class
        NX = self.NX
        NP = self.NP
        NU = self.NU

        # define random inputs
        f = numpy.zeros((NX,))
        t = numpy.random.random((1,))
        x = numpy.random.random((NX,))
        p = numpy.random.random((NP,))
        u = numpy.random.random((NU,))

        # evaluate back-ends
        backend_python.ffcn(f, t, x, p, u)
        expected = f.copy()

        backend.ffcn(f, t, x, p, u)
        actual = f.copy()

        # compare evaluations
        assert_almost_equal(expected, actual, self.TOL)

    def test_backend_ffcn_dot_vs_pure_python(self):
        """
        Test nominal function and first-order forward derivative evaluation.
        """
        # NOTE: Specify back-end when inheriting from this fixture class
        backend = self.backend
        backend_python = self.backend_python

        # NOTE: Specify model dimensions when inheriting from this class
        NX = self.NX
        NP = self.NP
        NU = self.NU

        # random inputs
        f = numpy.zeros((NX,))
        t = numpy.random.random((1,))
        x = numpy.random.random((NX,))
        p = numpy.random.random((NP,))
        u = numpy.random.random((NU,))

        P = NX + NP + NU
        f_dot = numpy.zeros((NX, P))
        x_dot = numpy.random.random((NX, P))
        p_dot = numpy.random.random((NP, P))
        u_dot = numpy.random.random((NU, P))

        # evaluate back-ends
        backend_python.ffcn_dot(f, f_dot, t, x, x_dot, p, p_dot, u, u_dot)
        expected = f.copy()
        expected_dot = f_dot.copy()

        backend.ffcn_dot(f, f_dot, t, x, x_dot, p, p_dot, u, u_dot)
        actual = f.copy()
        actual_dot = f_dot.copy()

        # compare values
        assert_almost_equal(expected, actual, self.TOL)
        assert_almost_equal(expected_dot, actual_dot, self.TOL)

    def test_backend_ffcn_bar_vs_pure_python(self):
        """
        Test nominal function and first-order reverse derivative evaluation.
        """
        # NOTE: Specify back-end when inheriting from this fixture class
        backend = self.backend
        backend_python = self.backend_python

        # NOTE: Specify model dimensions when inheriting from this class
        NX = self.NX
        NP = self.NP
        NU = self.NU

        # random inputs
        f = numpy.zeros((NX,))
        t = numpy.random.random((1,))
        x = numpy.random.random((NX,))
        p = numpy.random.random((NP,))
        u = numpy.random.random((NU,))

        # NOTE: These back-ends support more then one adjoint direction
        if isinstance(backend, (BackendPython,)):
            Q = NX
        # NOTE: These back-ends support *only one* adjoint direction
        elif isinstance(backend, (BackendAlgopy, BackendTapenade)):
            Q = 1
        else:
            err_str = 'This type of back-end does not exist.'
            raise Exception(err_str)

        f_bar = numpy.random.random((NX, Q))
        x_bar = numpy.zeros((NX, Q))
        p_bar = numpy.zeros((NP, Q))
        u_bar = numpy.zeros((NU, Q))

        # evaluate back-ends
        backend_python.ffcn_bar(f, f_bar, t, x, x_bar, p, p_bar, u, u_bar)
        expected = f.copy()
        expected_x_bar = x_bar.copy()
        expected_p_bar = p_bar.copy()
        expected_u_bar = u_bar.copy()

        x_bar[...] = 0.0
        p_bar[...] = 0.0
        u_bar[...] = 0.0
        backend.ffcn_bar(f, f_bar, t, x, x_bar, p, p_bar, u, u_bar)
        actual = f.copy()
        actual_x_bar = x_bar.copy()
        actual_p_bar = p_bar.copy()
        actual_u_bar = u_bar.copy()

        # compare values
        assert_almost_equal(expected, actual, self.TOL)
        assert_almost_equal(expected_x_bar, actual_x_bar, self.TOL)
        assert_almost_equal(expected_p_bar, actual_p_bar, self.TOL)
        assert_almost_equal(expected_u_bar, actual_u_bar, self.TOL)
