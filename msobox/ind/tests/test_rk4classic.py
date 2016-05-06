import os
import numpy

from numpy.testing import *
from numpy.testing.decorators import *

from msobox.ind.explicit_euler import ExplicitEuler
from msobox.ind.rk4classic import RK4Classic

from msobox.mf.python import BackendPython
from msobox.mf.mf_algopy import BackendAlgopy

try:
    from msobox.mf.tapenade import BackendTapenade
    tapenade_installed = True
except ImportError:
    tapenade_installed = False

try:
    from msobox.mf.pyadolc import BackendPyadolc
    pyadolc_installed = True
except ImportError:
    pyadolc_installed = False

DIR = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.dirname(DIR)
DIR = os.path.dirname(DIR)
DIR = os.path.dirname(DIR)


class Test_RK4Classic(TestCase):

    """."""

    @skipif(not pyadolc_installed)
    def test_forward_vs_reverse(self):
        """."""
        pyadolc = BackendPyadolc(
            os.path.join(DIR, './examples/python/bimolkat/ffcn.py')
        )
        e = RK4Classic(pyadolc)

        ts = numpy.linspace(0, 2, 50)
        x0 = numpy.ones(5)
        p = numpy.ones(5)
        q = numpy.zeros((4, ts.size, 2))
        q[0, :, 0] = 90.
        q[1:, :, 0] = 1.

        P = 1
        x0_dot = numpy.zeros((x0.size, P))
        p_dot = numpy.zeros((p.size, P))
        q_dot = numpy.zeros(q.shape + (P,))

        p_dot[:, 0] = 1.

        e.fo_forward(ts, x0, x0_dot, p, p_dot, q, q_dot)

        xs_bar = numpy.zeros(e.xs.shape)
        xs_bar[-1, 1] = 1.

        e.fo_reverse(xs_bar)

        a = numpy.sum(e.x0_bar * e.x0_dot[:, 0]) \
            + numpy.sum(e.p_bar * e.p_dot[:, 0]) \
            + numpy.sum(e.q_bar * e.q_dot[..., 0])
        b = numpy.sum(xs_bar * e.xs_dot[..., 0])

        assert_almost_equal(a, b)


if __name__ == "__main__":
    run_module_suite()
