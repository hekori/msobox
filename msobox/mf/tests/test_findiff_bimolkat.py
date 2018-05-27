import os
import numpy

from numpy.testing import (TestCase, run_module_suite)
from numpy.testing.decorators import (setastest,)

from msobox.mf.python import (BackendPython,)
from msobox.mf.findiff import (BackendFiniteDifferences,)
from .fixtures import (BackendvsPythonFixture,)

DIR = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.dirname(DIR)
DIR = os.path.dirname(DIR)
DIR = os.path.dirname(DIR)


class TestBackendFiniteDifferences(TestCase, BackendvsPythonFixture):

    """
    Test evaluations of finite differences back end.

    NOTE: No reverse mode is available using finite differences.
    """

    # load back end
    ppath = os.path.join(DIR, './examples/python/bimolkat/ffcn.py')
    backend = BackendFiniteDifferences(ppath)
    backend_python = BackendPython(ppath)

    # specify model dimensions
    NX = 5
    NP = 5
    NU = 4

    # disable adjoint derivatives tests
    @setastest(False)
    def test_backend_ffcn_bar_vs_pure_python(self):
        """
        Test nominal function and first-order reverse derivative evaluation.

        NOTE: No reverse mode is available using finite differences.
        """
        super(TestBackendFiniteDifferences, self) \
            .test_backend_ffcn_bar_vs_pure_python()


if __name__ == "__main__":
    run_module_suite()
