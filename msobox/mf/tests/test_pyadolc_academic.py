import os
import numpy

import unittest

from numpy.testing import (TestCase, run_module_suite)
from numpy.testing.decorators import (setastest, skipif)

from msobox.mf.python import (BackendPython,)
from .fixtures import (PureBackendFixture, BackendvsPythonFixture)

try:
    from msobox.mf.pyadolc import BackendPyadolc
except ImportError:
    err_str = 'Needs pyadolc installed to run backend_fortran.'
    raise unittest.SkipTest(err_str)

DIR = os.path.dirname(os.path.abspath(__file__))


class TestBackendPyAdolc(TestCase, PureBackendFixture, BackendvsPythonFixture):

    """Test evaluation of pyadolc back end."""

    # load back end
    backend_python = BackendPython('./examples/python/academic/ffcn.py')
    backend = BackendPyadolc('./examples/pyadolc/academic/ffcn.py')

    # define random inputs
    NX = 5
    NP = 5
    NU = 4


if __name__ == "__main__":
    run_module_suite()
