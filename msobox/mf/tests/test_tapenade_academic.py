import os
import numpy
import unittest

from numpy.testing import (TestCase, run_module_suite)
from numpy.testing.decorators import (setastest, skipif)

from msobox.ad.tapenade import (Differentiator,)
from msobox.mf.python import (BackendPython,)
from fixtures import (PureBackendFixture, BackendvsPythonFixture)

try:
    from msobox.mf.tapenade import BackendTapenade
except ImportError:
    err_str = 'Needs Tapenade installed to run backend_fortran.'
    raise unittest.SkipTest(err_str)

DIR = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.dirname(DIR)
DIR = os.path.dirname(DIR)
DIR = os.path.dirname(DIR)


class TestBackendTapenade(TestCase, PureBackendFixture, BackendvsPythonFixture):

    """Test evaluation of fortran back end."""

    # differentiate model function using tapenade
    fortran_path = os.path.join(DIR, './examples/fortran/academic/ffcn.f')
    d = Differentiator(fortran_path)

    # load back end
    ppath = os.path.join(DIR, './examples/python/academic/ffcn.py')
    backend_python = BackendPython(ppath)
    fpath = os.path.join(DIR, './examples/fortran/academic/libproblem.so')
    backend = BackendTapenade(fpath)

    # define random inputs
    NX = 2
    NP = 1
    NU = 0


if __name__ == "__main__":
    run_module_suite()
