"""Test for the pure python back-end."""
import os
import numpy

from numpy.testing import (TestCase, run_module_suite)
from msobox.mf.python import BackendPython
from fixtures import (PureBackendFixture,)

DIR = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.dirname(DIR)
DIR = os.path.dirname(DIR)
DIR = os.path.dirname(DIR)


class TestBackendPythonAcademic(TestCase, PureBackendFixture):

    """Test evaluation of pure python back end."""

    # load back end
    ppath = os.path.join(DIR, './examples/python/academic/ffcn.py')
    backend = BackendPython(ppath)

    # specify model dimensions
    NX = 2
    NP = 1
    NU = 0


if __name__ == "__main__":
    run_module_suite()