"""Test for the algopy back-end."""

import os
import numpy

from numpy.testing import (TestCase, run_module_suite)

from msobox.mf.python import (BackendPython,)
from msobox.mf.mf_algopy import (BackendAlgopy,)
from .fixtures import (PureBackendFixture, BackendvsPythonFixture)

DIR = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.dirname(DIR)
DIR = os.path.dirname(DIR)
DIR = os.path.dirname(DIR)


class TestBackendAlgopy(TestCase, PureBackendFixture, BackendvsPythonFixture):

    """Test evaluations of algopy back end."""

    # load back end
    ppath = os.path.join(DIR, './examples/python/bimolkat/ffcn.py')
    backend = BackendAlgopy(ppath)
    backend_python = BackendPython(ppath)

    # specify model dimensions
    NX = 5
    NP = 5
    NU = 4


if __name__ == "__main__":
    run_module_suite()
