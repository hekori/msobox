"""Tests for the model function class."""

import os
import json
import numpy
import pytest

from collections import (OrderedDict,)
from numpy.testing import (TestCase, run_module_suite)
from numpy.testing import (assert_equal, assert_allclose)

from conftest import (md_dict,)
from conftest import (ffcn_py, ffcn_d_xpu_v_py, ffcn_d_xpu_v_d_xx_dpp_duu_d_py)
from conftest import (hfcn_py, hfcn_d_xpu_v_py,)  # hfcn_d_xpu_v_d_xx_dpp_duu_d_py)

from msobox.mf.model import (Model,)


# ------------------------------------------------------------------------------
# LOCAL FIXTURES
#'''
def test_setup_of_model_definitions_from_file(temp_mf_py_file, temp_md_file):
    """Check setup of model definitions from file."""
    # load back end
    mf = str(temp_mf_py_file)
    md = str(temp_md_file)
    model = Model(mf, md, verbose=True)

    actual = model.definitions
    desired = md_dict

    assert actual == desired


def test_setup_of_model_definitions_from_json(temp_mf_py_file, temp_md_json):
    """Check setup of model definitions from json."""
    # load back end
    mf = str(temp_mf_py_file)
    md = temp_md_json
    model = Model(mf, md, verbose=True)

    actual = model.definitions
    desired = md_dict

    assert actual == desired


def test_setup_of_model_definitions_from_dict(temp_mf_py_file, temp_md_dict):
    """Check setup of model definitions from dictionary."""
    # load back end
    mf = str(temp_mf_py_file)
    md = temp_md_dict
    model = Model(mf, md, verbose=True)

    actual = model.definitions
    desired = md_dict

    assert actual == desired


# ------------------------------------------------------------------------------
def test_setup_of_model_dimensions_dict(temp_mf_py_file, temp_md_file):
    """Check assigned model dimensions."""
    # load back end
    mf = str(temp_mf_py_file)
    md = str(temp_md_file)
    model = Model(mf, md, verbose=True)

    actual = model.dimensions
    desired = md_dict["dims"]

    assert actual == desired
#'''


# ------------------------------------------------------------------------------
# ACTUAL TESTS
def test_model_function_assignment_from_mf_py(temp_mf_py_file, temp_md_file):
    """."""
    # load back end
    mf = str(temp_mf_py_file)
    md = str(temp_md_file)
    model = Model(mf, md, verbose=True)

    assert hasattr(model, "ffcn")
    assert hasattr(model, "ffcn_d_xpu_v")
    assert hasattr(model, "hfcn")
    assert hasattr(model, "hfcn_d_xpu_v")


'''
def test_model_function_assignment_from_mf_so(temp_mf_py_file, temp_md_file):
    """."""
    # load back end
    mf = str(temp_mf_py_file)
    md = str(temp_md_file)
    model = Model(mf, md, verbose=True)

    assert hasattr(model, "ffcn")
    assert hasattr(model, "ffcn_d_xpu_v")
    assert hasattr(model, "hfcn")
    assert hasattr(model, "hfcn_d_xpu_v")
'''


# ------------------------------------------------------------------------------
