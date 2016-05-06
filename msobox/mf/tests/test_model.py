"""Tests for the model function class."""

import os
import json
import numpy
import pytest

import scipy.linalg as lg

from copy import (deepcopy,)
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
# '''


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


@pytest.mark.parametrize("member", ["ffcn", "hfcn"])
def test_model_function_evaluation_from_mf_py(
    temp_mf_py_file, temp_md_file, member
):
    """."""
    # load back end
    mf = str(temp_mf_py_file)
    md = str(temp_md_file)
    model = Model(mf, md, verbose=True)

    # retrieve member
    function = getattr(model, member)

    # function declaration and dimensions
    f_dict = function._declaration
    f_dims = function._dimensions

    # define input values
    actual_args = [numpy.random.random(f_dims[arg]) for arg in f_dict["args"]]
    desired_args = deepcopy(actual_args)

    # get functions from globals() and model
    actual = getattr(model, member)
    desired = globals().get(member + "_py")

    # call functions
    actual(*actual_args)
    desired(*desired_args)

    print ""
    for i in range(len(actual_args)):
        print "actual:  ", actual_args[i]
        print "desired: ", desired_args[i]
        print "error:   ", lg.norm(desired_args[i] - actual_args[i])
        assert_allclose(actual_args[i], desired_args[i])
    print "successful!"


def test_model_function_assignment_from_mf_so(
    temp_mf_so_from_mf_f_file, temp_md_file
):
    """."""
    # load back end
    mf = str(temp_mf_so_from_mf_f_file)
    md = str(temp_md_file)
    model = Model(mf, md, verbose=True)

    assert hasattr(model, "ffcn")
    assert hasattr(model, "ffcn_d_xpu_v")
    assert hasattr(model, "hfcn")
    assert hasattr(model, "hfcn_d_xpu_v")


# ------------------------------------------------------------------------------
