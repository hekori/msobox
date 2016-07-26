"""Tests for msobox Function class."""

import os
import json
import numpy
import pytest
import tempfile

import scipy.linalg as lg

from cffi import (FFI,)
from collections import (OrderedDict,)
from numpy.testing import (TestCase, run_module_suite)
from numpy.testing import (assert_equal, assert_allclose)

from conftest import (md_dict,)
from conftest import (ffcn_py, ffcn_d_xpu_v_py,)
from conftest import (ffcn_d_xpu_v_d_xx_dpp_duu_d_v_py,)
from conftest import (ffcn_b_xpu_py,)
from conftest import (hfcn_py, hfcn_d_xpu_v_py,)
# from conftest import (hfcn_d_xpu_v_d_xx_dpp_duu_d_v_py,)
from conftest import (hfcn_b_xpu_py,)

from msobox.mf.model import (import_module_from_file)
from msobox.mf.model import (import_shared_library)
from msobox.mf.functions import (Function,)


# ------------------------------------------------------------------------------
# LOCAL FIXTURES

# ------------------------------------------------------------------------------
# ACTUAL TESTS
def test_msobox_function_interface_on_ffcn_so_calling_ffcn(
    temp_mf_so_from_mf_f_file
):
    """."""
    # path to shared library
    so_path = str(temp_mf_so_from_mf_f_file)
    # print "so_path: ", so_path

    # load shared library as module
    module = import_shared_library(so_path)

    # initialize foreign function interface for library
    header = """
    void ffcn_(double *f, double *t, double *x, double *p, double *u);
    void ffcn_d_xpu_v_(
      double *f, double *f_d,
      double *t,
      double *x, double *x_d,
      double *p, double *p_d,
      double *u, double *u_d,
      int *nbdirs
    );
    """
    # open shared library
    ffi = FFI()
    ffi.cdef(header)
    module = ffi.dlopen(so_path)

    # function declaration and dimensions
    func = {
        "type": "ffcn",
        "name": "ffcn",
        "args": ["f", "t", "x", "p", "u"],
        "deriv": []
    }
    dims = {"f": 5, "t": 1, "x": 5, "p": 5, "u": 4}

    # create function
    ffcn = Function(module, dims, func, ffi=ffi, verbose=False)

    # define input values
    t = numpy.random.random(dims["t"])
    x = numpy.random.random(dims["x"])
    p = numpy.random.random(dims["p"])
    u = numpy.random.random(dims["u"])

    # define output variables
    desired = numpy.zeros(dims["f"])
    actual = numpy.zeros(dims["f"])

    # call functions
    ffcn(actual, t, x, p, u)
    ffcn_py(desired, t, x, p, u)

    # compare values
    print ""
    print "actual:  ", actual
    print "desired: ", desired
    print "error:   ", lg.norm(desired - actual)
    assert_allclose(actual, desired)


def test_msobox_function_interface_on_ffcn_so_calling_hfcn(
    temp_mf_so_from_mf_f_file
):
    """."""
    # path to shared library
    so_path = str(temp_mf_so_from_mf_f_file)
    # print "so_path: ", so_path

    # load shared library as module
    module = import_shared_library(so_path)

    # initialize foreign function interface for library
    header = """
    void ffcn_(double *f, double *t, double *x, double *p, double *u);
    void hfcn_(double *h, double *t, double *x, double *p, double *u);
    void ffcn_d_xpu_v_(
      double *f, double *f_d,
      double *t,
      double *x, double *x_d,
      double *p, double *p_d,
      double *u, double *u_d,
      int *nbdirs
    );
    """
    # open shared library
    ffi = FFI()
    ffi.cdef(header)
    module = ffi.dlopen(so_path)

    # function declaration and dimensions
    func = {
        "type": "hfcn",
        "name": "hfcn",
        "args": ["h", "t", "x", "p", "u"],
        "deriv": []
    }
    dims = {"h": 3, "t": 1, "x": 5, "p": 5, "u": 4}

    # create function
    hfcn = Function(module, dims, func, ffi=ffi, verbose=False)

    # define input values
    t = numpy.random.random(dims["t"])
    x = numpy.random.random(dims["x"])
    p = numpy.random.random(dims["p"])
    u = numpy.random.random(dims["u"])

    # define output variables
    desired = numpy.zeros(dims["h"])
    actual = numpy.zeros(dims["h"])

    # call functions
    hfcn(actual, t, x, p, u)
    hfcn_py(desired, t, x, p, u)

    # compare values
    print ""
    print "actual:  ", actual
    print "desired: ", desired
    print "error:   ", lg.norm(desired - actual)
    assert_allclose(actual, desired)


def test_msobox_function_interface_on_ffcn_so_calling_ffcn_d_xpu_v(
    temp_mf_so_from_mf_f_file
):
    # path to shared library
    so_path = str(temp_mf_so_from_mf_f_file)
    # print "so_path: ", so_path

    # load shared library as module
    module = import_shared_library(so_path)

    # initialize foreign function interface for library
    header = """
    void ffcn_(double *f, double *t, double *x, double *p, double *u);
    void ffcn_d_xpu_v_(
      double *f, double *f_d,
      double *t,
      double *x, double *x_d,
      double *p, double *p_d,
      double *u, double *u_d,
      int *nbdirs
    );
    """
    # open shared library
    ffi = FFI()
    ffi.cdef(header)
    module = ffi.dlopen(so_path)

    # function declaration and dimensions
    func = {
        "type": "ffcn",
        "name": "ffcn_d_xpu_v",
        "args": ["f", "f_d", "t", "x", "x_d", "p", "p_d", "u", "u_d", "nbdirs"],
        "deriv": []
    }
    dims = {"f": 5, "t": 1, "x": 5, "p": 5, "u": 4}

    # define input values
    t = numpy.random.random(dims["t"])
    x = numpy.random.random(dims["x"])
    p = numpy.random.random(dims["p"])
    u = numpy.random.random(dims["u"])

    # define output variables
    P = numpy.sum(dims.values())
    x_d = numpy.random.random([dims["x"], P])
    p_d = numpy.random.random([dims["p"], P])
    u_d = numpy.random.random([dims["u"], P])

    # create function
    ffcn_d_xpu_v = Function(module, dims, func, ffi=ffi, verbose=False)

    # define output variables
    desired = numpy.zeros(dims["f"])
    actual = numpy.zeros(dims["f"])
    desired_d_xpu_v = numpy.zeros([dims["f"], P])
    actual_d_xpu_v = numpy.zeros([dims["f"], P])

    # call functions
    ffcn_d_xpu_v(actual, actual_d_xpu_v, t, x, x_d, p, p_d, u, u_d)
    ffcn_d_xpu_v_py(desired, desired_d_xpu_v, t, x, x_d, p, p_d, u, u_d)

    # compare values
    print ""
    print "actual:  ", actual
    print "desired: ", desired
    print "error:   ", lg.norm(desired - actual)
    assert_allclose(actual, desired)
    print ""

    print "actual_d_xpu_v: \n", actual_d_xpu_v
    print "desired_d_xpu_v: \n", desired_d_xpu_v
    print "error:   ", lg.norm(desired_d_xpu_v - actual_d_xpu_v)
    assert_allclose(actual_d_xpu_v, desired_d_xpu_v)
    print "successful!"


def test_msobox_function_interface_on_ffcn_py_calling_ffcn(temp_mf_py_file):
    # path to shared library
    f_path = str(temp_mf_py_file)
    # print "so_path: ", so_path

    # load shared library as module
    module = import_module_from_file(f_path, verbose=True)

    # function declaration and dimensions
    func = {
        "type": "ffcn",
        "name": "ffcn",
        "args": ["f", "t", "x", "p", "u"],
        "deriv": []
    }
    dims = {"f": 5, "t": 1, "x": 5, "p": 5, "u": 4}

    # define input values
    t = numpy.random.random(dims["t"])
    x = numpy.random.random(dims["x"])
    p = numpy.random.random(dims["p"])
    u = numpy.random.random(dims["u"])

    # create function
    func = Function(module, dims, func, verbose=False)

    # define output variables
    desired = numpy.zeros(dims["f"])
    actual = numpy.zeros(dims["f"])

    # call functions
    func(actual, t, x, p, u)
    ffcn_py(desired, t, x, p, u)

    # compare values
    print ""
    print "actual:  ", actual
    print "desired: ", desired
    print "error:   ", lg.norm(desired - actual)
    assert_allclose(actual, desired)
    print "successful!"


def test_msobox_function_interface_on_ffcn_py_calling_ffcn_d_xpu_v(
    temp_mf_py_file
):
    # path to shared library
    f_path = str(temp_mf_py_file)
    # print "so_path: ", so_path

    # load shared library as module
    module = import_module_from_file(f_path, verbose=True)

    # function declaration and dimensions
    func = {
        "type": "ffcn",
        "name": "ffcn_d_xpu_v",
        "args": ["f", "f_d", "t", "x", "x_d", "p", "p_d", "u", "u_d"],
        "deriv": []
    }
    dims = {"f": 5, "t": 1, "x": 5, "p": 5, "u": 4}

    # define input values
    t = numpy.random.random(dims["t"])
    x = numpy.random.random(dims["x"])
    p = numpy.random.random(dims["p"])
    u = numpy.random.random(dims["u"])

    # define output variables
    P = numpy.sum(dims.values())
    x_d = numpy.random.random([dims["x"], P])
    p_d = numpy.random.random([dims["p"], P])
    u_d = numpy.random.random([dims["u"], P])

    # create function
    ffcn_d_xpu_v = Function(module, dims, func, verbose=False)

    # define output variables
    desired = numpy.zeros(dims["f"])
    actual = numpy.zeros(dims["f"])
    desired_d_xpu_v = numpy.zeros([dims["f"], P])
    actual_d_xpu_v = numpy.zeros([dims["f"], P])

    # call functions
    ffcn_d_xpu_v(actual, actual_d_xpu_v, t, x, x_d, p, p_d, u, u_d)
    ffcn_d_xpu_v_py(desired, desired_d_xpu_v, t, x, x_d, p, p_d, u, u_d)

    # compare values
    print ""
    print "actual:  ", actual
    print "desired: ", desired
    print "error:   ", lg.norm(desired - actual)
    assert_allclose(actual, desired)
    print ""

    print "actual_d_xpu_v: \n", actual_d_xpu_v
    print "desired_d_xpu_v: \n", desired_d_xpu_v
    print "error:   ", lg.norm(desired_d_xpu_v - actual_d_xpu_v)
    assert_allclose(actual_d_xpu_v, desired_d_xpu_v)
    print "successful!"





# ------------------------------------------------------------------------------
