"""Tests for msobox Function class."""

import os
import json
import numpy
import pytest
import tempfile
import subprocess

import scipy.linalg as lg

from cffi import (FFI,)
from collections import (OrderedDict,)
from numpy.testing import (TestCase, run_module_suite)
from numpy.testing import (assert_equal, assert_allclose)

from msobox.mf.model import (import_module_from_file)
from msobox.mf.model import (import_shared_library)
from msobox.mf.functions import (Function,)


# ------------------------------------------------------------------------------
# PYTHON REFERENCE IMPLEMENTATION
def ffcn_py(f, t, x, p, u):
    """Dummy for test cases."""
    f[0] = x[0] + p[0] + u[0]
    f[1] = x[1] + p[1] + t*u[0]
    f[2] = x[2] + p[2] + u[1]
    f[3] = x[3] + p[3] + u[2]
    f[4] = x[4] + p[4] + u[3]


def ffcn_d_xpu_v_py(f, f_d, t, x, x_d, p, p_d, u, u_d):
    """Dummy for test cases."""
    f_d[0] = x_d[0] + p_d[0] + u_d[0]
    f_d[1] = x_d[1] + p_d[1] + t*u_d[0]
    f_d[2] = x_d[2] + p_d[2] + u_d[1]
    f_d[3] = x_d[3] + p_d[3] + u_d[2]
    f_d[4] = x_d[4] + p_d[4] + u_d[3]

    f[0] = x[0] + p[0] + u[0]
    f[1] = x[1] + p[1] + t*u[0]
    f[2] = x[2] + p[2] + u[1]
    f[3] = x[3] + p[3] + u[2]
    f[4] = x[4] + p[4] + u[3]


def ffcn_b_xpu_py(f, f_b, t, x, x_b, p, p_b, u, u_b):
    """Dummy for test cases."""
    f[0] = x[0] + p[0] + u[0]
    f[1] = x[1] + p[1] + t*u[0]
    f[2] = x[2] + p[2] + u[1]
    f[3] = x[3] + p[3] + u[2]
    f[4] = x[4] + p[4] + u[3]

    # TODO calculate derivative
    f_b[0] = x_b[0] + p_b[0] + u_b[0]
    f_b[1] = x_b[1] + p_b[1] + t*u_b[0]
    f_b[2] = x_b[2] + p_b[2] + u_b[1]
    f_b[3] = x_b[3] + p_b[3] + u_b[2]
    f_b[4] = x_b[4] + p_b[4] + u_b[3]


def ffcn_d_xpu_v_d_xx_dpp_duu_d_py(
    f, f_d0, f_d, f_d_d,
    t,
    x, x_d0, x_d, x_d_d,
    p, p_d0, p_d, p_d_d,
    u, u_d0, u_d, u_d_d
):
    """Dummy for test cases."""
    f_d0[0] = x_d0[0] + p_d0[0] + u_d0[0]
    f_d0[1] = x_d0[1] + p_d0[1] + t*u_d0[0]
    f_d0[2] = x_d0[2] + p_d0[2] + u_d0[1]
    f_d0[3] = x_d0[3] + p_d0[3] + u_d0[2]
    f_d0[4] = x_d0[4] + p_d0[4] + u_d0[3]

    f_d[0] = x_d[0] + p_d[0] + u_d[0]
    f_d[1] = x_d[1] + p_d[1] + t*u_d[0]
    f_d[2] = x_d[2] + p_d[2] + u_d[1]
    f_d[3] = x_d[3] + p_d[3] + u_d[2]
    f_d[4] = x_d[4] + p_d[4] + u_d[3]

    f[0] = x[0] + p[0] + u[0]
    f[1] = x[1] + p[1] + t*u[0]
    f[2] = x[2] + p[2] + u[1]
    f[3] = x[3] + p[3] + u[2]
    f[4] = x[4] + p[4] + u[3]


def hfcn_py(h, t, x, p, u):
    """Dummy for test cases."""
    h[0] = x[0]
    h[1] = x[1]
    h[2] = x[2]


def hfcn_d_xpu_v_py(h, h_d, t, x, x_d, p, p_d, u, u_d):
    """Dummy for test cases."""
    h_d[0, :] = x[0, :]
    h_d[1, :] = x[1, :]
    h_d[2, :] = x[2, :]

    h[0] = x[0]
    h[1] = x[1]
    h[2] = x[2]


# ------------------------------------------------------------------------------
# PYTHON STR IMPLEMENTATION
ffcn_py_str = """
def ffcn(f, t, x, p, u):
    '''Dummy for test cases.'''
    f[0] = x[0] + p[0] + u[0]
    f[1] = x[1] + p[1] + t*u[0]
    f[2] = x[2] + p[2] + u[1]
    f[3] = x[3] + p[3] + u[2]
    f[4] = x[4] + p[4] + u[3]


def ffcn_d_xpu_v(f, f_d, t, x, x_d, p, p_d, u, u_d):
    '''Dummy for test cases.'''
    f_d[0] = x_d[0] + p_d[0] + u_d[0]
    f_d[1] = x_d[1] + p_d[1] + t*u_d[0]
    f_d[2] = x_d[2] + p_d[2] + u_d[1]
    f_d[3] = x_d[3] + p_d[3] + u_d[2]
    f_d[4] = x_d[4] + p_d[4] + u_d[3]

    f[0] = x[0] + p[0] + u[0]
    f[1] = x[1] + p[1] + t*u[0]
    f[2] = x[2] + p[2] + u[1]
    f[3] = x[3] + p[3] + u[2]
    f[4] = x[4] + p[4] + u[3]


def ffcn_b_xpu(f, f_b, t, x, x_b, p, p_b, u, u_b):
    '''Dummy for test cases.'''
    f[0] = x[0] + p[0] + u[0]
    f[1] = x[1] + p[1] + t*u[0]
    f[2] = x[2] + p[2] + u[1]
    f[3] = x[3] + p[3] + u[2]
    f[4] = x[4] + p[4] + u[3]

    f_b[0] = x_b[0] + p_b[0] + u_b[0]
    f_b[1] = x_b[1] + p_b[1] + t*u_b[0]
    f_b[2] = x_b[2] + p_b[2] + u_b[1]
    f_b[3] = x_b[3] + p_b[3] + u_b[2]
    f_b[4] = x_b[4] + p_b[4] + u_b[3]


def ffcn_d_xpu_v_d_xx_dpp_duu_d(
    f, f_d0, f_d, f_d_d,
    t,
    x, x_d0, x_d, x_d_d,
    p, p_d0, p_d, p_d_d,
    u, u_d0, u_d, u_d_d
):
    '''Dummy for test cases.'''
    f_d0[0] = x_d0[0] + p_d0[0] + u_d0[0]
    f_d0[1] = x_d0[1] + p_d0[1] + t*u_d0[0]
    f_d0[2] = x_d0[2] + p_d0[2] + u_d0[1]
    f_d0[3] = x_d0[3] + p_d0[3] + u_d0[2]
    f_d0[4] = x_d0[4] + p_d0[4] + u_d0[3]

    f_d[0] = x_d[0] + p_d[0] + u_d[0]
    f_d[1] = x_d[1] + p_d[1] + t*u_d[0]
    f_d[2] = x_d[2] + p_d[2] + u_d[1]
    f_d[3] = x_d[3] + p_d[3] + u_d[2]
    f_d[4] = x_d[4] + p_d[4] + u_d[3]

    f[0] = x[0] + p[0] + u[0]
    f[1] = x[1] + p[1] + t*u[0]
    f[2] = x[2] + p[2] + u[1]
    f[3] = x[3] + p[3] + u[2]
    f[4] = x[4] + p[4] + u[3]


def hfcn_py(h, t, x, p, u):
    '''Dummy for test cases.'''
    h[0] = x[0]
    h[1] = x[1]
    h[2] = x[2]


def hfcn_d_xpu_v_py(h, h_d, t, x, x_d, p, p_d, u, u_d):
    '''Dummy for test cases.'''
    h_d[0, :] = x[0, :]
    h_d[1, :] = x[1, :]
    h_d[2, :] = x[2, :]

    h[0] = x[0]
    h[1] = x[1]
    h[2] = x[2]


"""


# ------------------------------------------------------------------------------
# FORTRAN IMPLEMENTATION
ffcn_f_str = """
C-------------------------------------------------------------------------------

      subroutine ffcn(f, t, x, p, u)
C       Dummy for test cases.
        implicit none
        real*8 f(5), t, x(5), p(5), u(4)
C       ------------------------------------------------------------------------
        ! Independent values
        f(1) = x(1) + p(1) + u(1)
        f(2) = x(2) + p(2) + t*u(1)
        f(3) = x(3) + p(3) + u(2)
        f(4) = x(4) + p(4) + u(3)
        f(5) = x(5) + p(5) + u(4)
C       ------------------------------------------------------------------------
      end

C-------------------------------------------------------------------------------

      subroutine ffcn_d_xpu_v(f, f_d, t, x, x_d, p, p_d, u, u_d
     *, nbdirs)
C       ------------------------------------------------------------------------
C       Dummy for test cases.
        implicit none
        real*8 f(5), t, x(5), p(5), u(4)
        integer nbdirs
        real*8 f_d(nbdirs, 5), x_d(nbdirs, 5), p_d(nbdirs, 5)
        real*8 u_d(nbdirs, 4)
        integer nd0
C       ------------------------------------------------------------------------
        ! Derivative evaluation
        DO nd0=1,nbdirs
          f_d(nd0, 1) = x_d(nd0, 1) + p_d(nd0, 1) + u_d(nd0, 1)
          f_d(nd0, 2) = x_d(nd0, 2) + p_d(nd0, 2) + u_d(nd0, 1)*t
          f_d(nd0, 3) = x_d(nd0, 3) + p_d(nd0, 3) + u_d(nd0, 2)
          f_d(nd0, 4) = x_d(nd0, 4) + p_d(nd0, 4) + u_d(nd0, 3)
          f_d(nd0, 5) = x_d(nd0, 5) + p_d(nd0, 5) + u_d(nd0, 4)
        ENDDO

        ! Independent values
        f(1) = x(1) + p(1) + u(1)
        f(2) = x(2) + p(2) + t*u(1)
        f(3) = x(3) + p(3) + u(2)
        f(4) = x(4) + p(4) + u(3)
        f(5) = x(5) + p(5) + u(4)
C       ------------------------------------------------------------------------
      end

C-------------------------------------------------------------------------------

      subroutine hfcn(h, t, x, p, u)
C       ------------------------------------------------------------------------
        implicit none
        real*8 h(3), t, x(5), p(5), u(4)
C       ------------------------------------------------------------------------

        h(1) = x(1)
        h(2) = x(2)
        h(3) = x(3)

C       ------------------------------------------------------------------------
      end

            subroutine hfcn_d_xpu_v(h, h_d, t, x, x_d, p, p_d, u, u_d
     *, nbdirs)
C       ------------------------------------------------------------------------
C       Dummy for test cases.
        implicit none
        real*8 h(3), t, x(5), p(5), u(4)
        integer nbdirs
        real*8 h_d(nbdirs, 3), x_d(nbdirs, 5), p_d(nbdirs, 5)
        real*8 u_d(nbdirs, 4)
        integer nd0
C       ------------------------------------------------------------------------
        ! Derivative evaluation
        DO nd0=1,nbdirs
          h_d(nd0, 1) = x_d(nd0, 1)
          h_d(nd0, 2) = x_d(nd0, 2)
          h_d(nd0, 3) = x_d(nd0, 3)
        ENDDO

        ! Independent values
        h(1) = x(1)
        h(2) = x(2)
        h(3) = x(3)
C       ------------------------------------------------------------------------
      end

C-------------------------------------------------------------------------------
"""


"""
def ffcn_d_xpu_v(f, f_d, t, x, x_d, p, p_d, u, u_d):
    '''Dummy for test cases.'''
    f_d[0] = x_d[0] + p_d[0] + u_d[0]
    f_d[1] = x_d[1] + p_d[1] + t*u_d[0]
    f_d[2] = x_d[2] + p_d[2] + u_d[1]
    f_d[3] = x_d[3] + p_d[3] + u_d[2]
    f_d[4] = x_d[4] + p_d[4] + u_d[3]

    f[0] = x[0] + p[0] + u[0]
    f[1] = x[1] + p[1] + t*u[0]
    f[2] = x[2] + p[2] + u[1]
    f[3] = x[3] + p[3] + u[2]
    f[4] = x[4] + p[4] + u[3]


def ffcn_b_xpu(f, f_b, t, x, x_b, p, p_b, u, u_b):
    '''Dummy for test cases.'''
    f[0] = x[0] + p[0] + u[0]
    f[1] = x[1] + p[1] + t*u[0]
    f[2] = x[2] + p[2] + u[1]
    f[3] = x[3] + p[3] + u[2]
    f[4] = x[4] + p[4] + u[3]

    f_b[0] = x_b[0] + p_b[0] + u_b[0]
    f_b[1] = x_b[1] + p_b[1] + t*u_b[0]
    f_b[2] = x_b[2] + p_b[2] + u_b[1]
    f_b[3] = x_b[3] + p_b[3] + u_b[2]
    f_b[4] = x_b[4] + p_b[4] + u_b[3]

C-------------------------------------------------------------------------------

      subroutine hfcn(f, t, x, p, u)
C       Dummy for test cases.
        implicit none
        real*8 h(3), t, x(5), p(5), u(4)
C       ------------------------------------------------------------------------
        ! Independent values
        h(1) = x(1)
        h(2) = x(2)
        h(3) = x(3)
C       ------------------------------------------------------------------------
      end


"""


# ------------------------------------------------------------------------------
# LOCAL FIXTURES
@pytest.fixture
def temp_ffcn_f_file(tmpdir):
    f = tmpdir.mkdir('temp_mf').join("ffcn.f")
    f.write(ffcn_f_str)
    return f


@pytest.fixture
def temp_ffcn_py_file(tmpdir):
    f = tmpdir.mkdir('temp_mf').join("ffcn.py")
    f.write(ffcn_py_str)
    return f


@pytest.fixture
def temp_shared_library_from_ffcn_f(temp_ffcn_f_file):
    """Compile FORTRAN file and create shared library."""
    # unpack file path
    fpath = str(temp_ffcn_f_file)
    fpath = os.path.abspath(fpath)
    f_dir = os.path.dirname(fpath)
    f_name = os.path.basename(fpath)

    # print ""
    # print "fpath:   ", fpath
    # print "f_dir:   ", f_dir
    # print "f_name:  ", f_name

    # retrieve current working directory
    temp_dir = os.getcwd()

    # set working directory to the one from file path
    os.chdir(f_dir)

    # compile using gfortran
    command = "gfortran -fPIC -shared -O2 -o {fname}.so {fname}.f"
    command = command.format(fname=os.path.splitext(f_name)[0])
    command = command.split(" ")
    try:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            # universal_newlines=True,
            # shell=True,
        )
        # catch output during process
        while proc.poll() is None:
            output = proc.stdout.readline()
            # print output,
            # catch rest after while loop breaks
            output = proc.communicate()[0]
            # print output,
    except Exception:
        raise

    # set working directory back to old one
    os.chdir(temp_dir)

    # return path to so
    path_to_so = os.path.join(f_dir, os.path.splitext(f_name)[0] + ".so")
    return path_to_so


# ------------------------------------------------------------------------------
# VERIFY FIXTURES
def test_temp_ffcn_f_file(temp_ffcn_f_file):
    """Check content of temporary definition file against source."""
    actual = temp_ffcn_f_file.read()
    desired = ffcn_f_str
    # check content
    assert actual == desired


def test_temp_ffcn_py_file(temp_ffcn_py_file):
    """Check content of temporary definition file against source."""
    actual = temp_ffcn_py_file.read()
    desired = ffcn_py_str
    # check content
    assert actual == desired


def test_temp_shared_library_from_ffcn_f(temp_shared_library_from_ffcn_f):
    """Check if library was build from fortran file."""
    path_to_so = str(temp_shared_library_from_ffcn_f)
    # check if shared library exists
    assert os.path.isfile(path_to_so)


# ------------------------------------------------------------------------------
# ACTUAL TESTS
def test_msobox_function_interface_on_ffcn_so_calling_ffcn(
    temp_shared_library_from_ffcn_f
):
    """."""
    # path to shared library
    so_path = str(temp_shared_library_from_ffcn_f)
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
        "name": "ffcn_",
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
    temp_shared_library_from_ffcn_f
):
    """."""
    # path to shared library
    so_path = str(temp_shared_library_from_ffcn_f)
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
        "name": "hfcn_",
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
    temp_shared_library_from_ffcn_f
):
    # path to shared library
    so_path = str(temp_shared_library_from_ffcn_f)
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
        "name": "ffcn_d_xpu_v_",
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


def test_msobox_function_interface_on_ffcn_py_calling_ffcn(temp_ffcn_py_file):
    # path to shared library
    f_path = str(temp_ffcn_py_file)
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
    temp_ffcn_py_file
):
    # path to shared library
    f_path = str(temp_ffcn_py_file)
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
