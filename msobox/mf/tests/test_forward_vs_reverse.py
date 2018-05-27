"""Collection of INDegrator unit test fixtures."""

import os
import json
import numpy
import pytest
import tempfile

import scipy.linalg as lg

from copy import (deepcopy,)
from cffi import (FFI,)
from pprint import (pprint,)
from collections import (OrderedDict,)
from numpy.testing import (TestCase, run_module_suite)
from numpy.testing import (assert_equal, assert_allclose)

from msobox.mf.functions import (Functor,)
from msobox.mf.model import (Model,)

from .conftest import (load_fortran_example_model_by_name,)

# ------------------------------------------------------------------------------
# SET ACCURACY OF NUMERICAL TESTS
ATOL = 1e-07
RTOL = 1e-07
TTOL = 7


# ------------------------------------------------------------------------------
def test_forward_vs_reverse_evaluation(get_msobox_examples_path):
    """
    Test forward vs reverse evaluation.

    Evaluate the AD-identity:

      y_bar.T * y_dot = x_bar.T * x_dot + p_bar.T * p_dot + u_bar.T * u_dot

    """
    # retrieve model functions
    so_path, ds = load_fortran_example_model_by_name("bimolkat")
    mf = Model(model_functions=so_path, model_definitions=ds, verbose=True)

    # look for family of functions in model functions, e.g.
    # ffcn = [ffcn, ffcn_dot, ffcn_d_xpu_v, ...]
    functions = {}
    for key, val in list(mf.__dict__.items()):
        if isinstance(val, Functor):
            func = key.split("_")[0]
            if func not in functions:
                functions[func] = []
            functions[func].append(key)

    # walk through function families and perform tests
    for func in list(functions.keys()):
        # perform first-order forward vs reverse test
        nom = func
        dot = func + "_dot"
        bar = func + "_bar"
        if nom in functions[func] and dot in functions[func] \
        and bar in functions[func]:
            # nominal function evaluation
            nom = getattr(mf, nom)
            nom_dims = nom._dimensions
            nom_args = nom._declaration["args"]

            # create dummy variables for forward function call
            n_args = OrderedDict()
            for i, arg in enumerate(nom_args):
                if arg == nom_args[0]:
                    n_args[arg] = numpy.zeros(nom_dims[arg])
                else:
                    n_args[arg] = numpy.random.random(nom_dims[arg])

            # evaluate nominal call
            nom(*list(n_args.values()))

            # prepare first-order forward derivative call
            # compute sum of dimension to provide proper forward directions
            Ps = {}
            P = [0]
            for arg in nom_args:
                if not arg == "t":
                    P.append(P[-1] + nom_dims[arg])
                    Ps[arg] = slice(P[-2], P[-1])
            else:
                pass

            P = P[-1] - nom_dims[nom_args[0]]

            # first-order derivative evaluation
            dot = getattr(mf, dot)
            dot_dims = dot._dimensions
            dot_args = dot._declaration["args"]
            dot_directions = numpy.zeros([P + nom_dims[nom_args[0]], P])
            dot_directions[nom_dims[nom_args[0]]:, :] = numpy.eye(P)

            # create dummy variables for forward function call
            d_args = OrderedDict()
            for i, arg in enumerate(dot_args):
                if "_d" in arg:
                    key = arg.split("_")[0]
                    d_args[arg] = dot_directions[Ps[key], :]
                else:
                    if arg == nom_args[0]:
                        d_args[arg] = numpy.zeros(dot_dims[arg])
                    else:
                        # NOTE we reuse values from nom run, because call to
                        #      random would render results incomparable
                        d_args[arg] = n_args[arg]
                        # d_args[arg] = numpy.random.random(dot_dims[arg])

            # evaluate forward derivative call
            dot(*list(d_args.values()))

            bar = getattr(mf, bar)
            bar_dims = bar._dimensions
            bar_args = bar._declaration["args"]
            Q = 1

            # create dummy variables for reverse function call
            b_args = OrderedDict()
            for i, arg in enumerate(bar_args):
                if "_b" in arg:
                    key = arg.split("_")[0]
                    if key == nom_args[0]:
                        b_args[arg] = numpy.ones([bar_dims[key], Q])
                    else:
                        b_args[arg] = numpy.zeros([bar_dims[key], Q])
                else:
                    if arg == nom_args[0]:
                        b_args[arg] = numpy.zeros(bar_dims[arg])
                    else:
                        # NOTE we reuse values from nom run, because call to
                        #      random would render results incomparable
                        b_args[arg] = n_args[arg]
                        # b_args[arg] = numpy.random.random(bar_dims[arg])

            # NOTE tapenade *_bar code alters adjoint direction. Therefore we
            #      have to copy it and restore it afterwards for testing!
            temp_b_args = deepcopy(b_args)

            # evaluate forward derivative call
            bar(*list(b_args.values()))

            list(b_args.values())[1][...] = list(temp_b_args.values())[1]

            # Test the nominal evaluation:
            assert_allclose(
                list(n_args.values())[0], list(d_args.values())[0], rtol=RTOL, atol=ATOL
            )
            assert_allclose(
                list(n_args.values())[0], list(b_args.values())[0], rtol=RTOL, atol=ATOL
            )

            # Test the AD identity:
            #  y_bar.T * y_dot = x_bar.T * x_dot + ... + u_bar.T * u_dot
            # calculate out vars
            oarg_dot = list(d_args.values())[1]
            oarg_bar = list(b_args.values())[1]
            oarg = oarg_bar.T.dot(oarg_dot)

            # calculate in vars
            iarg = numpy.zeros([Q, P])
            for i, arg in enumerate(list(d_args.keys())[2:]):
                if "_d" in arg:
                    iarg_dot = list(d_args.values())[i+2]
                    iarg_bar = list(b_args.values())[i+2]
                    iarg += iarg_bar.T.dot(iarg_dot)

            assert_allclose(
                oarg, iarg, rtol=RTOL, atol=ATOL
            )
        else:
            continue

# ------------------------------------------------------------------------------
