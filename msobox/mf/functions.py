"""Base class of MSOBox model function implementation."""

import os
import sys
import cffi
import json
import numpy
import types
import importlib

from copy import deepcopy

from cffi import FFI
from inspect import (getargspec, getcallargs)
from functools import (wraps,)
from collections import (OrderedDict,)


# ------------------------------------------------------------------------------
class Functor(object):

    """Functor implementation for MSO box function interface."""

    def __init__(self, module, function, dimensions, declaration, ffi=None):
        """Functor implementation for MSO box function interface."""
        err_str = "function shall be a callable object"
        assert callable(function), err_str

        self._module = module
        self._function = function
        self._dimensions = dimensions
        self._declaration = declaration
        self._ffi = ffi

        self._cast_args = False
        if ffi:
            self._cast_args = True

    def __call__(self, *args, **kwargs):
        """TODO get doc_string from function or auto-generated one."""
        if self._cast_args:
            ffi_args = []
            ffi_shps = []
            ffi_Ps = []
            for arg in args:
                ffi_args.append(self._ffi.cast("double*", arg.ctypes.data))
                ffi_shps.append(arg.shape)
                if len(arg.shape) == 2 and arg.shape[1] not in ffi_Ps:
                    ffi_Ps.append(arg.shape[1])

            # Add directions to arguments
            # FIXME: higher order derivatives include multiple directions
            ffi_Ps = numpy.asarray(ffi_Ps)
            for i, P in enumerate(ffi_Ps):
                ffi_args.append(
                    self._ffi.cast("int*", ffi_Ps[i:i+1].ctypes.data)
                )

            # call FFI function with proper args
            return self._function(*ffi_args, **kwargs)
        else:
            return self._function(*args, **kwargs)


# ------------------------------------------------------------------------------
class Function(object):

    """Base class for function wrapping."""

    def __new__(cls, module, dims_d, func_d, ffi=None, verbose=False):
        """Create hierarchy of Functors and return them."""
        _dims = {}
        _dims.update(dims_d)

        _func = {
            "name": "",
            "type": "",
            "args": [],
            "deriv": []
        }
        _func.update(func_d)

        # create functor from module and data
        functor = cls.create_functor_from_module_and_data(
            module, _dims, _func, ffi=ffi
        )
        return functor

    @classmethod
    def create_functor_from_module_and_data(
        cls, module, _dims, _func, ffi=None
    ):
        """Create a functor from module, dimensions and declaration."""
        # retrieve function by name
        f_name = _func["name"]
        if ffi:
            f_name += "_"
        func = cls.import_function_from_module_by_name(module, f_name)

        # check arguments if specified
        if _func["args"] and isinstance(func, types.FunctionType):
            cls.check_arguments(_func["args"], func)

        # create and return functor
        return Functor(module, func, _dims, _func, ffi=ffi)

    @classmethod
    def check_arguments(cls, f_args, func, verbose=False):
        """Check order of arguments of python function."""
        err_str = "function arguments do not coincide"
        args, _, _, _ = getargspec(func)
        if verbose:
            for actual, desired in zip(args, f_args):
                print actual, " == ", desired
            print ""
        assert args == f_args

    @classmethod
    def import_function_from_module_by_name(cls, module, name):
        """Return function from module by name."""
        try:
            return getattr(module, name)
        except AttributeError:
            err_str = "Could not load symbol {name} from shared library"
            err_str = err_str.format(name=name)
            raise AttributeError(err_str)

    def __init__(self, *args, **kwargs):
        """Implementation of constructor raising Error for safety."""
        err_str = "This shall never be called!"
        raise NotImplementedError(err_str)


# ------------------------------------------------------------------------------
