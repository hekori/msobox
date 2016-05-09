"""Base class of MSOBox model function implementation."""

import os
import re
import sys
import cffi
import json
import numpy
import logging
import importlib

from copy import deepcopy

from cffi import FFI
from inspect import (getargspec, getcallargs)
from functools import (wraps,)
from collections import (OrderedDict,)

from functions import (Function)


# ------------------------------------------------------------------------------
def import_module_from_file(f_path, verbose=True):
    """
    Import module from file and return it.

    Parameters
    ----------
    f_path: strin
        pass to file containing module code

    verbose: bool
        flag for additional output

    Raises
    ------
    ImportError
        if module cannot be imported
    """
    # get absolute path
    f_path = os.path.abspath(f_path)
    assert os.path.isfile(f_path)

    f_dir = os.path.dirname(f_path)
    f_name = os.path.basename(f_path)
    f_id = os.path.splitext(f_name)[0]

    try:
        # add f_dir to system path for later import
        sys.path.insert(0, f_dir)
        # import module by name
        module = importlib.import_module(f_id)
        return module
    except ImportError:
        err_str = "ERROR: Could not import module '{}' from '{}'.\n"
        err_str = err_str.format(f_name, f_dir)
        raise ImportError(err_str)


# ------------------------------------------------------------------------------
def import_shared_library(path_to_so, verbose=True):
    """Load shared library using foreign function interface."""
    class FFILibrary(object):
        def __init__(self, path_to_so, verbose):
            # initialize foreign function interface for library
            self._ffi = FFI()

        def _update_header(self, header):
            _ffi = object.__getattribute__(self, "_ffi")
            _ffi.cdef(header)

        def _get_lib(self):
            _ffi = object.__getattribute__(self, "_ffi")
            return _ffi.dlopen(path_to_so)

        def __getattribute__(self, name):
            """Open library lazy and return member."""
            if name == "_update_header":
                return object.__getattribute__(self, "_update_header")

            elif name == "_get_lib":
                return object.__getattribute__(self, "_get_lib")

            else:
                # Load symbols from the current process (Python).
                return getattr(self._get_lib(), name)

    return FFILibrary(path_to_so, verbose)


# ------------------------------------------------------------------------------
def generate_derivative_declarations(
    declarations, derivatives, dimensions, function_d, verbose=True
):
    """Recursively get all derivatives from function declarations."""
    # look-up table for mode assignment
    _mode_d = {
        "forward_single": ("d", ""),
        "forward_vector": ("d", "v"),
        "reverse_single": ("b", ""),
        "reverse_vector": ("b", "v"),
    }

    # assign first level of derivative
    for deriv in derivatives:
        # copy original dictionaries
        _dims = deepcopy(dimensions)
        _func = deepcopy(function_d)

        _deriv_dims = _dims
        mode, vector = _mode_d[deriv["mode"]]
        if vector:
            vector = [vector]
        else:
            vector = []

        # generate derivative name
        invar = deriv["in"]
        outvar = deriv["out"]
        invar_s = "".join(deriv["in"])
        outvar_s = "".join(deriv["out"])
        name = "_".join([_func["name"], mode, invar_s] + vector)

        # update args of derivative
        args = deepcopy(_func["args"])
        for v in outvar + invar:
            new_v = "_".join([v, mode])
            v_cnt = 0
            while new_v in args:
                new_v = "_".join([v, mode]) + str(v_cnt)
                v_cnt += 1
                if v_cnt > 10:
                    raise Exception()
            args.insert(args.index(v) + 1, new_v)

        _deriv_func = {
            # NOTE function and file name coincide if no 'func_d' is given!
            "name": name,
            "type": _func["type"],
            "args": args,
            "deriv": deriv.get("deriv", [])
        }

        # create functor from module and data
        # _d[function_d["name"]] = (_dims, _func)
        declarations[name] = (_deriv_dims, _deriv_func)

        # add 'dot' alias for total derivative
        # FIXME: if invar + outvar == args then add '_dot' or '_bar' as alias
        if outvar + ["t"] + invar == _func["args"]:
            if "d" in mode:
                declarations[_func["name"] + "_dot"] = (
                    _deriv_dims, _deriv_func
                )
            elif "b" in mode:
                declarations[_func["name"] + "_bar"] = (
                    _deriv_dims, _deriv_func
                )

        # TODO: add recursively higher order derivatives
        # NOTE: there will be a problem with nbdirs of fortran calls
        continue
        # assign next level of derivatives when specified
        if _deriv_func["deriv"]:
            generate_derivative_declarations(
                declarations, _deriv_func["deriv"], _dims, _func, verbose
            )


def args_str_from_args(fargs):
    """Generate arguments for C header."""
    pt = re.compile(r"_d(?P<cnt>\d?)")

    s = []
    dirs = []

    for arg in fargs:
        s.append("double* {}".format(arg))
        m = pt.findall(fargs[1])
        for m in pt.finditer(arg):
            m = m.group("cnt")
            if m not in dirs:
                dirs.append(m)
    for cnt in dirs:
        s.append("int* nbdirs{}".format(cnt))

    s = ", ".join([str(x) for x in s])
    return s


def header_from_function_name_and_args(fname, fargs):
    """
    Create C header file for the FFI interface.

    fname = 'ffcn'
    fargs = ['f', 't', 'x', 'p', 'u']
    Returns
        void ffcn_(double *f, double *t, double *x, double *p, double *u);

    fname = 'ffcn_d_xpu_v'
    fargs = ['f', 'f_d', 't', 'x', 'x_d', 'p', 'p_d', 'u', 'u_d']
    Returns
        void ffcn_(double *f, double *t, double *x, double *p, double *u);
        void ffcn_d_xpu_v(
            double* f, double* f_d,
            double* t,
            double* x, double* x_d,
            double* p, double* p_d,
            double* u, double* u_d,
            int* nbdirs
        )
    """
    header = "void {fname}_({fargs_str});".format(
        fname=fname, fargs_str=args_str_from_args(fargs)
    )
    return header


def generate_header_from_declarations(function_declarations, verbose=True):
    """Create C header file for the FFI interface."""
    header = ""
    for (f_name, (f_dims, f_dict)) in function_declarations.iteritems():
        s = header_from_function_name_and_args(f_name, f_dict["args"])
        header += s + "\n"

    return header


# ------------------------------------------------------------------------------
class Model(object):

    """
    Model base class implementing model functions back end of MSO tool box.

    Model class loads model definitions and model function declarations, and
    wraps them by automatic assignment of attributes needed for MSO tools.
    """

    def __init__(self, model_functions, model_definitions, verbose=True):
        """."""
        # enable verbose output
        self.verbose = verbose

        # load model definitions
        self.definitions = self.load_model_definitions(model_definitions)

        # assign functions and declarations
        self.dimensions = self.definitions["dims"]
        self.declarations = self.definitions["functions"]

        # convert declarations to single function declarations
        # NOTE: returns an OrderedDict
        _function_declarations = \
            self.convert_declaration_to_function_declarations(
                self.dimensions, self.declarations, verbose=self.verbose
            )

        # load model functions
        self.module, self.ffi = self.load_model_functions(
            model_functions, _function_declarations, verbose
        )

        # load functions from module
        for (f_name, (f_dims, f_dict)) in _function_declarations.iteritems():
            # create function
            func = Function(
                self.module, f_dims, f_dict, ffi=self.ffi, verbose=self.verbose
            )
            setattr(self, f_name, func)

    @staticmethod
    def load_model_definitions(model_definitions):
        """Load model definitions."""
        _d = {}
        # is model_definitions a dictionary?
        if isinstance(model_definitions, dict):
            _d.update(model_definitions)

        # is model_definitions a string?
        elif isinstance(model_definitions, str):
            # is model_definitions a path to a file?
            if os.path.isfile(model_definitions):
                # load json string from file
                with open(model_definitions) as f:
                    _d.update(json.load(f))
            else:
                # load json string
                _d = json.loads(model_definitions)
        else:
            err_str = "provided 'model_definitions' is neither an existing"
            err_str += "file nor an json string nor an dictionary.\n"
            err_str += "bailing out ...\n"
            raise TypeError(err_str)

        # assign definitions
        return _d

    @staticmethod
    def convert_declaration_to_function_declarations(
        dimensions, declarations, verbose=True
    ):
        """."""
        # function declaration container
        _d = OrderedDict()

        # iterate over all specified functions
        for func in declarations:
            _dims = deepcopy(dimensions)
            _func = {
                "name": func["name"],
                "type": func["type"],
                "args": func["args"],
                "deriv": func["deriv"],
            }
            # assign function declaration
            _d[func["name"]] = (_dims, _func)

            # assign derivative declarations
            generate_derivative_declarations(
                _d, func["deriv"], _dims, _func, verbose
            )

        return _d

    @staticmethod
    def load_model_functions(
        model_functions, function_declarations, verbose=True
    ):
        """
        Load model functions.

        Parameters
        ----------
        model_functions : object or module
            Where model_functions is either:
                1) a path to a directory containing a file 'model_functions.py'
                2) a class or an object providing a specified API or

        function_declarations : dict
            function declarations for header generation

        Returns
        -------
        module: class-like
            returns a class-like module object.

        ffi: None or cffi.FFI()
            returns None for Python or FFI interface for shared libraries

        """
        # is model_functions a file?
        if os.path.isfile(model_functions):
            # unpack path
            f_path = os.path.abspath(model_functions)
            f_dir = os.path.dirname(f_path)
            f_name = os.path.basename(f_path)
            (f_id, f_ext) = os.path.splitext(f_name)
            # print "f_path:   ", f_path
            # print "f_dir:   ", f_dir
            # print "f_name:  ", f_name
            # print "f_id:    ", f_id
            # print "f_ext:   ", f_ext

            # is it a shared library?
            if f_ext == ".so":
                # generate header file from declarations
                header = generate_header_from_declarations(
                    function_declarations
                )

                # instantiate foreign function interface (FFI)
                ffi = FFI()
                ffi.cdef(header)
                module = ffi.dlopen(f_path)

                ret = (module, ffi)
                return ret

            elif f_ext == ".py":
                ret = import_module_from_file(f_path, verbose=verbose), None
                return ret
            else:
                err_str = "'model_functions' extension is not 'so' or 'py'.\n"
                err_str += "bailing out ..."
                raise TypeError(err_str)

        elif isinstance(model_functions, object):
            return (model_functions, None)
        else:
            err_str = "'model_functions' is neither an importable file nor a "
            err_str += "class inherited from 'object'.\n"
            err_str += "bailing out ..."
            raise TypeError(err_str)


# ------------------------------------------------------------------------------
class OldModel(object):

    """
    Model base class implementing model functions back end of MSO tool box.

    Model class loads model definitions and model function declarations, and
    wraps them by automatic assignment of attributes needed for MSO tools.
    """

    def __init__(
        self, model_functions, model_definitions,
        check_arguments=False, verbose=True
    ):
        """
        Load the model functions and definitions into the model class.

        From the provided path the model function together with their prober
        dimensions are loaded and automatically assigned to the model function
        class as members.

        Model functions and their respective derivatives are replaced by dummy
        functions raising an `NotImplementedError` when called.

        Parameters
        ----------
        model_functions : object or module
            Where model_functions is either:
                1) a path to a directory containing a file 'model_functions.py'
                2) a class or an object providing a specified API or

        model_definitions : path to or json compatible str or dict
            A configuration file specifying function name plus the call
            arguments and their order as well the dimensions.

        check_arguments : bool
            Enables argument dimension checking for each function call.

        verbose : bool
            Enables additional output to the used.
        """
        # enable dimension checking of model functions
        self.check_arguments = check_arguments

        # enable verbose output
        self.verbose = verbose

        # assign dummy dimensions
        self.NT = -1  # dimension of time (== 1)
        self.NX = -1  # no. of total states
        self.NY = -1  # no. of differential states
        self.NZ = -1  # no. of algebraic states
        self.NP = -1  # no. of parameters
        self.NU = -1  # no. of controls

        # load model definitions
        self.definitions = self.load_model_definitions(model_definitions)
        self.assign_dimensions()  # assign dimensions
        self.assign_functions()  # assign functions

        # load model functions
        self.load_model_functions(model_functions)

    @staticmethod
    def load_model_definitions(model_definitions):
        """Load model definitions."""
        _d = {}
        if isinstance(model_definitions, dict):
            try:
                _d = model_definitions
            except:
                raise

        elif isinstance(model_definitions, str):
            if os.path.isfile(model_definitions):
                try:
                    with open(model_definitions) as f:
                        _d = json.load(f)
                except:
                    raise

            else:
                try:
                    _d = json.loads(model_definitions)
                except:
                    raise
        else:
            err_str = "provided 'model_definitions' is neither an existing"
            err_str += "file nor an json string nor an dictionary.\n"
            err_str += "bailing out ...\n"
            raise TypeError(err_str)

        # assign definitions
        return _d

    def assign_dimensions(self):
        """Assign dimensions from loaded definitions."""
        _d_assignment = {
            "t": "NT",
            "x": "NX",
            "y": "NY",
            "z": "NZ",
            "u": "NU",
            "p": "NP",
        }
        self.dimensions = self.definitions["dims"]
        for key, val in self.dimensions.iteritems():
            if not _d_assignment.has_key(key):
                err_str = "'{} is not accepted as function argument!".format(
                    key=key
                )
                err_str += "bailing out ..."
                raise AttributeError(err_str)
            else:  # assign dimensions
                setattr(self, _d_assignment[key], val)

    def assign_functions(self):
        """Assign function declarations from loaded definitions."""
        # get function declarations from json string
        self.functions = self.definitions.get("functions", [])

        # generate function declaration in header file
        header = cls.header_from_function_name_and_args(
            _func["name"], _func["args"]
        )

        _functions = OrderedDict()
        for func in self.functions:
            _name = func["name"]
            _type = func["type"]
            _args = func["args"]
            _deriv = self.get_derivatives(func.get("deriv", []))
            _functions[_name] = {
                "name": _name,
                "type": _type,
                "args": _args,
                "deriv": _deriv,
            }
        self._functions = _functions


    def load_model_functions(self, model_functions):
        """
        Load model functions.

        Parameters
        ----------
        model_functions : object or module
            Where model_functions is either:
                1) a path to a directory containing a file 'model_functions.py'
                2) a class or an object providing a specified API or
        """
        # import from path
        if os.path.isfile(model_functions):
            self.path = os.path.abspath(model_functions)
            self.dir = os.path.dirname(self.path)

            sys.path.insert(0, self.dir)
            try:
                # TODO import module by name or path to be more flexible
                import model_functions
                self.mf = model_functions
            except ImportError:
                if self.verbose:
                    err_str = "Could not import 'model_functions' from module {}"
                    err_str = err_str.format(model_functions)
                    sys.stderr.write(err_str + '\n')
                pass
                # raise ImportError(err_str)
        elif isinstance(model_functions, object):
            pass
        else:
            err_str = "'model_functions' is neither an importable file nor a "
            err_str += "class inherited from 'object'.\n"
            err_str += "bailing out ..."
            raise TypeError(err_str)

        # or load from object
        if not hasattr(self, "mf"):
            self.mf = model_functions

        # avoid rechecking by iterating over check_members
        for method, specs in self._functions.iteritems():
            if hasattr(self.mf, method):
                setattr(self, method, getattr(self.mf, method))
                # d = getcallargs(f, *args, **kwargs)
                # TODO check callargs
            else:
                # implemented abstract methods are taken
                setattr(self, method, self._dummy)
                if self.verbose:
                    err_str = (
                        'Could not register method {method},'
                        .format(method=method)
                        + ' dummy is taken instead'
                    )
                    sys.stderr.write(err_str + '\n')

    def print_model_definitions(self):
        """Pretty printing model definitions."""
        sys.stdout.write("Model Dimensions\n")
        sys.stdout.write("----------------\n")
        for key, val in self.dimensions.iteritems():
            sys.stdout.write("{key}: {val}\n".format(key=key, val=val))

    def print_model_functions(self):
        """Assign function declarations from loaded definitions."""
        # TODO replace print statements with stdout.write
        # TODO get derivatives recursively
        self.functions = self.definitions.get("functions", [])
        for func in self.functions:
            print "type: ", func["type"]
            print "args: ", func["args"]
            print "derivatives: "
            for deriv in func.get("deriv", []):
                for key, val in deriv.iteritems():
                    print "  ", key, val
                print ""

    def _dummy(*args, **kwargs):
        """Python interface dummy function."""
        err_str = ""
        raise NotImplementedError()

