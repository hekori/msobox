"""Set of module-wide fixtures."""

import os
import pytest
import subprocess


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
mf_py_str = """
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
mf_f_str = """
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


# FIXME add this to fortran version
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

"""


# ------------------------------------------------------------------------------
# GLOBAL FIXTURES
@pytest.fixture
def temp_ffcn_f_file(tmpdir):
    f = tmpdir.mkdir('temp_mf').join("ffcn.f")
    f.write(mf_f_str)
    return f


@pytest.fixture
def temp_ffcn_py_file(tmpdir):
    f = tmpdir.mkdir('temp_mf').join("ffcn.py")
    f.write(mf_py_str)
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
    desired = mf_f_str
    # check content
    assert actual == desired


def test_temp_ffcn_py_file(temp_ffcn_py_file):
    """Check content of temporary definition file against source."""
    actual = temp_ffcn_py_file.read()
    desired = mf_py_str
    # check content
    assert actual == desired


def test_temp_shared_library_from_ffcn_f(temp_shared_library_from_ffcn_f):
    """Check if library was build from fortran file."""
    path_to_so = str(temp_shared_library_from_ffcn_f)
    # check if shared library exists
    assert os.path.isfile(path_to_so)



# ------------------------------------------------------------------------------
