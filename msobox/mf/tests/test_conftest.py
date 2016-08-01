"""Set of module-wide fixtures."""

import os
import json
import pytest
import subprocess

from conftest import (md_dict, mf_f_str, mf_py_str)
from conftest import (ffcn_py, ffcn_d_xpu_v_py,)
from conftest import (ffcn_d_xpu_v_d_xx_dpp_duu_d_v_py,)
from conftest import (ffcn_b_xpu_py,)
from conftest import (hfcn_py, hfcn_d_xpu_v_py,)
from conftest import (hfcn_d_xpu_v_d_xx_dpp_duu_d_v_py,)
from conftest import (hfcn_b_xpu_py,)


# ------------------------------------------------------------------------------
# VERIFY FIXTURES
def test_temp_mf_f_file(temp_mf_f_file):
    """Check content of temporary definition file against source."""
    actual = temp_mf_f_file.read()
    desired = mf_f_str
    # check content
    assert actual == desired


def test_temp_mf_py_file(temp_mf_py_file):
    """Check content of temporary definition file against source."""
    actual = temp_mf_py_file.read()
    desired = mf_py_str
    # check content
    assert actual == desired


def test_temp_mf_so_from_mf_f_file(temp_mf_so_from_mf_f_file):
    """Check if library was build from fortran file."""
    path_to_so = str(temp_mf_so_from_mf_f_file)
    # check if shared library exists
    assert os.path.isfile(path_to_so)


def test_temp_md_file(temp_md_file):
    """Check content of temporary definition file against source."""
    actual = json.load(temp_md_file)
    desired = md_dict
    # check content
    assert actual == desired


def test_temp_md_json(temp_md_json):
    """Check content of temporary definition string against source."""
    actual = json.loads(temp_md_json)
    desired = md_dict
    # check content
    assert actual == desired


# ------------------------------------------------------------------------------
