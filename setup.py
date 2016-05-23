"""MSOBox setup file."""

from __future__ import print_function
import io
import os
import sys
import codecs

from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand

import msobox


# ------------------------------------------------------------------------------
class PyTest(TestCommand):

    """Wrapper for untitests with pytest."""

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errcode = pytest.main(self.test_args)
        sys.exit(errcode)


# ------------------------------------------------------------------------------
setup(
    # Name of the module.
    name="msobox",

    # Versions should comply with PEP440. For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=msobox.__version__,

    # brief description of the project
    description=(
        "MSOBox - A Toolbox for Model-based Optimization"
    ),
    # detailed description of the project
    # TODO add link to read-the-docs
    # long_description=long_description,

    # The project"s main homepage.
    # TODO add proper url
    # url="",

    # Choose your license.
    license="BSD license",

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        "Development Status :: 4 - Beta",

        # Indicate who your project is intended for
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",

        # Pick your license as you wish (should match "license" above)
        "License :: OSI Approved :: BSD License",

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 2.7",

        # Natural language of the implementation
        "Natural Language :: English",

        # Tested on Ubuntu 14.04
        "Operating System :: POSIX :: Linux",

        # Indicate audience
        "Topic :: Education",
        "Topic :: Education :: Testing",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities",
    ],

    # What does your project relate to?
    keywords="numerical integration sensitivity generation",

    # The project"s contents.
    packages=find_packages(),

    # Author details
    author="MSOBox Development Team",
    author_email="msobox@hekori.com",

    url = 'https://bitbucket.org/msobox/msobox',

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip"s
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    # install_requires=[
    #     "numpy",  # used internally for all computations
    #     # "scipy",  # using linear algebra routines
    #     "cffi",  # to enable calls to compiled fortran derivatives
    # ],

    # Testing with pytest and specific settings
    tests_require=["pytest"],
    cmdclass={"test": PyTest},
    # include_package_data=True,
    # test_suite="msobox.test.test_sandman",

    extras_require={
        "testing": ["pytest"],
    }
)


# ------------------------------------------------------------------------------
