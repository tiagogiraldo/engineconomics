r""" \
Engineconomics
=====
Engineconomics is a library that allows to make computations and graphs used in Economic Engineering in Python.
---------
__version__
    Engineconomics version string
__author__
    Author of Engineconomics
__contributors__
    List of all contributors to the project
__homepage__
    Web URL of the Caer documentation    
"""


from ._meta import (
    version,
    release,
    author,
    author_email,
    contributors
)



r"""
Root Package Info
"""
__version__ = version
__release__ = release
__author__ = author
__author_email__ = author_email
__contributors__ = contributors
__license__ = "AGPL3 License"
__copyright__ = r"""
Copyright (c) 203 Santiago Giraldo <tiagogiraldo>
All Rights Reserved.
"""

def license():
    return __license__

def copyright():
    return __copyright__


from .engineconomics import(
    factor,
    time_value,
    time_value_plot,
    time_value_table,
    compound_interest,
)

def get_engineconomics_version():
    return __version__

# Stop polluting the namespace

## Remove root package info
del author
del version
del release
del contributors
del author_email