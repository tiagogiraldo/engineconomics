from gettext import find
from setuptools import find_packages, setup, find_namespace_packages

VERSION = '0.0.1'
DESCRIPTION = 'This package develops a finnancial calculator with python.'
LONG_DESCRIPTION = 'This package computes the basic financial formulas in python as those used in engineering economics.'


# Setting up
setup(
    name= "engineconomics",
    version=VERSION,
    author="Santiago Giraldo",
    author_email="",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages = find_packages(),
    install_requires=['numpy', 'pandas', 'matplotlib', 'plotly', 'scipy'],

    keywords=['python', 'finance calculator', 'engineering economics', 'fixed income'],
    classifiers= [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Finance" ,
        "Programing Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        ]
)

'''
Development Status :: 1 - Planning
Development Status :: 2 - Pre-Alpha
Development Status :: 3 - Alpha
Development Status :: 4 - Beta
Development Status :: 5 - Production/Stable
Development Status :: 6 - Mature
Development Status :: 7 - Inactive
'''