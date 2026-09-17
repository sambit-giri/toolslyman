# toolslyman

[![License](https://img.shields.io/github/license/sambit-giri/toolslyman.svg)](https://github.com/sambit-giri/toolslyman/blob/main/LICENSE.md)
[![GitHub Repository](https://img.shields.io/github/repo-size/sambit-giri/toolslyman)](https://github.com/sambit-giri/toolslyman)
![CI Status](https://github.com/sambit-giri/toolslyman/actions/workflows/ci.yml/badge.svg)
[![PyPI version](https://badge.fury.io/py/toolslyman.svg)](https://badge.fury.io/py/toolslyman)
[![Docs](https://github.com/sambit-giri/toolslyman/actions/workflows/docs.yml/badge.svg)](https://sambit-giri.github.io/toolslyman/)

A python package to study lyman-alpha photons in our Universe. More documentation can be found at its [documentation](https://sambit-giri.github.io/toolslyman/) page.

**Note:** Some modules in the package are still under active development. Please contact the authors if you encounter any issues.

## Package details

The package provides modules to model transmission of lyman-alpha photons in cosmological volumes.

## INSTALLATION

The package is available on [PyPI](https://pypi.org/project/toolslyman/). To install the latest released version, run::

    pip install toolslyman

To install the latest development version directly from GitHub instead, use::

    pip install git+https://github.com/sambit-giri/toolslyman.git

To install the package from source, one should clone this package running the following::

    git clone https://github.com/sambit-giri/toolslyman.git

To install the package in the standard location, run the following in the root directory::

    python setup.py install

In order to install it in a separate directory::

    python setup.py install --home=directory

The dependencies should be installed automatically during the installation process. The list of required packages can be found in the requirements.txt file present in the root directory.

### Tests

For testing, one can use [pytest](https://docs.pytest.org/en/stable/). To run all the test script, run the either of the following::

    python -m pytest tests
    
## CONTRIBUTING

If you find any bugs or unexpected behavior in the code, please feel free to open a [Github issue](https://github.com/sambit-giri/toolslyman/issues). The issue page is also good if you seek help or have suggestions for us. For more details, please see [here](https://sambit-giri.github.io/toolslyman/contributing.html).

## CREDIT


    This package uses the template provided at https://github.com/sambit-giri/SimplePythonPackageTemplate/ 
    
