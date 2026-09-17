# toolslyman

[![License](https://img.shields.io/github/license/sambit-giri/toolslyman.svg)](https://github.com/sambit-giri/toolslyman/blob/main/LICENSE.md)
[![GitHub Repository](https://img.shields.io/github/repo-size/sambit-giri/toolslyman)](https://github.com/sambit-giri/toolslyman)
![CI Status](https://github.com/sambit-giri/toolslyman/actions/workflows/ci.yml/badge.svg)
[![PyPI version](https://badge.fury.io/py/toolslyman.svg)](https://badge.fury.io/py/toolslyman)
[![Docs](https://github.com/sambit-giri/toolslyman/actions/workflows/docs.yml/badge.svg)](https://sambit-giri.github.io/toolslyman/)

A python package to study lyman-alpha photons in our Universe. More documentation can be found at its [documentation](https://sambit-giri.github.io/toolslyman/) page.

**Note:** Some modules in the package are still under active development. Please contact the authors if you encounter any issues.

## Package details

`toolslyman` provides modules to model and analyze the transmission of Lyman-alpha
photons through the intergalactic medium (IGM) during and after the epoch of
reionization. This includes the Lyman-alpha forest transmission (via the
fluctuating Gunn-Peterson approximation and full radiative transfer), the
Lyman-alpha damping wing from neutral hydrogen (both numerically along
cosmological skewers and via closed-form analytic formulas), photometric IGM
tomography for mapping residual neutral islands from mock or real
observations, toy models of reionization topology, and supporting cosmological
and observational utilities.

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

## EXAMPLES

Tutorial notebooks are in the [examples](https://github.com/sambit-giri/toolslyman/tree/main/examples) folder (also linked from the [tutorials](https://sambit-giri.github.io/toolslyman/tutorials.html) page of the documentation):

* [`Lyman_alpha_damping_wing.ipynb`](https://github.com/sambit-giri/toolslyman/blob/main/examples/Lyman_alpha_damping_wing.ipynb) — modelling the Lyman-alpha damping wing along cosmological skewers, both numerically and with the Miralda-Escude (1998) analytic formula.
* [`Photometric_IGM_tomography.ipynb`](https://github.com/sambit-giri/toolslyman/blob/main/examples/Photometric_IGM_tomography.ipynb) — reconstructing 2D maps of the IGM Lyman-alpha transmission from mock photometric observations of background galaxies, following [Giri, Kakiichi, Bianco & Meerburg (2025)](https://arxiv.org/abs/2505.06350).

## CONTRIBUTING

If you find any bugs or unexpected behavior in the code, please feel free to open a [Github issue](https://github.com/sambit-giri/toolslyman/issues). The issue page is also good if you seek help or have suggestions for us. For more details, please see [here](https://sambit-giri.github.io/toolslyman/contributing.html).
    
