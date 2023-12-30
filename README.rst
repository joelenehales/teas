==================================================================================
``teas`` - Tools for the Extraction and Analysis of Spectra from JWST observations
==================================================================================

.. image:: https://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: https://www.astropy.org
    :alt: Powered by Astropy

.. image:: https://img.shields.io/badge/powered%20by-STScI-blue.svg?colorA=707170&colorB=3e8ddd&style=flat
  :target: http://www.stsci.edu
  :alt: Powered by STScI

``teas`` is a package of utilities to analyze spectra from the James Webb Space Telescope. It provides tools to align and process JWST data products, extract 1D spectra from 3D integral-field-unit data cubes from NIRSpec and MIRI MRS, and create images of observed regions with NIRCam observations.

Features:
---------

* Extract 1D spectra from 3D IFU data from NIRSpec and MIRI MRS
* Create images of spectrally imaged regions using NIRCam data
* Align and reproject the WCS of data cubes and images
* Find the field of view of NIRSpec or MIRI MRS data products within NIRCam images
* Fit the continuum using either an anchors-and-splines fitting or  linear regression
* Stitch together spectra
* Shift, scale, and normalize spectrum flux

Installation
------------

**Distribution is currently delayed due to new user registration to PyPI being
disabled. Please check back later.**

It is recommended to install ``teas`` in a new environment to avoid
version conflicts with other packages. To do this, run:

.. code-block:: bash

   conda create -n my-env python=3.9


Then, activate the environment and install the package. The latest released version can be installed with pip:

.. code-block:: bash

  conda activate my-env
  pip install teas --upgrade
