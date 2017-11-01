[![Build Status](https://travis-ci.org/msmbuilder/mdentropy.svg?branch=master)](https://travis-ci.org/msmbuilder/mdentropy)
[![Code Health](https://landscape.io/github/msmbuilder/mdentropy/master/landscape.svg?style=flat)](https://landscape.io/github/msmbuilder/mdentropy/master)
[![PyPI version](https://badge.fury.io/py/mdentropy.svg)](http://badge.fury.io/py/mdentropy)
[![License](https://img.shields.io/badge/license-MIT-red.svg?style=flat)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg?style=flat)](http://msmbuilder.org/mdentropy/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1000997.svg)](https://doi.org/10.5281/zenodo.1000997)


MDEntropy
=========

MDEntropy is a python library that allows users to perform information-theoretic
analyses on molecular dynamics (MD) trajectories. It includes methods to
calculate:

+ Bias-Corrected Entropy
+ Conditional Entropy
+ Mutual Information
+ Normalized Mutual Information
+ Conditional Mutual Information
+ Normalized Conditional Mutual Information


## Documentation

Full documentation can be found at [http://msmbuilder.org/mdentropy/](http://msmbuilder.org/mdentropy/).
For information about installation, please refer to our [Installation](http://msmbuilder.org/mdentropy/0.3.0/installation.html) page.

We also have [example notebooks](http://msmbuilder.org/mdentropy/0.3.0/examples/index.html) with common use cases for MDEntropy.
Please feel free to add your own as a pull-request!

## Requirements

+ `python`>=3.4
+ `numpy`>=1.10.4
+ `scipy`>=0.17.0
+ `scikit-learn`>=0.17.0
+ `msmbuilder`>=3.5.0
+ `nose` (optional, for testing)

## Citing

Please cite:

```bibtex
@misc{mdentropy,
  author       = {Carlos X. Hern{\'{a}}ndez and Vijay S. Pande},
  title        = {msmbuilder/mdentropy: v0.3},
  month        = oct,
  year         = 2017,
  doi          = {10.5281/zenodo.1000997},
  url          = {https://doi.org/10.5281/zenodo.1000997}
}
```
