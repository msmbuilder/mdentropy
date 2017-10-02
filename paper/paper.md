---
title: 'MDEntropy: Information-Theoretic Analyses for Molecular Dynamics'
tags:
  - Python
  - information theory
  - molecular dynamics
  - time-series
authors:
  - name: Carlos X. Hernández
    orcid: 0000-0002-8146-5904
    affiliation: 1
  - name: Vijay S. Pande
    affiliation: 1
affiliations:
  - name: Stanford University
    index: 1
date: 1 October 2017
bibliography: paper.bib
repository: https://github.com/msmbuilder/mdentropy
archive_doi: https://doi.org/10.5281/zenodo.1000997
---


# Summary

*MDEntropy* is a Python package for information-theoretic (IT) analysis of data
generated from molecular dynamics simulations. While correlation studies
have long been of interest to the molecular dynamics (MD) community
[@mccammon, @mcclendon], IT tools to analyze MD trajectories have been much
less developed. *MDEntropy* seeks to fill this niche by providing an
easy-to-use Python API that works seamlessly with other Python packages, such
as ``mdtraj``, ``msmbuilder``, and ``numpy`` [@mdtraj, @numpy, @msmbuilder].

Functionality in *MDEntropy* is centered around ``mdtraj`` trajectories and the
statistical tools available in ``msmbuilder``. Leveraging these tools allows
for statistically robust analyses of many IT estimators across a variety of
biomolecular feature-spaces [@schreiber, @grassberger].

*MDEntropy* is actively developed and maintained by researchers at Stanford
University. Source code for *MDEntropy* is hosted on GitHub and is
continuously archived to Zenodo [@mdent_archive]. Full documentation, including
Jupyter Notebook tutorials, can be found at
[http://msmbuilder.org/mdentropy](http://msmbuilder.org/mdentropy).


# References
