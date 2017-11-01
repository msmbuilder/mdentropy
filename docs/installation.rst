Installation
============

MDEntropy is written in Python, and can be installed with standard Python
machinery; although, we highly recommend using an
`Anaconda Python distribution <https://www.continuum.io/downloads>`_.


Release Version
---------------


With Anaconda, installation is as easy as:

.. code-block:: bash

  $ conda install -c omnia mdentropy

You can also install mdentropy with `pip`:

.. code-block:: bash

  $ pip install mdentropy

Alternatively, you can install directly our
`GitHub repository <https://github.com/msmbuilder/mdentropy>`_.:

.. code-block:: bash

  $ git clone https://github.com/msmbuilder/mdentropy.git
  $ cd mdentropy && git checkout v0.3.0
  $ python setup.py install


Development Version
-------------------

To grab the latest version from github, run:

.. code-block:: bash

  $ pip install git+git://github.com/pandegroup/mdentropy.git

Or clone the repo yourself and run `setup.py`:

.. code-block:: bash

  $ git clone https://github.com/pandegroup/mdentropy.git
  $ cd mdentropy && python setup.py install


Dependencies
------------
- ``python>=3.4``
- ``numpy>=1.10.4``
- ``scipy>=0.17.0``
- ``scikit-learn>=0.17.0``
- ``msmbuilder>=3.5.0``
- ``nose`` (optional, for testing)

You can grab most of them with conda. ::

  $ conda install -c omnia scipy numpy scikit-learn msmbuilder nose
