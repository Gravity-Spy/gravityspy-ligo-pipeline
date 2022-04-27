.. GravitySpy documentation master file, created by
   sphinx-quickstart on Thu Apr 21 14:05:08 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to GravitySpy's documentation!
======================================

`Gravity Spy <https://gravityspy.org>`_ is an innovative citizen-science meets Machine Learning meets gravitational wave physics project. This repository is meant to faciliate the creation of new similar citizen science projects on `Zooniverse <https://zooniverse.org>`_

The module level docstrings attempt to follow the following format : `Google Style Sphinx <http://www.sphinx-doc.org/en/master/ext/example_google.html>`_


Installing GravitySpy
---------------------

The easiest method to install gravityspy is using `pip <https://pip.pypa.io/en/stable/>`_ directly from the `GitHub repository <https://github.com/Gravity-Spy/GravitySpy.git>`_:

.. code-block:: bash

   $ conda create --name gravityspy-plus-py38 -c conda-forge gwpy python-ldas-tools-frameapi python-ldas-tools-framecpp pandas scikit-image python-lal python-ligo-lw python=3.8 --yes
   $ conda activate gravityspy-plus-py38
   $ python -m pip install git+https://github.com/Gravity-Spy/gravityspy-ligo-pipeline.git

For more details see :ref:`install`.

Publications
------------

If you use Gravity Spy in your scientific publications or projects, we ask that you acknowlege our work by citing the publications that describe Gravity Spy.

* For general citations and information on Gravity Spy use the methods paper : `Zevin et al. Gravity Spy: Integrating Advanced LIGO Detector Characterization, Machine Learning, and Citizen Science <https://iopscience.iop.org/article/10.1088/1361-6382/aa5cea>`_

* `K. Crowston, & The Gravity Spy Collaboration. Gravity Spy: Humans, machines and the future of citizen science <https://citsci.syr.edu/sites/crowston.syr.edu/files/cpa137-crowstonA.pdf>`_

* `K. Crowston, C. Østerlund, T. Kyoung Lee. Blending machine and human learning processes <https://crowston.syr.edu/sites/crowston.syr.edu/files/training%20v3%20to%20share_0.pdf>`_ 

* `T. Kyoung Lee, K. Crowston, C. Østerlund, & G. Miller. Recruiting messages matter: Message strategies to attract citizen scientists <https://citsci.syr.edu/sites/crowston.syr.edu/files/cpa143-leeA.pdf>`_

* `S. Bahaadini, N. Rohani, S. Coughlin, M. Zevin, V. Kalogera, & A. Katsaggelos. Deep multi-view models for glitch classification <https://arxiv.org/pdf/1705.00034.pdf>`_

* For a thorough discussion of versionn 1.0 of the training set used see: `S. Bahaadini, V. Noroozi, N. Rohani, S. Coughlin, M. Zevin, J. R. Smith, V. Kalogera, & A. Katsaggelos. Machine learning for Gravity Spy: Glitch classification and dataset <https://www.sciencedirect.com/science/article/pii/S0020025518301634>`_

* C. Jackson, C. Østerlund, K. Crowston, M. Harandi, S. Allen, S. Bahaadini, S. Coughlin, V. Kalogera, A. Katsaggelos, S. Larson, N. Rohani, J. Smith, L. Trouille, and M. Zevin. [Making High-Performing Contributors: An Experiment With Training in an Online Production Community], submitted to IEEE Transactions on Learning Technologies, 2018.

* S. Bahaadini, V. Noroozi, N. Rohani, S. Coughlin, M. Zevin, & A. Katsaggelos. DIRECT: Deep DIscRiminative Embedding for ClusTering of LIGO Data, submitted to IEEE International Conference on Image Processing, 2018.

Package documentation
---------------------

Please consult these pages for more details on using GravitySpy:

.. toctree::
   :maxdepth: 1

   install/index
   classify/index
   events/index
   trainmodel/index
   examples/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
