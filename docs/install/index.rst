.. _install:

############
Installation
############


=====
Conda
=====

Linux
-----

.. code-block:: bash


   $ conda create --name gravityspy-py38 python=3.8
   $ conda activate gravityspy-py38
   $ python -m pip install git+https://github.com/Gravity-Spy/gravityspy-ligo-pipeline.git


=======================
Pre-Existing VirtualEnv
=======================

On CIT, LLO, and LHO

.. code-block:: bash

   $ source /cvmfs/oasis.opensciencegrid.org/ligo/sw/conda/etc/profile.d/conda.sh
   $ conda activate /home/gravityspy/.conda/envs/gravityspy-gpu-py37/
