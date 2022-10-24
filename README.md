# MolPredbyMPNN

Build a molecule affinity values  predictor by MPNN 

## Dataset
* covid-19 3CL pro 的已知活性分子
## Dependencies
Firstly, ensure that the version of your Python >= 3.7. We recommend Anaconda to manage the version of Python and installed packages.
* numpy >= 1.19
* Scikit-Learn >= 0.23
* Pandas >= 1.2.2
* PyTorch == 1.6
* Matplotlib >= 2.0
* RDKit >= 2020.03
## Usage
* train.py 

  Training the predictor under the MPNN
* models/*.py

  It contains all of the deep learning models that possibly used in this project
