# MolPredbyMPNN

common molecule affinity regression and classification predictors have been built by MPNN, DNN, and machine learning methods(SVM, RF, KNN, NB, PLS), and there are both regression and classification models.

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

* ML-predictor/train.py 

  Training the predictor under the machine learning methods
* ML-predictor/models/classifier.py

  DNN algorithm
* ML-predictor/utils/objective.py

  calculating the fingerprint of moleculer and Physical and Chemical Properties
  
* ML-predictor/utils/sascorer.py

  calculating the SA
* mpnn-predictor/Train.py

  Training the predictor via MPNN algorithm
  
* models/*.py
  It contains all of the deep learning models that possibly used in this project
