# HMM
Python3 Implementation of Hidden Markov Model

## Requirements
- Python >= 3.6 (Earlier version could be applicable.)
- [NumPy](http://www.numpy.org/), [Matplotlib](https://matplotlib.org/), [scikit-learn](https://scikit-learn.org/stable/index.html) (Only the function [sklearn.model_selection.KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html) for splitting the training set is used.)

## Descriptions
The standard functions in a homogeneous multinomial hidden Markov model with discrete state spaces are implmented. The initial probability distribution, transition probability matrix, and emission probability matrix are stored in [NumPy](http://www.numpy.org/) arrays. To use these functions, 
```bash
from HMM import *
```
For instance, in order to envoke the Forward algorithm to compute the forward probabilities and the observed likelihood value, 
```bash
import numpy as np

A = np.array([[0.6, 0.2, 0.2], [0.5, 0.3, 0.2], [0.4, 0.1, 0.5]])   ## Transition probability matrix (|states|*|states|)
pi = np.array([0.5, 0.2, 0.3])   ## Initial probability distribution
O = np.array([[0.6, 0.1, 0.1, 0.2], [0.1, 0.5, 0.3, 0.1], [0.15, 0.15, 0.3, 0.4]])  ## Emission probability matrix  (|states|*|outcomes|)

print(Forward(v=pi, P=A, E=O, Obser=[0, 0, 1]))
```
All the hidden states and outcomes are encoded as integer values. In addition, the Baum-Welch algorithm returns the transition probability matrix and emission probability matrix based on the variances of emission probabilities. (Suppose that there are M hidden states. In state 0, emission probabilities are [0.3  0.3  0.2  0.2]; in state 1 they are [0.5  0.2  0.3  0]; in state 2 they are [0.9  0.1  0  0]; etc.
All other detailed descriptions of the notations and implementations can be found in the [final report](). 
