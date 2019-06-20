#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yikun zhang

Last edit: June 12, 2019
"""

import numpy as np
import time
from HMM import Data_preprocess, Backward, Forward, Baum_Welch, Viterbi, Predict

# read the training and test set
train = Data_preprocess("data/train534.dat")
test = Data_preprocess("data/test1_534.dat")

M = 4
np.random.seed(301)
# Randomly initialize the parameters
pi_0 = np.random.rand(M,)
pi_0 = pi_0/sum(pi_0)
A_0 = np.random.rand(M, M)
A_0 = A_0/np.reshape(A_0.sum(axis=1), (A_0.shape[0],1))
B_0 = np.random.rand(M, np.unique(train).shape[0])
B_0 = B_0/np.reshape(B_0.sum(axis=1), (B_0.shape[0],1))
    
# Use Baum-Welch to estimate the parameters for HMM
BW_start = time.time()
pi_c, A_c, B_c = Baum_Welch(training=train, v_0=pi_0, P_0=A_0, E_0=B_0, M=M, accuracy=1e-5, num_iter=10**5)
print('The running time of Baum-Welch algorithm is '+str(time.time()-BW_start))
print('\n')
print('The transition matrix A is ')
print(A_c)
print('\n')
print('The emission probability matrix B is ')
print(B_c)
print('\n')

test_loglik_forward = 0
for i in range(test.shape[0]):
    a, a_lik = Forward(v=pi_c, P=A_c, E=B_c, Obser=test[i,:])
    test_loglik_forward += np.log(a_lik)
    
test_loglik_backward = 0
for i in range(test.shape[0]):
    b, b_lik = Backward(v=pi_c, P=A_c, E=B_c, Obser=test[i,:])
    test_loglik_backward += np.log(b_lik)

assert test_loglik_forward == test_loglik_backward, "The computation of log-likelihood is not correct."
print('The log-likelihood on the test set is ' + str(test_loglik_forward))

# Use Viterbi algorithm to calculate the most likely sequence of states for test set
mostLik_seq_test = np.copy(test)
## Find the most likely sequence of states for each observed sequence
start_vertibi = time.time()
for i in range(test.shape[0]):
    mostLik_seq_test[i,:] = np.array(Viterbi(v=pi_c, P=A_c, E=B_c, Obser=test[i,:])[0])

print('The running time for the Viterbi algorithm on the test set is ' + str(time.time()- start_vertibi))

pred_output1 = []
pred_state1 = []
pred_prob = np.empty([test.shape[0], len(np.unique(test))], dtype='float')
pred_output_viterbi = []
pred_state_viterbi = []
for i in range(test.shape[0]):
    output, state, prob = Predict(v=pi_c, P=A_c, E=B_c, Obser=test[i,:], h=1)
    # Record the prediction probabilities for the output
    pred_prob[i,:] = prob
    output2, state2, prob2 = Predict(v=pi_c, P=A_c, E=B_c, Obser=test[i,:], h=1, viterbi=True)
    pred_output1.append(output)
    pred_state1.append(state)
    pred_output_viterbi.append(output2)
    pred_state_viterbi.append(state2)

# Write parameters and outputs to the files as required
with open('modelpars.dat', 'w') as f:
    f.write(str(M))
    f.write('\n')
    for i in range(M):
        A_str = [str(k) for k in A_c[i,:]]
        f.write(",".join(A_str))
        f.write('\n')
        
    for i in range(M):
        B_str = [str(k) for k in B_c[i,:]]
        f.write(",".join(B_str))
        f.write('\n')
    
    pi_str = [str(k) for k in pi_c]
    f.write(",".join(pi_str))

with open('loglik.dat', 'w') as f:
    f.write(str(test_loglik_forward))

with open('viterbi.dat', 'w') as f:
    for i in range(test.shape[0]):
        v_str = [str(k) for k in mostLik_seq_test[i,:]]
        f.write(",".join(v_str))
        f.write('\n')

with open('predict.dat', 'w') as f:
    for i in range(pred_prob.shape[0]):
        p_str = [str(k) for k in pred_prob[i,:]]
        f.write(",".join(p_str))
        f.write('\n')
