#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yikun zhang

Last edit: June 12, 2019
"""

import numpy as np
import time
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

## Backward algorithm
def Backward(v, P, E, Obser):
    '''
    Input: Initial distribution v; Transition probability P; Emission probability E;
    Observations from HMM: Obers (Integer list with outcomes). 
    Output: (backward probabilities for all the time steps, likelihood value based on 'Obser')
    '''
    n = len(Obser)
    num_S = P.shape[0]
    b = np.empty([num_S, n], dtype='float')
    # Base case
    b[:,n-1] = 1
    # Recursive case
    for t in reversed(range(n-1)):
        b[:,t] = np.dot(P, np.multiply(E[:, Obser[t+1]], b[:, t+1]))
        
    return (b, np.dot(v, np.multiply(E[:, Obser[0]], b[:, 0])))


## Forward algorithm
def Forward(v, P, E, Obser):
    '''
    Input: Initial distribution v; Transition probability P; Emission probability E;
    Observations from HMM: Obers (Integer list with outcomes). 
    Output: (forward probabilities for all the time steps, likelihood value based on 'Obser')
    '''
    n = len(Obser)
    num_S = P.shape[0]
    a = np.empty([num_S, n], dtype='float')
    # Base case
    a[:,0] = np.multiply(v, E[:, Obser[0]])
    # Recursive case
    for t in range(1, n):
        a[:,t] = np.multiply(E[:,Obser[t]], np.dot(P.T, a[:,t-1]))
        
    return (a, np.sum(a[:,n-1]))


## Baum-Welch algorithm
def Baum_Welch(training, M, v_0=0, P_0=0, E_0=0, init=True, accuracy=1e-5, num_iter=10**5, norm=False, LogL=False):
    '''
    Input: Initial distribution v_0; Transition probability P_0; Emission probability E_0; (Initial values for EM);
    The number of states M; An indicator of whether the initial values for EM is provided: 'init';
    Threshold for the relative difference of log-likelihood between two iterations: 'accuracy'; 
    The maximum number of iteration: 'num_iter';
    Binary variable for controlling the stopping criterion: 'norm' (True: Use L2 norm for all parameters)
    Output: (v, P, E)
    '''
    if init:
        v = np.copy(v_0)
        P = np.copy(P_0)
        E = np.copy(E_0)
    else:
        v = np.random.rand(M,)
        v = v/sum(v)
        P = np.random.rand(M, M)
        P = P/np.reshape(P.sum(axis=1), (P.shape[0],1))
        E = np.random.rand(M, np.unique(training).shape[0])
        E = E/np.reshape(E.sum(axis=1), (E.shape[0],1))
    assert M == P.shape[0], 'The number of states does not match up with the size of transition matrix'
    print('The number of states is ' + str(M))
    # Number of sequences
    n = training.shape[0]
    
    LL = []
    ## Iterate EM until converged or the maximum step is met
    for i in range(num_iter):
        v_new = np.zeros_like(v)
        P_new = np.zeros_like(P)
        E_new = np.zeros_like(E)
        old_Loglik = 0
        new_Loglik = 0
        
        for j in range(n):
            T = len(training[j,:])
            # Compute the forward and backward probabilities
            a, a_lik = Forward(v=v, P=P, E=E, Obser=training[j,:])
            b, b_lik = Backward(v=v, P=P, E=E, Obser=training[j,:])
            assert abs(a_lik - b_lik) < 1e-6, 'The observed likelihood computed by forward and backward algorithms does not agree!'
            # Update HMM parameters
            gamma = np.multiply(a, b)
            gamma = gamma/gamma.sum(axis=0)
            ## Update v
            v_new += gamma[:,0]
            ## Update E
            for k in range(T):
                O = training[j,k]
                E_new[:, O] += gamma[:,k]
            ## Update P
            for k in range(1, T):
                O = training[j,k]
                for s1 in range(M):
                    for s2 in range(M):
                        P_new[s1, s2] += b[s2,k]*E[s2,O]*P[s1,s2]*a[s1,k-1]/a_lik
            old_Loglik += np.log(a_lik)
        
        ## Normalize v_new, P_new, E_new
        v_new = v_new/np.sum(v_new)
        P_new = P_new/np.reshape(P_new.sum(axis=1), (P_new.shape[0],1))
        E_new = E_new/np.reshape(E_new.sum(axis=1), (E_new.shape[0],1))
        
        for j in range(n):
            a, a_lik = Forward(v=v_new, P=P_new, E=E_new, Obser=training[j,:])
            new_Loglik += np.log(a_lik)
        
        LL.append(new_Loglik)
        # Choose a stopping criterion
        if norm:
            if np.sqrt(np.linalg.norm(v-v_new, ord=2)**2 + np.linalg.norm(P-P_new, ord=2)**2 + np.linalg.norm(E-E_new, ord=2)**2)<= accuracy:
                v = np.copy(v_new)
                P = np.copy(P_new)
                E = np.copy(E_new)
                break
        else:
            if abs(old_Loglik - new_Loglik)/abs(old_Loglik) <= accuracy:
                break
        
        v = np.copy(v_new)
        P = np.copy(P_new)
        E = np.copy(E_new)
        # print(new_Loglik/n)
        
    print('The number of iteration steps is ' + str(i+1))
    ## Relabel the states based on the uniformity of emission probabilities
    new_ind = np.var(E, axis=1).argsort()
    E = E[new_ind,:]
    P = P[new_ind,:]
    # Reorder the column of P too
    P = P[:,new_ind]
    v = v[new_ind]
    
    if LogL:
        return(v, P, E, LL)
    else:
        return (v, P, E)



## Viterbi algorithm
def Viterbi(v, P, E, Obser):
    '''
    Input: Initial distribution v; Transition probability P; Emission probability E;
    Observations from HMM: Obers (Integer list with outcomes). 
    Output: (the most likely sequence of states, maximal joint probability)
    '''
    n = len(Obser)
    num_S = P.shape[0]
    d = np.empty([num_S, n])
    f = np.empty([num_S, n-1], dtype=int)  ## Matrix for backtracking
    # Base case
    # d[:,0] = np.multiply(v, E[:,Obser[0]])
    d[:,0] = np.log(np.multiply(v, E[:,Obser[0]]))
    # Recursive case
    for t in range(1, n):
        for i in range(num_S):
            # temp_vec = np.multiply(d[:,t-1], P[:,i])
            temp_vec = d[:,t-1] + np.log(P[:,i])
            f[i,t-1] = np.argmax(temp_vec)
            # d[i,t] = E[i, Obser[t]] * np.max(temp_vec)
            d[i,t] = np.log(E[i, Obser[t]]) + np.max(temp_vec)
    
    p_star = np.exp(np.max(d[:,n-1]))
    most_lik = []
    # The last element of the most likely sequence of states
    x = np.argmax(d[:,n-1])
    most_lik.append(x)
    # Backtracking
    for t in reversed(range(n-1)):
        x = f[x,t]
        most_lik.append(x)
        
    return (list(reversed(most_lik)), p_star)


def AIC(L_Lik, M, k):
    '''
    Input: Observed log-likelihood: L_Lik; Number of states: M; Number of observed states: k. 
    Output: AIC-score
    '''
    return -2*L_Lik + 2*(M^2 + (k-1)*M -1)

def BIC(L_Lik, M, k, T):
    '''
    Input: Observed log-likelihood: L_Lik; Number of states: M; Number of observed states: k; Length of the observed sequence: T
    Output: BIC-score
    '''
    return -2*L_Lik + (M^2 + (k-1)*M -1)*np.log(T)


def Data_preprocess(file_dir):
    ### Load the training set
    with open(file_dir) as file:
        read_strings = file.read().splitlines()

    # Split each line by blanks
    data = [line.split(' ') for line in read_strings]
    # Eliminate the last element '' from each line
    data = [line[:len(line)-1] for line in data]
    # Store the training set in a numpy array
    train = np.array(data, dtype=int)
    return train


def Cross_validation(train, M=list(range(3, 11)), seed=123):
    # Setting up the 5-fold cross-validation
    kf = KFold(n_splits=5, random_state=123, shuffle=True)
    Log_lik = []
    BIC_score = []
    AIC_score = []

    for i in range(len(M)):
        np.random.seed(seed)
        # Randomly initialize the parameters
        pi_0 = np.random.rand(M[i],)
        pi_0 = pi_0/sum(pi_0)
        A_0 = np.random.rand(M[i], M[i])
        A_0 = A_0/np.reshape(A_0.sum(axis=1), (A_0.shape[0],1))
        B_0 = np.random.rand(M[i], np.unique(train).shape[0])
        B_0 = B_0/np.reshape(B_0.sum(axis=1), (B_0.shape[0],1))
    
        Llik = []
        BIC_s = []
        AIC_s = []
        for train_ind, test_ind in kf.split(train):
            # Setting up the training set and validation set
            d_train = train[train_ind,:]
            d_test = train[test_ind,:]
            # Run the Baum-Welch algorithm
            pi_c, A_c, B_c = Baum_Welch(training=d_train, v_0=pi_0, P_0=A_0, E_0=B_0, M=M[i], accuracy=1e-5, num_iter=10**5)
            Log_Likelihood = 0
            for k in range(d_test.shape[0]):
                # Compute the log-likelihood for the sequence k in the validation set
                Log_Likelihood += np.log(Backward(v=pi_c, P=A_c, E=B_c, Obser=d_test[k,:])[1])
            
            Llik.append(Log_Likelihood/d_test.shape[0])
            BIC_s.append(BIC(L_Lik=Log_Likelihood/d_test.shape[0], M=M[i], k=np.unique(d_test).shape[0], T=d_test.shape[1]))
            AIC_s.append(AIC(L_Lik=Log_Likelihood/d_test.shape[0], M=M[i], k=np.unique(d_test).shape[0]))
        
        # Record the average log-likelihood, BIC, and AIC scores across all folds
        Log_lik.append(np.mean(Llik))
        BIC_score.append(np.mean(BIC_s))
        AIC_score.append(np.mean(AIC_s))
    
    return Log_lik, BIC_score, AIC_score


def cv_plot(path, fig_path, write=True, Log_lik=None, AIC_score=None, BIC_score=None):
    if write:
        # Write the cross-validation results to the file
        with open(path, 'w') as file:
            file.write('log-likelihood ' + " ".join([str(k) for k in Log_lik]))
            file.write('\n')
            file.write('AIC ' + " ".join([str(k) for k in AIC_score]))
            file.write('\n')
            file.write('BIC ' + " ".join([str(k) for k in BIC_score]))
    # Read the cross-validation results from the file
    with open(path) as f:
        strings = f.read().splitlines()
    
    Log_lik = strings[0].split(' ')[1:]
    Log_lik = [float(k) for k in Log_lik]
    AIC_score = strings[1].split(' ')[1:]
    AIC_score = [float(k) for k in AIC_score]
    BIC_score = strings[2].split(' ')[1:]
    BIC_score = [float(k) for k in BIC_score]
    M = list(range(3, 3+len(Log_lik)))
    
    fig = plt.figure(figsize=(8,6))
    plt.plot(M, Log_lik, 'bo-', label='Log-likelihood')
    plt.plot(M, AIC_score, label='AIC')
    plt.plot(M, BIC_score, label='BIC')

    plt.legend(fontsize=15)
    plt.xlabel("The number of states", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('')
    plt.tight_layout()
    plt.show()
    fig.savefig(fig_path)
    

## Predicts the next output and the next state given a sequence of observations
def Predict(v, P, E, Obser, h=1, viterbi=False):
    '''
    Input: Initial distribution v; Transition probability P; Emission probability E;
    Observations from HMM: Obers (Integer list with outcomes); Time step: h=1; An indicator of whether the Viterbi
    is used to predict the next hidden state: viterbi=False.
    Output: (the next output of HMM, the next state, output probabilities)
    '''
    n = len(Obser)
    # compute forward probabilities
    a, a_lik = Forward(v=v, P=P, E=E, Obser=Obser)
    # Compute the h-step transition probability matrix 
    P_h = np.linalg.matrix_power(P, h)
    # Compute the likelihood for the next output
    pred_lik = np.dot(a[:,n-1].T, np.dot(P_h, E))/a_lik
    # Predict the next output based on the calculated likelihood
    output = np.argmax(pred_lik)
    if viterbi:
        # Apply Viterbi algorithm to construct the next state
        seq, p_star = Viterbi(v=v, P=P, E=E, Obser=np.append(Obser, output))
        state = seq[-1]
    else:
        # Compute the likelihood for the next state
        pred_lik_state = np.dot(a[:,n-1].T, P_h)/a_lik
        # Predict the next state
        state = np.argmax(pred_lik_state)
        
    return (output, state, pred_lik)


# Choose a seed with maximal accuracy
def cv_seed(fig_path='output/acc_seed.pdf', seed=[123, 1, 101, 201, 301, 384], M=4):
    acc_seed = []
    log_lik = []
    for N in seed:
        np.random.seed(N)
        # Randomly initialize the parameters
        pi_0 = np.random.rand(M,)
        pi_0 = pi_0/sum(pi_0)
        A_0 = np.random.rand(M, M)
        A_0 = A_0/np.reshape(A_0.sum(axis=1), (A_0.shape[0],1))
        B_0 = np.random.rand(M, np.unique(train).shape[0])
        B_0 = B_0/np.reshape(B_0.sum(axis=1), (B_0.shape[0],1))

        pi_c, A_c, B_c = Baum_Welch(training=train, v_0=pi_0, P_0=A_0, E_0=B_0, M=M, accuracy=1e-5, num_iter=10**5)
        
        ## Computing log-likelihood and predicting on all the given sequences. The first T-1 elements are provided and we predict the Tth element.
        pred_hmm2 = np.empty([train.shape[0], ], dtype=int)
        logl = 0
        for i in range(train.shape[0]):
            logl += np.log(Backward(v=pi_c, P=A_c, E=B_c, Obser=train[i,:])[1])
            output, state, prob = Predict(v=pi_c, P=A_c, E=B_c, Obser=train[i,0:(train.shape[1]-1)], h=1)
            pred_hmm2[i] = output
        
        log_lik.append(logl/train.shape[1])
        acc_seed.append(np.mean(pred_hmm2 == train[:,train.shape[1]-1]))
        
    fig = plt.figure(figsize=(8,6))
    log_lik = np.array(log_lik)
    log_lik1 = (log_lik - min(log_lik))/max(log_lik)
    log_lik2 = log_lik1.tolist()
    seed1 = sorted(seed)
    acc_seed1 = [acc_seed[i] for i in np.argsort(seed).tolist()]
    log_lik2 = [log_lik2[i] for i in np.argsort(seed).tolist()]
    print(seed1)
    print(acc_seed1)
    print(log_lik2)
    plt.plot(seed1, acc_seed1, label = 'Accuracy')
    # plt.plot(seed1, log_lik2, label = 'Log-likelihood')
    
    # plt.legend()
    plt.xlabel("The values of random seed", fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('')
    plt.tight_layout()
    plt.show()
    fig.savefig(fig_path)
    

def Plot_Learning(Log_lik, fig_dir):
    N = list(range(1, len(Log_lik)+1))
    
    fig = plt.figure(figsize=(8,6))
    plt.plot(N, Log_lik, 'go-')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel("Iteration Step", fontsize=15)
    plt.ylabel('Log-likelihood', fontsize=15)
    # plt.show()
    plt.tight_layout()
    fig.savefig(fig_dir)

def main():
    # Load the data
    train = Data_preprocess("data/train534.dat")
    # Uncomment it to utilize 5-fold cross-validation to select the number of states (Time-consuming)
    '''
    Log_lik, BIC_score, AIC_score = Cross_validation(train=train, M=list(range(3, 11)), seed=384) # 11h 23m 5s
    Log_lik, BIC_score, AIC_score = Cross_validation(train=train, M=list(range(3, 11)), seed=123) # 11h 31m 7s
    cv_plot(path='cv384.txt', fig_path='cv384.pdf', write=True, Log_lik=Log_lik, AIC_score=AIC_score, BIC_score=BIC_score)
    cv_plot(path='cv123.txt', fig_path='cv123.pdf', write=True, Log_lik=Log_lik, AIC_score=AIC_score, BIC_score=BIC_score)
    '''
    # Uncomment it to use prediction accuracy to select the initial random seed for Baum-Welch algorithm
    # cv_seed(fig_path='acc_seed.pdf', seed=[123, 1, 101, 201, 301, 384], M=4)
    
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
    pi_c, A_c, B_c, LL_c = Baum_Welch(training=train, v_0=pi_0, P_0=A_0, E_0=B_0, M=M, accuracy=1e-5, num_iter=10**5, LogL=True)
    print('The running time of Baum-Welch algorithm is '+str(time.time()-BW_start))
    print('\n')
    print('The transition matrix A is ')
    print(A_c)
    print('\n')
    print('The emission probability matrix B is ')
    print(B_c)
    print('\n')
    
    Plot_Learning(Log_lik=LL_c, fig_dir='learning_curve.pdf')
    
    # Use Viterbi algorithm to calculate the most likely sequence of states for all the data
    mostLik_seq = np.copy(train)
    ## Find the most likely sequence of states for each observed sequence
    start_vertibi = time.time()
    for i in range(train.shape[0]):
        mostLik_seq[i,:] = np.array(Viterbi(v=pi_c, P=A_c, E=B_c, Obser=train[i,:])[0])
    print('The running time for the Viterbi algorithm on all the training data is ' + str(time.time()- start_vertibi))
    
    # Predicts the next output and the next state given 1:39 observations
    pred_output = []
    pred_state = []
    pred_prob = np.empty([train.shape[0], len(np.unique(train))], dtype='float')
    for i in range(train.shape[0]):
        output, state, prob = Predict(v=pi_c, P=A_c, E=B_c, Obser=train[i,0:(train.shape[1]-1)], h=1)
        pred_output.append(output)
        pred_state.append(state)
        pred_prob[i,:] = prob
    print('The prediction accuracy on the training set is ' + str(np.mean(pred_output == train[:, train.shape[1]-1])))


if __name__ == '__main__':
    main()
