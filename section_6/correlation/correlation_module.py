from ctypes.wintypes import POINT
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from functools import lru_cache

import itertools
import time
import os
import pickle
import scipy
import random
import sys
sys.path.insert(0, "../../../lib")  # add the library folder to the path I look for modules
sys.path.insert(0, "../../lib")  # add the library folder to the path I look for modules specific to symmetric matrix
import latexify
import cavity_symmetric
def directory(kin):
    return 'kin='+str(kin)
def save_obj(obj,kin,T,kind):
    if not os.path.exists(directory(kin)+"/data"):
          os.makedirs(directory(kin)+"/data")
    name=kind+'_T='+str(T)+'.pkl'
    if os.path.isfile(directory(kin)+'/data/dic-' + name ):
        name = name[:-4]+'_'+ str(time.time())+'.pkl'
    with open(directory(kin)+'/data/dic-' + name , 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def replics_parallel(J, P_init, T, N_replics,N_iterations,threads):
    '''Simulation at fixed T for different replicas
    Initial condition is chosen to be the same as for cavity'''
    '''
    if threads < 0:
        pool = Pool()
    else:
        pool = Pool(int(threads))
    '''
    different_seed = np.random.randint(1, 2 ** 32 - 1, N_replics)
    from multiprocessing import Pool
    if threads < 0:
        pool = Pool()
    else:
        pool = Pool(int(threads))         
    data = pool.starmap(dynamics_parallel, itertools.product([J], [P_init], [T],different_seed, [N_iterations]))
    pool.close()
    cutoff_correlation = 1000
    C_run_mean = np.zeros(N_iterations)
    C = np.zeros((N,N_iterations))
    m = np.zeros(N)
    for trj in data:
        m+=np.mean(trj,axis = 1) 
        for ind,a in enumerate(trj):
            a = a-np.mean(a)
            C[ind]+= np.correlate(a,a,'same')
    C_run_mean=C/N_replics
    lag = np.arange(-np.floor(C.shape[1]/2),np.floor(C.shape[1]/2+0.5),dtype = int)
    N_denominator = N_iterations-np.abs(lag) #number of terms in the sum
    return (C_run_mean/N_denominator)[:,(lag>-1)&(lag<cutoff_correlation)],m/N_replics


def replics_gpu(J, P_init, T, N_replics,N_iterations):
    '''Simulation at fixed T for different replicas
    Initial condition is chosen to be the same as for cavity'''
    '''
    if threads < 0:
        pool = Pool()
    else:
        pool = Pool(int(threads))
    '''
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as csr_gpu
    N = J.shape[0]
    initial_states = cp.where(cp.random.rand(N_replics,N,dtype=cp.float32) > P_init, 1, 0)
    data = itertools.starmap(dynamics_gpu, itertools.product([J], initial_states, [T], [N_iterations]))
    # for replica in range(N_replic):
    #        data+=[dynamics_light(J,psi_init,T)]
    #pool.close()
    cutoff_correlation = 1000
    C_run_mean = np.zeros(N_iterations)
    C = np.zeros((N,N_iterations))
    m = cp.zeros(N)
    for trj in data:
        m+=cp.mean(trj,axis = 1) 
        for ind,a in enumerate(trj):
            a = a-cp.mean(a)
            C[ind]+= cp.correlate(a,a,'same').get()
    C_run_mean=C/N_replics
    lag = np.arange(-np.floor(C.shape[1]/2),np.floor(C.shape[1]/2+0.5),dtype = int)
    N_denominator = N_iterations-np.abs(lag) #number of terms in the sum
    return (C_run_mean/N_denominator)[:,(lag>-1)&(lag<cutoff_correlation)],m.get()/N_replics

def dynamics_gpu(J, n, T,N_iterations):
    if not (type(J) is csr_gpu):
        #print('Coupling matrix should be of type'+str(csr_gpu)+', I try to convert')
        J = csr_gpu(J)
    #local_state = cp.random.seed(seed)
    N = J.shape[0]
    N_therm = 100
    t = 0
    while t < N_therm:
        z = cp.random.logistic(0, T , (1, N))
        # z=numpy.random.normal(0,2*T,(1,N))
        a = n * J - z
        n = cp.where(a > 0, 1, 0.)[0]
        t += 1
    t = 0
    #m = cp.zeros(N)
    trj = cp.zeros((N,N_iterations))
    while t < N_iterations:
        z = cp.random.logistic(0, T , (1, N),dtype=cp.float32)
        # z=numpy.random.normal(0,2*T,(1,N))
        a = n * J - z
        n = cp.where(a > 0, 1, 0.)[0]
        trj[:,t]= n
        t += 1
    return trj

def dynamics_parallel(J, P_init, T, seed,N_iterations):
    local_state = np.random.RandomState(seed)
    N = J.shape[0]
    N_therm = 100
    n_start = np.where(np.random.rand(N) > P_init, 1, 0)
    n = scipy.sparse.csr_matrix(n_start)
    t = 1

    while t < N_therm:
        z = local_state.logistic(0, T, (1, N))
        # z=numpy.random.normal(0,2*T,(1,N))
        a = (n * J).toarray() - z
        n = np.where(a > 0, 1, 0)
        n = scipy.sparse.csr_matrix(n)
        t += 1
    t = 0
    #N_iterations =max(int(np.log(1-alpha)/np.log(0.5*(1+np.tanh(1/2/T)))),1000)# number iterations grows at low temperature. See notes

    trj = np.zeros((N,N_iterations))
    time_average=0
    while t < N_iterations:
        z = local_state.logistic(0, T , (1, N))
        # z=numpy.random.normal(0,2*T,(1,N))
        a = (n * J).toarray() - z
        n = np.squeeze(np.where(a > 0, 1, 0))
        trj[:,t]= n
        n = scipy.sparse.csr_matrix(n)
        t += 1
    return trj
def correlation_cavity(J,T,theta,P_A,P_B):
    epsilon = cavity_symmetric.make_epsilon(J)
    N = J.shape[0]
    J_transpose = J.transpose()
    js = J_transpose.tolil().rows
    interaction = J_transpose.tolil().data
    corr_cav = np.zeros(N)
    for i in range(N):
        cav_neigh = np.array(js[i])
        P_B_temp = np.where(epsilon[cav_neigh,i].toarray(),P_B[cav_neigh,i].toarray(),P_A[cav_neigh,i].toarray())
        corr_cav[i]= cavity_symmetric.cavity(P_B_temp,inter = np.array(interaction[i]),T = T,theta=theta)
    return corr_cav
