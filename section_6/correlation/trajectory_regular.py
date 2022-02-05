import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from functools import lru_cache
import cupy as cp
from cupyx.scipy.sparse import csr_matrix as csr_gpu
import itertools
import time
import os
import pickle
import scipy
import random
import correlation_module
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

def generate_degree_seq(gamma, N):
    kseq = np.ceil( 1+np.random.pareto(gamma, N))
    cond = kseq > N
    while any(cond):
        temp_seq = np.ceil( np.random.pareto(gamma, np.count_nonzero(cond)))
        kseq[cond] = temp_seq
        cond = kseq > N
    return np.array(kseq, dtype=int)
def asymmetric_sign(J):
        sign_interaction = np.where(np.random.rand(J.nnz) > 0.5, 1, -1) #assign random sign to links, sign not symmetric
        J.data = np.array(sign_interaction,dtype=np.float32)
        return J
def make_network(N,kin,kind):
    choises = {'symmetric':symmetric_sign,'antisymmetric':antisymmetric_sign,'asymmetric':asymmetric_sign}
    
    #sequence =  generate_degree_seq(gamma,N)
    #make oriented network
    G = nx.generators.degree_seq.configuration_model([kin]*N)
    #G = nx.generators.degree_seq.directed_configuration_model(sequence, np.random.permutation(sequence))
    #G = nx.DiGraph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    J = nx.adjacency_matrix(G)
    J = choises[kind](J)#select the symmetry of network interactions
    return J
def symmetric_sign(J):
    N = J.shape[0]
    row = J.tocoo().row
    col = J.tocoo().col
    cond = row>col
    interaction = np.where(np.random.rand(np.count_nonzero(cond))>.5,1,-1).astype(np.float32)
    A = scipy.sparse.coo_matrix((interaction,(row[cond],col[cond])),shape = (N,N))
    J = (A+A.T).tocsc()
    return J
def antisymmetric_sign(J):
    N = J.shape[0]
    row = J.tocoo().row
    col = J.tocoo().col
    cond = row>col
    interaction = np.where(np.random.rand(np.count_nonzero(cond))>.5,1,-1).astype(np.float32)
    A = scipy.sparse.coo_matrix((interaction,(row[cond],col[cond])),shape = (N,N))
    J = (A-A.T).tocsc()
    return J

def main():
    N = 10000
    kin = 3
    T = 1.5
    theta = 0
    N_replics = 100
    N_iterations = 10000
    for kind in ['symmetric','asymmetric','antisymmetric']:
        #kind = 'symmetric'
        J = make_network(N, kin,kind=kind)
        if no_gpu:
            threads = -1
            C,P_sim = correlation_module.replics_parallel(J, np.random.rand(N), T, N_replics, N_iterations,threads)
        else:
            C,P_sim = correlation_module.replics_gpu(J, cp.random.rand(N), T, N_replics,N_iterations)


        P_A,P_B,P_t = cavity_symmetric.cavity_iteration(J,T,max_iter = 10)
        dic = {'C':C,'P_cav':P_t,'P_sim': P_sim,'N_replics':N_replics,'N_iterations':N_iterations,'T':T,'N':N,'kin':kin,'J':J, 'descr' : 'C has 2 dimension, dimension 0 runs over the nodes, dimension 1 gives the lag up to cutoff. C is averaged over replics'}
        save_obj(dic,kin,T,kind)
        print('saved '+kind)


if __name__ == '__main__':
    main()
