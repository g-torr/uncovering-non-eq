import numpy as np
import scipy
import itertools
from collections import Counter,defaultdict
import networkx as nx
import sys
import logging
import os
import pickle
sys.path.insert(0, "../../lib")  # add the library folder to the path I look for modules
import dynamical_cavity
import cavity_symmetric
import simulation

def round_even(x):
    '''round to closest even'''
    return round(x/2.)*2
def save_obj(obj,gamma_G,T,kind):
    directory='gamma_G_'+str(gamma_G)+'/T_'+str(T)
    if not os.path.exists(directory+"/data"):
          os.makedirs(directory+"/data")
    name=  kind +'.pkl'
    '''
    if os.path.isfile(directory+'/data/dic-'+name):
        name = name[:-4]+'_'+ str(time.time())+'.pkl'
    '''
    with open(directory+'/data/dic-' + name , 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def generate_degree_seq(gamma, N):
    kseq = np.ceil( 1+np.random.pareto(gamma, N))
    cond = kseq > N
    while any(cond):
        temp_seq = np.ceil( np.random.pareto(gamma, np.count_nonzero(cond)))
        kseq[cond] = temp_seq
        cond = kseq > N
    if sum(kseq)%2==1:
        kseq[-1]+=1
    return np.array(kseq, dtype=int)
def assign_bidirectional_symmetry(J,kind):
    '''
    return interaction matrix with chosed symmetry of interaction sign'''
    triang = scipy.sparse.triu(J)
    a = triang.tocoo().row
    b = triang.tocoo().col
    if kind == 'symmetric':
        J[b,a] = J[a,b]
    elif kind == 'antisymmetric':
        J[b,a] = -J[a,b]
    elif kind == 'asymmetric':
        J[b,a] = np.where(np.random.rand(len(b))>0.5,1,-1)
    else:
        raise ValueError("kind should be one of those: asymmetric, antisymmetric  symmetric. Instead you select: "+ kind)
    J=J
    return J    

def make_network(N,gamma,bias = 0.5):
    sequence =  generate_degree_seq(gamma,N)
    n_bi_links = round_even(sum(sequence))#number of bidirectional links
    stub_bi_list = list(itertools.chain.from_iterable([[i]*k for i,k in zip(range(len(sequence)),sequence)]))
    a = sorted(Counter(stub_bi_list).items())
    b = defaultdict(lambda:0, a)
    sequence_sym = [b[k] for k in range(N)]#degree sequence of  symmetric links
    G = nx.generators.configuration_model(sequence_sym)
    A_symm = nx.adjacency_matrix(G)
    sign_interaction = np.where(np.random.rand(A_symm.nnz) > bias, 1, -1)  # bias in positive regulation
    A_symm.data = np.array(sign_interaction,dtype=np.float32)
    J = A_symm.tolil()
    return A_symm


def make_epsilon(J):
    N = J.shape[0]
    row = J.tocoo().row
    col = J.tocoo().col
    bi_link = set(zip(row,col))&set(zip(col,row))#set of bidirectional link
    if len(bi_link)==0:
        return scipy.sparse.coo_matrix((N,N))
    a,b = zip(*bi_link)
    return scipy.sparse.coo_matrix((np.ones(len(a),dtype=bool),(np.array(a),np.array(b))),shape = (N,N)).tocsc()



def main():
    '''
    It generate one undirected network, then three interaction matrix are generated:
    - symmetric interaction (J_{ij}==J_{ji})
    - asymmetric interactrion (J_{ij}and J_{ji} independent)
    - antisymmetric interaction (J_{ij}==-J_{ji})
    '''
    no_gpu = False
    try:
        import cupy as cp
        from cupyx.scipy.sparse import csr_matrix as csr_gpu
    except ImportError:
            no_gpu = True
    N = 10000
    gamma = 3
    T = .5
    J = make_network(N,gamma)
    logging.basicConfig(filename='comparison'+str(T)+'.log', level=logging.INFO,format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p')
    logging.info('Network made')
    for kind in ['asymmetric','symmetric','antisymmetric']: #fraction of simmetric couplings
        J_kind = assign_bidirectional_symmetry(J,kind)
        trj = cavity_symmetric.cavity_iteration(J_kind,T,max_iter = 500,precision = 1e-4)
        logging.info('OTA completed for  %s', kind)
        N_replics = 500
        N_iterations = 1e4
        if no_gpu:
            threads = -1 #number of process to be run in parallel, negative if all cores available
            sim = simulation.replics_parallel(J_kind, np.random.rand(J.shape[0]), T, N_replics,N_iterations,threads)
        else:
            sim =  simulation.replics_gpu(csr_gpu(J_kind), cp.random.rand(J.shape[0]), T, N_replics,N_iterations)
        logging.info('simulation completed for  %s', kind)
        P_memless = dynamical_cavity.cavity_caller(J_kind,T,0,J0 = 1,precision=1e-8) #Try without taking into account memory effects
        dic = {'J':J_kind,'trj':trj,'P_memless':P_memless,'sim':sim,'T':T,'N_iterations':N_iterations}
        save_obj(dic,gamma,T,kind)
        logging.info('dictionary saved for  %s', kind)


if __name__ == '__main__':
    main()
