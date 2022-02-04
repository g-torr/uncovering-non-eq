import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
import itertools
import warnings
import scipy
import random
import argparse
import pickle
from functools import lru_cache
import sys
from multiprocessing import Pool
import os
sys.path.insert(0, "../lib")  # add the library folder to the path I look for modules
from dynamical_cavity import  cavity_AND_parallel
from configurational_model_regulatory import mirroring
def directory(gamma_G,gamma_TF):
    path = '.'+os.path.dirname(__file__)
    return path+'/gamma_G_'+str(gamma_G)+'gamma_TF_'+str(gamma_TF)

def save_obj(obj,gamma_G,gamma_TF,theta):
    directory_ = directory(gamma_G,gamma_TF)
    if not os.path.exists(directory+"/data"):
          os.makedirs(directory_+"/data")
    name="theta_" + str(theta)+'.pkl'
    if os.path.isfile(directory_+'/data/dic-'+name):
        name = name[:-4]+'_'+ str(time.time())+'.pkl'
    with open(directory_+'/data/dic-' + name , 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def create_graph( N1,N2,gamma_G,gamma_TF,bias= 0.5):
    '''Generate graph according with power law distribution, in-degree sequence of genes and out-degree  sequence of TFs are the same'''
    def generate_degree_seq(gamma, N):
        kseq = np.ceil(np.random.pareto(gamma, N))
        cond = kseq > N
        while any(cond):
            temp_seq = np.ceil(np.random.pareto(gamma, np.count_nonzero(cond)))
            kseq[cond] = temp_seq
            cond = kseq > N
        return np.array(kseq, dtype=int)

    aseq = generate_degree_seq(gamma_G, N1)
    bseq = generate_degree_seq(gamma_TF, N2)
    if sum(bseq) < N1:
        raise ValueError('degree sequence non compatible for regulations')
    # generate the graph

    BG = mirroring(aseq, bseq, nx.MultiDiGraph())
    M = bipartite.biadjacency_matrix(BG, range(N1))  # membership matrix
    R = bipartite.biadjacency_matrix(BG, range(N1, N1 + N2), format="csc")  # regulatory matrix
    avg_degree = len(R.data) / N1
    sign_interaction = np.where(np.random.rand(R.nnz) > bias, 1, -1)  # bias in sign of  regulation
    R.data = np.ravel(sign_interaction)/ np.sqrt(avg_degree)
    return M,R

def replics_parallel(R,M, P_init, T, N_replics,N_iterations):
    '''Simulation at fixed T for different replicas
    Initial condition is chosen to be the same as for cavity. I am not using here, but they may be useful if one wants to check cavity against simulations'''
    pool = Pool()
    N1,N2 = M.shape
    interaction = [R.data[R.indptr[i]:R.indptr[i + 1]] for i in range(N1)]  # list of list, structure is [el[i]]
    #interaction_sum = np.sum(interaction, axis=1)
    theta = 0 #interaction_sum / 2
    data = pool.starmap(dynamics_light_parallel, itertools.product([R],[M] ,[P_init], [T], [theta], range(N_replics), [N_iterations]))
    # for replica in range(N_replic):
    #        data+=[dynamics_light(J,psi_init,T)]
    pool.close()
    return data
def dynamics_light_parallel(R,M, P_init, T,theta, process,N_iterations):
    random.seed(process)
    N1,N2 = M.shape
    N_therm = 100
    n_start = np.where(numpy.random.rand(N1) > P_init, 1, 0)
    n = scipy.sparse.csr_matrix(n_start)
    M = M.tocsc()
    in_deg = np.diff(M.indptr)  # in degree of each TF
    t = 1
    while t < N_therm:
        tau=np.where((n*M).toarray()!=in_deg,0,1)# fancy way to compute AND logic for TF membership.
        z = numpy.random.logistic(0, T, (1, N1))
        # z=numpy.random.normal(0,2*T,(1,N))
        n=scipy.sparse.csr_matrix(tau)*R
        n=np.where(n.toarray()-z-theta>0,1,0)
        n=scipy.sparse.csr_matrix(n)
        t += 1
    t = 0
    m = np.zeros(N1)
    #C = zeros((N,N))
    while t < N_iterations:
        tau=np.where((n*M).toarray()!=in_deg,0,1)# fancy way to compute AND logic for TF membership.
        z = numpy.random.logistic(0, T, (1, N1))
        # z=numpy.random.normal(0,2*T,(1,N))
        n=scipy.sparse.csr_matrix(tau)*R
        n=np.where(n.toarray()-z-theta>0,1,0)
        m += n[0]
        n=scipy.sparse.csr_matrix(n)
        t += 1
    return m / N_iterations  #, C/N_iterations



def load_input(args):
    N1 = args.N1  # number genes
    N2 = args.N2  # number genes
    gamma_G = args.gamma_G
    gamma_TF = args.gamma_TF
    return N1,N2,gamma_G,gamma_TF

def load_obj(gamma_G,gamma_TF,name):
    path= directory(gamma_G,gamma_TF)+'/dic-' + name + '.pkl'
    with open(path, 'rb') as f:
        return pickle.load(f)
    print("Couplings loaded from"+path)

def main():
    
    parser = argparse.ArgumentParser(
        description='Simulation of magnetisation for spin glass. Many replica considered. Use the argument --create_graph if you do not want to load the matrix of couplings from the dictionary. Data will be saved in the data folder. Read the Readme.md for more info ')
    parser.add_argument("-N1", help="Number of genes", type=int, const=200000, default=200000, nargs='?')
    parser.add_argument("-N2", help="Number of TFs", type=int, const=200000, default=200000, nargs='?')
    #parser.add_argument("-N2", help="Number of TF", type=int, const=1000, default=1000, nargs='?')
    parser.add_argument('--gamma_G', type=float, default=1.81, help=" gamma in degree of genes. Default set to 3")
    parser.add_argument('--gamma_TF', type=float, default=1.81, help="gamma in degree of TFs. Default set to 4")
    parser.add_argument('--create_graph', help='Create a new graph and does not load ', action='store_true')
    args = parser.parse_args()
    N1,N2, gamma_G,gamma_TF = load_input(args)
    bias = 0.379
    if args.create_graph == False:
        try:
            dic = load_obj(gamma_G,gamma_TF,"couplings")
            print("I am loading the graph from dictionary")
            gamma_G = dic["gamma_G"]
            gamma_TF = dic["gamma_TF"]
            N1 = dic["N1"]
            N2 = dic["N2"]
            R = dic["R"]
            M = dic["M"]
        except FileNotFoundError:
            print('I did not find the topology file in "gamma_G:'+str(gamma_G)+'gamma_TF:'+str(gamma_TF)+'/dic-couplings.pkl')
            while True:
                answer = input("Do you want to create a new graph? Type y for yes, or q to quit\n")
                if answer == "q":
                    return 1
                elif answer == "y":
                    M,R = create_graph( N1,N2,gamma_G,gamma_TF,bias)
                    directory_ = directory(gamma_G,gamma_TF)
                    if not os.path.exists(directory_):
                         os.makedirs(directory_)
                         print("Folder didn't exist, create a  new directory  "+directory_)
                    dic = {"gamma_G": gamma_G,"gamma_TF": gamma_TF,  "N1": N1, "N2": N2,  "R": R,  "M": M}
                    with open(directory_+'/dic-couplings.pkl', 'wb') as f:
                             pickle.dump(dic, f, pickle.HIGHEST_PROTOCOL)
                    print("finish to write couplings")
                    break


    else:
        M, R = create_graph(N1, N2, gamma_G, gamma_TF)
    Ts = np.arange(0.01,1.,0.003)
    theta=0
    P_g = 0.5 * np.ones(N1)
    data = cavity_AND_parallel(P_g,Ts,R,M,theta,J0 = 1)
    dic = {"gamma_G": gamma_G,"gamma_TF": gamma_TF,  "N1": N1, "N2": N2,  "R": R, "M":M, "Ts": Ts,  "data": data}
    save_obj(dic, gamma_G,gamma_TF,theta)


if __name__ == '__main__':
    main()
