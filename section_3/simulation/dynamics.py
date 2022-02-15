import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
import itertools
import warnings
import scipy
import numpy.random
import time
import argparse
from functools import lru_cache
import sys
import os
sys.path.insert(0, "../../lib")  # add the library folder to the path I look for modules
import pickle
import dynamical_cavity as cavity # this script computes the dynamical cavity
from generate_random_regular import *
from multiprocessing import Pool
import simulation

def generate_degree_seq(gamma, N):
    kseq = np.ceil( np.random.pareto(gamma, N))
    cond = kseq > N
    while any(cond):
        temp_seq = np.ceil( np.random.pareto(gamma, np.count_nonzero(cond)))
        kseq[cond] = temp_seq
        cond = kseq > N
    return np.array(kseq, dtype=int)
def make_network(N, gamma,bias=0.5):

    seq = generate_degree_seq(gamma, N)
    G = nx.generators.degree_seq.directed_configuration_model(seq, np.random.permutation(seq))
    G = nx.DiGraph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    J = nx.adjacency_matrix(G)
    A = J.tocsc()
    Ks = np.diff(A.indptr)  # in degree of each gene
    #if min(Ks)==0:
    #    raise ValueError('Network having isolated nodes')
    sign_interaction = np.where(np.random.rand(J.nnz) > bias, 1., -1.) #/np.sqrt(np.mean(Ks)) # bias in positive regulation
    J.data = np.ravel(sign_interaction)  # add negative element
    return J

def save_obj(obj,gamma,T,theta):
    directory='gamma_'+str(gamma)
    if not os.path.exists(directory+"/data"):
          os.makedirs(directory+"/data")
    name='T_' + str(T)  +'_theta'+str(theta) +'.pkl'
    with open(directory+'/data/dic-' + name , 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)





# --------This is the simulation------

def replics_parallel(J, P_init, T, N_replics,N_iterations,threads):
    if threads < 0:
        pool = Pool()
    else:
        pool = Pool(int(threads))
    '''Simulation at fixed T for different replicas
    Initial condition is chosen to be the same as for cavity'''
    different_seed = np.random.randint(1, 2 ** 32 - 1, N_replics)
    data = pool.starmap(dynamics_light_parallel, itertools.product([J], [P_init], [T], different_seed, [N_iterations]))
    # for replica in range(N_replic):
    #        data+=[dynamics_light(J,psi_init,T)]
    pool.close()
    return data


def dynamics_light_parallel(J, P_init, T,theta, seed,N_iterations):
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
        n = np.where(a > theta, 1, 0)
        n = scipy.sparse.csr_matrix(n)
        t += 1
    t = 0
    m = np.zeros(N)
    #C = zeros((N,N))
    while t < N_iterations:
        z = local_state.logistic(0, T , (1, N))
        # z=numpy.random.normal(0,2*T,(1,N))
        a = (n * J).toarray() - z
        n = np.where(a > theta, 1, 0)
        m += n[0]
        #C += outer(2*n[0]-1,2*n[0]-1)
        n = scipy.sparse.csr_matrix(n)
        t += 1

    return m / N_iterations  #, C/N_iterations


def replics_gpu(J, P_init, T, theta,N_replics,N_iterations):
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as csr_gpu
    def dynamics_gpu(J, n, T,N_iterations):
        if not (type(J) is csr_gpu):
            print('Coupling matrix should be of type'+str(csr_gpu)+', I try to convert')
            J = csr_gpu(J)
        #local_state = cp.random.seed(seed)
        N1 = J.shape[0]
        N_therm = 100
        t = 0
        while t < N_therm:
            z = cp.random.logistic(0, T , (1, N1))
            # z=numpy.random.normal(0,2*T,(1,N1))
            a = n * J - z
            n = cp.where(a > theta, 1, 0.)[0]
            t += 1
        t = 0
        #N_iterati%load_ext line_profilerons =max(int(np.log(1-alpha)/np.log(0.5*(1+np.tanh(1/2/T)))),1000)# number iterations grows at low temperature. See notes
        m = cp.zeros(N1)
        while t < N_iterations:
            z = cp.random.logistic(0, T , (1, N1),dtype=cp.float32)
            # z=numpy.random.normal(0,2*T,(1,N1))
            a = n * J - z
            n = cp.where(a > theta, 1, 0.)[0]
            m += n
            t += 1

        return m / t


    N = J.shape[0]
    initial_states = cp.where(cp.random.rand(N_replics,N,dtype=cp.float32) > P_init, 1, 0)
    P_sim = itertools.starmap(dynamics_gpu, itertools.product([J], initial_states, [T], [N_iterations]))
    #P_sim = cp.mean(cp.asarray(list(P_sim)), axis=0)
    return cp.asnumpy(cp.asarray(list(P_sim)))




def load_input(args):
    N = args.N  # number genes
    gamma = args.gamma
    return N,  gamma

def load_obj(gamma,name):
    path='gamma_'+str(gamma)+'/dic-' + name + '.pkl'
    with open(path, 'rb') as f:
        return pickle.load(f)
    print("Couplings loaded from"+path)

def main():
    parser = argparse.ArgumentParser(
        description='Simulation of magnetisation for spin glass. Many replica considered. Use the argument --create_graph if you do not want to load the matrix of couplings from the dictionary. Data will be saved in the data folder. Read the Readme.md for more info ')
    parser.add_argument("-N", help="Number of genes", type=int, const=1000, default=1000, nargs='?')
    #parser.add_argument("-N2", help="Number of TF", type=int, const=1000, default=1000, nargs='?')
    parser.add_argument('--gamma', type=float, default=3, help=" in degree of genes. Default set to 2")
    parser.add_argument('-T', type=float, help=" Temperature")
    parser.add_argument('--theta', type=float, default=0., help="theta. Default set to 0")
    parser.add_argument('--alpha', type=float, default= 0.9, help="confidence level that simulation last long enough to allow noise to cause spin flip. Meaningful in the range [0.5,1[, the closer to 1 the better, however simulation will take longer (expecially at low T). Default set to 0.9")
    parser.add_argument('--Nreplics', type=int, default=500, help=" Number of replicas. Default set to 500")
    parser.add_argument('--create_graph', help='Create a new graph and does not load ', action='store_true')
    #parser.add_argument('--eta', type=float, default=0.5, help="eta. Probability of positive couplings /all lins. Default set to 0.5")
    parser.add_argument('--nprocess', type=int, const=-1,default=-1,nargs='?', help="number of processes run in parallel, i.e. number of cores to be used in your local machine. Default all cores available")
    parser.add_argument('--no_gpu', help='Use Nvidia GPU through CUDA ', action='store_true')
    args = parser.parse_args()
    N, gamma = load_input(args)
    if args.create_graph == False:
        try:
            dic = load_obj(gamma,"couplings")
            print("I am loading the graph from dictionary")
            gamma = dic["gamma"]
            N = dic["N"]
            J = dic["J"]
        except FileNotFoundError:
            print('I did not find the topology file in "gamma_'+str(gamma)+'/dic-couplings.pkl')
            while True:
                answer = input("Do you want to create a new graph? Type y for yes, or q to quit\n")
                if answer == "q":
                    return 1
                elif answer == "y":
                    J = make_network(N, gamma)
                    directory='gamma_'+str(gamma)
                    if not os.path.exists(directory):
                         os.makedirs(directory)
                         print("Folder didn't exist, create a  new directory  "+directory)
                    dic = {"gamma": gamma,  "N": N,  "J": J}
                    with open(directory+'/dic-couplings.pkl', 'wb') as f:
                             pickle.dump(dic, f, pickle.HIGHEST_PROTOCOL)
                    print("finish to write couplings")
                    break


    else:
        J = make_network(N, gamma)
    T = args.T
    if T is None:
        raise ValueError(' Specify the argument -T')
    no_gpu = args.no_gpu
    theta = args.theta
    alpha=args.alpha
    N_replics = args.Nreplics
    threads = args.nprocess
    N_iterations =max(np.log(1-alpha)/np.log(0.5*(1+np.tanh(1/2/T))),100)# number iterations grows at low temperature. See notes
    N_iterations = min(N_iterations, 10000) # we avoid N_iterations to explode at low T
    try:
        import cupy as cp
        from cupyx.scipy.sparse import csr_matrix as csr_gpu
    except ImportError:
            no_gpu = True
    P_init = 0.5 * np.ones(N)
    start_time = time.time()
    if no_gpu:
        sim = simulation.replics_parallel(J, np.random.rand(J.shape[0]), T, N_replics,N_iterations,threads) 
    else:
        N_replics = 1
        N_iterations = 1e8
        sim =  simulation.replics_gpu(csr_gpu(J), cp.random.rand(J.shape[0]), T, N_replics,N_iterations)
    end_time = time.time()
    f = open("execution_time.txt", "w")
    f.write("Simulation takes \t"+str(end_time-start_time)+'\n')
    print("simulation finished, now run cavity" )
    start_time = time.time()
    magn_cavity=cavity.cavity_caller(J,T,theta,J0 = 1)
    end_time = time.time()
    f.write("Cavity takes \t"+str(end_time-start_time))
    f.close()
    dic = {"gamma": gamma,  "N": N,  "J": J, "T": T, "sim": sim, "theta":theta,"magn_cavity": magn_cavity,
           "N_replics": N_replics,'N_iterations':N_iterations,'execution_time':str(end_time-start_time)+' s'}
    '''while True:
        flag = input("save the results? [y/n]")
        if (flag == "y" or flag == "yes"):
            save_obj(dic, "states:cin" + str(cin)+"din:" + str(din))
            break
        elif (flag == "n" or flag == "not"):
            print("I do not save the result")
            break
        else:
            "I do not understand your answer, type yes or not"
    '''
    save_obj(dic, gamma,T,theta)


if __name__ == '__main__':
    main()
