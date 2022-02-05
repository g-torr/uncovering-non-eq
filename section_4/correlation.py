import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import scipy
import sys

sys.path.insert(0, "lib")  # add the library folder to the path I look for modules
import argparse
import itertools
from multiprocessing import Pool
from datetime import datetime
import sys
import os
import pickle
sys.path.insert(0, "../lib")  # add the library folder to the path I look for modules
from correlation_cavity2 import correlation_parallel,correct_correlation
from dynamical_cavity import  cavity_caller # this script computes the dynamical cavity



def make_network( N,kin):
    G = nx.generators.degree_seq.directed_configuration_model([kin]*N, [kin]*N)
    G = nx.DiGraph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    J = nx.adjacency_matrix(G)
    sign_interaction = np.where(np.random.rand(J.nnz) > 0.5, 1, -1) /np.sqrt(kin) # bias in positive regulation
    J.data = sign_interaction
    return J

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
    P_sim, C_sim = zip(*data)
    del data
    C_sim = np.mean(C_sim, axis=0)
    P_sim = np.mean(P_sim, axis=0)
    return P_sim,C_sim

def dynamics_light_parallel(J, n_start, T, seed,N_iterations):
    local_state = np.random.RandomState(seed)
    N1 = J.shape[0]
    N_therm = 100
    np.where(np.random.rand(N1) > P_init, 1, 0)
    n = scipy.sparse.csr_matrix(n_start)
    t = 1
    while t < N_therm:
        z = local_state.logistic(0, T, (1, N1))
        # z=numpy.random.normal(0,2*T,(1,N1))
        a = (n * J).toarray() - z
        n = np.where(a > 0, 1, 0)
        n = scipy.sparse.csr_matrix(n)
        t += 1
    t = 0
    #N_iterations =max(int(np.log(1-alpha)/np.log(0.5*(1+np.tanh(1/2/T)))),1000)# number iterations grows at low temperature. See notes
    m = np.zeros(N1)
    C = np.zeros((N1,N1))
    time_average=0
    while t < N_iterations:
        z = local_state.logistic(0, T , (1, N1))
        # z=numpy.random.normal(0,2*T,(1,N1))
        a = (n * J).toarray() - z
        n = np.squeeze(np.where(a > 0, 1, 0))
        m += n
        x = 2*n-1
        C += np.outer(x,x)
        time_average+=1
        n = scipy.sparse.csr_matrix(n)
        t += 1

    return m / time_average, C/time_average

def replics_gpu(J, P_init, T, N_replics,N_iterations):
    N = J.shape[0]
    initial_states = cp.where(cp.random.rand(N_replics,N,dtype=cp.float32) > P_init, 1, 0)
    data = itertools.starmap(dynamics_gpu, itertools.product([J], initial_states, [T], [N_iterations]))
    # for replica in range(N_replic):
    #        data+=[dynamics_light(J,psi_init,T)]
    P_sim, C_sim = zip(*data)
    C_sim = cp.mean(cp.asarray(C_sim), axis=0)
    P_sim = cp.mean(cp.asarray(P_sim), axis=0)
    return cp.asnumpy(P_sim),cp.asnumpy(C_sim)

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
        n = cp.where(a > 0, 1, 0.)[0]
        t += 1
    t = 0
    #N_iterati%load_ext line_profilerons =max(int(np.log(1-alpha)/np.log(0.5*(1+np.tanh(1/2/T)))),1000)# number iterations grows at low temperature. See notes
    m = cp.zeros(N1)
    C = cp.zeros((N1,N1),dtype=cp.float32)
    time_average=0
    indx = np.triu_indices(N1,k = 1)
    while t < N_iterations:
        z = cp.random.logistic(0, T , (1, N1),dtype=cp.float32)
        # z=numpy.random.normal(0,2*T,(1,N1))
        a = n * J - z
        n = cp.where(a > 0, 1, 0.)[0]
        m += n
        x =2*n-1
        C += cp.outer(x,x)
        time_average+=1
        t += 1

    return m / time_average, C[indx]/time_average




def load_input(args):
    N = args.N  # number genes
    kin = args.kin
    return N,  kin

def load_obj(kin,name):
    path ='.'+ os.path.dirname(__file__)
    name='kin='+str(kin)+'/dic-' + name + '.pkl'
    print(path+'/'+name)
    with open(path+'/'+name, 'rb') as f:
        return pickle.load(f)
    print("Couplings loaded from"+path)


def save_obj(obj,kin,T):
    path = os.path.dirname(__file__)
    directory=path+'/kin='+str(kin)
    if not os.path.exists(directory+"/data"):
          os.makedirs(directory+"/data")
    name='T=' + str(T) +'.pkl'
    with open(directory+'/data/dic-' + name , 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)



def make_matrix(N,  corr):
    row, column = zip(*itertools.combinations(range(N), 2))
    rows = list(itertools.chain.from_iterable([row, np.arange(N), column]))
    columns = list(itertools.chain.from_iterable([column, np.arange(N), row]))
    data = list(itertools.chain.from_iterable([corr, np.ones(N), corr]))
    return scipy.sparse.coo_matrix((data, (rows, columns)), shape=(N, N)).toarray()


def main():
    startTime = datetime.now()
    parser = argparse.ArgumentParser(
        description='Simulation of magnetisation for spin glass. Many replica considered. Use the argument --create_graph if you do not want to load the matrix of couplings from the dictionary. Data will be saved in the data folder. Read the Readme.md for more info ')
    parser.add_argument("-N", help="Number of genes", type=int, const=1000, default=1000, nargs='?')
    #parser.add_argument("-N2", help="Number of TF", type=int, const=1000, default=1000, nargs='?')
    parser.add_argument('--kin', type=int, default=3, help=" in degree of genes. Default set to 3")
    parser.add_argument('-T', type=float, help=" Temperature")
    #parser.add_argument('--theta', type=float, default=0., help="theta. Default set to 0")
    parser.add_argument('--alpha', type=float, default= 0.9, help="confidence level that simulation last long enough to allow noise to cause spin flip. Meaningful in the range [0.5,1[, the closer to 1 the better, however simulation will take longer (expecially at low T). Default set to 0.9")
    parser.add_argument('--Nreplic', type=int, default=100, help=" Number of replicas. Default set to 500")
    parser.add_argument('--create_graph', help='Create a new graph and does not load ', action='store_true')
    #parser.add_argument('--eta', type=float, default=0.5, help="eta. Probability of positive couplings /all lins. Default set to 0.5")
    parser.add_argument('--nprocess', type=int, const=-1,default=-1,nargs='?', help="number of processes run in parallel, i.e. number of cores to be used in your local machine. Default all cores available")
    parser.add_argument('--parallel_corr',action="store_true",default=False)
    parser.add_argument('--no_gpu',action="store_true",default=False)
    args = parser.parse_args()
    N, kin = load_input(args)
    if args.create_graph == False:
        try:
            dic = load_obj(kin,"couplings")
            print("I am loading the graph from dictionary")
            kin = dic["kin"]
            N = dic["N"]
            J = dic["J"]
        except FileNotFoundError:
            print('I did not find the topology file in "kin='+str(kin)+'/dic-couplings.pkl')
            while True:
                answer = input("Do you want to create a new graph? Type y for yes, or q to quit\n")
                if answer == "q":
                    return 1
                elif answer == "y":
                    J = make_network(N, kin)
                    directory='kin='+str(kin)
                    if not os.path.exists(directory):
                         os.makedirs(directory)
                         print("Folder didn't exist, create a  new directory  "+directory)
                    dic = {"kin": kin,  "N": N,  "J": J}
                    with open(directory+'/dic-couplings.pkl', 'wb') as f:
                             pickle.dump(dic, f, pickle.HIGHEST_PROTOCOL)
                    print("finish to write couplings")
                    break


    else:
        J = make_network(N, kin)
    T = args.T
    if T is None:
        raise ValueError(' Specify the argument -T')
    #theta = args.theta
    no_gpu = args.no_gpu
    N_replics = args.Nreplic
    threads = args.nprocess
    N_iterations = 30000
    N = J.shape[0]
    try :
        import cupy as cp #if not 
        from cupyx.scipy.sparse import csr_matrix as csr_gpu
    except ImportError:
        print('GPU mode not available')
        no_gpu = True
    if no_gpu:
        P_sim,C_sim = replics_parallel(J, np.random.rand(N), T, N_replics, N_iterations,threads)
    else:
        P_sim,C_sim = replics_gpu(csr_gpu(J), cp.random.rand(N), T, N_replics, N_iterations)
    P_cav = cavity_caller(J,T,0,1e-6)
    C_cav = correlation_parallel(P_cav, T, J,args.parallel_corr)
    new_C_cav = correct_correlation(C_cav,P_cav,J,T)
    dic = {"kin": kin,  "N": N,  "J": J, "T": T, "P_sim": P_sim,"C_sim": C_sim, "P_cav": P_cav, 'C_cav':C_cav[np.triu_indices(N,k = 1)],'new_C_cav':new_C_cav[np.triu_indices(N,k = 1)],
           "N_replics": N_replics,'N_iterations':N_iterations}
    save_obj(dic, kin,T)
    print('code takes ',datetime.now() - startTime)

if __name__ == '__main__':
    main()
