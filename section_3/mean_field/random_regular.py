'''Compute the stationary node activation probability for a graph with fat tailed distribution using dynamical cavity'''
import numpy as np
import pickle
import networkx as nx
import itertools
import sys
import argparse
from argparse import RawTextHelpFormatter
import os
import time

sys.path.insert(0, "../../lib")  # add the library folder to the path I look for modules
from dynamical_cavity import cavity
from mean_field import mean_field

def directory(kin):
    path = '.'+os.path.dirname(__file__)
    return path+'/kin='+str(kin)

def save_obj(obj,theta,eta,kin):
    name='theta_'+str(theta)+'_eta_'+str(eta)+'.pkl'
    if not os.path.exists(directory(kin)+"/data"):
          os.makedirs(directory(kin)+"/data")
    if os.path.isfile(directory(kin)+'/data/dic-' + name ):
        name = name[:-4]+'_'+ str(time.time())+'.pkl'
    with open(directory(kin)+'/data/dic-' + name , 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(kin,name):
    path=directory(kin)+'/dic-' + name + '.pkl'
    with open(path, 'rb') as f:
        return pickle.load(f)
    print("Couplings loaded from"+path)


def make_network(N, kin,bias):
    def neighbouring(A):
        interaction = []
        js = []
        Ks = []
        for l, u in zip(A.indptr[:-1], A.indptr[1:]):
            js += [A.indices[l:u]]
            interaction += [A.data[l:u] / np.sqrt(u - l)]
            Ks += [u - l]
        return js, interaction, Ks

    seq = [kin]*N
    G = nx.generators.degree_seq.directed_configuration_model(seq, np.random.permutation(seq))
    G = nx.DiGraph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    J = nx.adjacency_matrix(G)
    sign_interaction = np.where(np.random.rand(J.nnz) > bias, 1, -1)  # bias in positive regulation
    J.data = np.ravel(sign_interaction)  # add negative element
    return J


def compute_mean_magn(J,T,theta):
    J_transpose = J.transpose().tolil()
    js = J_transpose.rows  # list of list, structure is [el[i]] where el[i]
    # is the list of  predecessors of gene i ( the index)
    interaction = J_transpose.data  # list of list, structure is [el[i]]
    # where el[i] is the list of  predecessors of gene i (interaction strength with sign)
    Ks = np.array([len(neigh) for neigh in js])  # in degree of each gene
    N = J.shape[0]
    J0 = 1/np.sqrt(Ks.mean())
    P_mean = mean_field(np.random.rand(N), js, T, interaction, N, Ks, theta,J0,max_iter=100)
    P_cavity = cavity(np.random.rand(N), js, T, interaction, N, Ks, theta,J0)
    return np.mean(P_mean), np.mean(P_cavity)
def main():
    parser = argparse.ArgumentParser(
        description='Compute the stationary  node activation probability for a graph with fat tailed distribution using dynamical programming.\n'
                    '\n'
                    'Returns:\n'
                    'a dictionary containing the topology "J", and the activation probability "data". \n'
                    '"J" is a scipy.sparse matrix.\n'
                    '"data" is a 2d list containing single node activation probabilities at different noise parameters T.\n'
                    'Output is saved in /data/ folder with unique identifier. Simulation for different values of T are run in parallel. By default, code runs on  all cores available on your machine.',formatter_class=RawTextHelpFormatter)
    parser.add_argument("-N", help="Number of nodes", type=int, const=50000, default=50000, nargs='?')
    #parser.add_argument('--theta', type=float, default=0., help="theta. Default set to 0")
    #parser.add_argument('--eta', type=float, default=0.5, help="eta. Probability of positive couplings /all lins. Default set to 0.5")
    parser.add_argument('--nprocess', type=int, const=-1,default=-1,nargs='?', help="number of processes run in parallel, i.e. number of cores to be used in your local machine. Default all cores available")
    parser.add_argument('--T', type = float,  default = 0.2,help = "[Tmin,Tmax,dT]. Simulation investigates noise parameter values: np.arange(Tmin,Tmax,dt). Default [0.05,1.1,0.05] ")
    args = parser.parse_args()
    N = args.N
    #theta = args.theta
    #eta = args.eta
    threads = args.nprocess
    kin = 3
    try:
        dic = load_obj(kin, "couplings")
        print("I am loading the graph from dictionary")
        J = dic["J"]
        kin = dic["kin"]
    except FileNotFoundError:
        print('I did not find the topology file in ' + directory(kin) + '/dic-couplings.pkl')
        while True:
            answer = input("Do you want to create a new graph? Type y for yes, or q to quit\n")
            if answer == "q":
                return 1
            elif answer == "y":
                J = make_network(N, kin, 1)

                if not os.path.exists(directory(kin)):
                    os.makedirs(directory(kin))
                    print("Folder didn't exist, create a  new directory  " + directory(kin))
                dic = {"kin": kin, "J": J}
                with open(directory(kin) + '/dic-couplings.pkl', 'wb') as f:
                    pickle.dump(dic, f, pickle.HIGHEST_PROTOCOL)
                print("finish to write couplings")
                break

    T = args.T
    N = J.shape[0]
    Ks = np.diff(J.indptr)
    J0 = 1/np.sqrt(np.mean(Ks))
    max_outdegree = max(Ks)
    max_recursions = int((max_outdegree+1)*(max_outdegree+2)/2)
    if max_recursions> sys.getrecursionlimit():
        print("Warning! maximum degree larger than default recursion limit, I 'll update recursion limit to", max_recursions )
        sys.setrecursionlimit(max_recursions)
    print('Network done')
    for theta in [0.,0.5]:
        for eta in [0.1,0.3,0.5,0.7,0.9]:
            J.data = np.where(np.random.rand(len(J.data))>eta,1,-1)
            mean_mean , mean_cav = compute_mean_magn(J,T * J0,theta*J0)
            dic = {'mean_mean':mean_mean,'mean_cav':mean_cav,  'T': T, 'theta':theta,'eta': eta}
            save_obj(dic,theta,eta,kin)


if __name__ == '__main__':
    main()
