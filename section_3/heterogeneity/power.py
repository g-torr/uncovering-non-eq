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
from utilities import make_network
sys.path.insert(0, "../../lib")  # add the library folder to the path I look for modules
from dynamical_cavity import cavity_parallel

def save_obj(obj,theta):
    path = '.'+os.path.dirname(__file__)
    name='theta_'+str(theta)+'.pkl'
    if not os.path.exists(path+"/data"):
        os.makedirs(path+"/data")
    if os.path.isfile(path+'/data/dic-' + name ):
        name = name[:-4]+'_'+ str(time.time())+'.pkl'
    with open(path+'/data/dic-' + name , 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)



def main():
    parser = argparse.ArgumentParser(
        description='Compute the stationary  node activation probability for a graph with fat tailed distribution using dynamical programming.\n'
                    '\n'
                    'Returns:\n'
                    'a dictionary containing the topology "J", and the activation probability "data". \n'
                    '"J" is a scipy.sparse matrix.\n'
                    '"data" is a 2d list containing single node activation probabilities at different noise parameters T.\n'
                    'Output is saved in /data/ folder with unique identifier. Simulation for different values of T are run in parallel. By default, code runs on  all cores available on your machine.',formatter_class=RawTextHelpFormatter)
    parser.add_argument("-N", help="Number of nodes", type=int, const=200000, default=200000, nargs='?')
    parser.add_argument('--theta', type=float, default=0., help="theta. Default set to 0")
    parser.add_argument('--nprocess', type=int, const=-1,default=-1,nargs='?', help="number of processes run in parallel, i.e. number of cores to be used in your local machine. Default all cores available")
    parser.add_argument('--Ts', type = float, nargs = '*', default = [0.05, 1.1, 0.01],help = "[Tmin,Tmax,dT]. Simulation investigates noise parameter values: np.arange(Tmin,Tmax,dt). Default [0.05,1.1,0.05] ")
    args = parser.parse_args()
    N = args.N
    theta = args.theta
    threads = args.nprocess
    gamma = 1.81
    bias = 0.379
    Ts = np.arange((args.Ts)[0], (args.Ts)[1], (args.Ts)[2])

    J = make_network(N,gamma,bias)
    print('Network done')

    data = cavity_parallel([0.5]*N,Ts,J,theta,threads)# run in parallel at different temperatures
    dic = {'data': data, 'J': J, 'Ts': Ts, 'theta': theta}
    save_obj(dic,theta)


if __name__ == '__main__':
    main()
