from functools import lru_cache
import pickle
import scipy
import numpy as np
import networkx as nx
import cupy as cp
from cupyx.scipy.sparse import csr_matrix as csr_gpu
import itertools
import os
import sys
sys.path.insert(0, "../../lib") 
import simulation

def save_obj(obj,kin,kind):
    directory='kin_'+str(kin)
    if not os.path.exists(directory+"/data"):
          os.makedirs(directory+"/data")
    name=  kind +'.pkl'
    '''
    if os.path.isfile(directory+'/data/dic-'+name):
        name = name[:-4]+'_'+ str(time.time())+'.pkl'
    '''
    with open(directory+'/data/dic-' + name , 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def make_network(N,k,kind):
    '''generate a random regular network'''
    #sequence =  generate_degree_seq(gamma,N)
    sequence = np.ones(N,dtype=int)*k
    if np.sum(sequence)%2==1:
        sequence[-1]=sequence[-1]+1
    #make oriented network
    G = nx.generators.degree_seq.configuration_model(sequence)
    G.remove_edges_from(nx.selfloop_edges(G))
    J = nx.adjacency_matrix(G)
    row = J.tocoo().row
    col = J.tocoo().col
    cond = row>col
    interaction = np.where(np.random.rand(np.count_nonzero(cond))>.5,1.,-1.)
    A = scipy.sparse.csr_matrix((interaction,(row[cond],col[cond])),shape = (N,N))
    return assign_bidirectional_symmetry(A.tolil(),kind)
def assign_bidirectional_symmetry(J,kind):
    '''
    return interaction matrix with chosed symmetry of interaction sign'''
    triang = scipy.sparse.tril(J)
    a,b = triang.nonzero()
    if kind == 'symmetric':
        J[b,a] = J[a,b]
    elif kind == 'antisymmetric':
        J[b,a] = -J[a,b]
    elif kind == 'asymmetric':
        J[b,a] = np.where(np.random.rand(len(b))>0.5,1,-1)
    else:
        raise ValueError("kind should be one of those: asymmetric, antisymmetric  symmetric. Instead you select: "+ kind)
    J=J
    return J.tocsr()    
     
def make_epsilon(J):
    N = J.shape[0]
    row = J.tocoo().row
    col = J.tocoo().col
    bi_link = set(zip(row,col))&set(zip(col,row))#set of bidirectional link
    if len(bi_link)==0:
        return scipy.sparse.coo_matrix((N,N))
    a,b = zip(*bi_link)
    return scipy.sparse.coo_matrix((np.ones(len(a),dtype=bool),(np.array(a),np.array(b))),shape = (N,N)).tocsc()

def cavity(P,inter,T,theta):
    @lru_cache(maxsize=None)
    def recursion(bias, l):
        if (l == K):
            bias = (bias + theta) #/ np.sqrt(avg_degree)
            return np.tanh(bias / 2 / T)

        include = P[l] * recursion(bias + inter[l], l + 1)  # include node l with prob. P[j[l]]
        exclude = (1 - P[l]) * recursion(bias, l + 1)  # ignore node l
        return include + exclude
    bias = 0
    K = len(inter)
    P_new = 0.5 + 0.5 * recursion(bias, 0)
    recursion.cache_clear()
    return P_new
def cavity_C2(J,T,max_iter = 10,precision = 1e-4):
    def probability_upd(i,cond,theta,P_t_2,P_A,P_B):
        cav_neigh = np.array(js[i])[cond]

        A = cavity(P_A[cav_neigh,i].toarray(),inter = np.array(interaction[i])[cond],T = T,theta=theta)
        P_B_temp = np.where(epsilon[cav_neigh,i].toarray(),P_B[cav_neigh,i].toarray(),P_A[cav_neigh,i].toarray())
        #selects either P_A[j,i] or P_B[j,i] depending on epsilon[j,i]=1
        B = cavity(P_B_temp,inter = np.array(interaction[i])[cond],T = T,theta=theta)
        return (1-P_t_2[i])*A+P_t_2[i]*B
    J_transpose = J.transpose()


    #create network properties
    epsilon = make_epsilon(J)
    js = J_transpose.tolil().rows
    interaction = J_transpose.tolil().data
    #R = epsilon.multiply(J_transpose)
    row = J.tocoo().row#link to 
    col = J.tocoo().col#link from
    #data= J_transpose.tocoo().data#interaction strenght from col to row
    Ks = np.array([len(neigh) for neigh in js])  # in degree of each gene
    avg_degree = np.mean(Ks)



    trj = [] 
    P_A =J.copy()
    P_B= J.copy()
    P_A.data =np.ones(J.nnz)#np.random.rand(J.nnz)#prob of node activation
    P_B.data =np.ones(J.nnz)#np.random.rand(J.nnz)
    P_t_1 = np.random.rand(J.shape[0])#P^{t-1}
    P_t_2 = np.random.rand(J.shape[0])#P^{t-2}
    P_t = np.zeros(J.shape[0])
    P_A_new = P_A.copy()
    P_B_new = P_B.copy()
    #print(len(list(zip(row,col))))
    for t in range(max_iter):
        # here it is a time step of marginal and cavity
        for i in range(J.shape[0]):
            cond = np.array([True]*len(js[i]))#this is for non-cavity probabilities, predecessors. of i 
            P_t[i] = probability_upd(i,cond,0,P_t_2,P_A,P_B)

        for i,l in zip(row,col):

            cond = (np.array(js[i])!=l)#this is for cavity probabilities,# predecessors. of i /{l}
            P_A_new[i,l] = probability_upd(i,cond,0,P_t_2,P_A,P_B)
            theta = J[l,i]*epsilon[i,l]
            P_B_new[i,l] = probability_upd(i,cond,theta,P_t_2,P_A,P_B)# even if P_B depends on P_A, the index we care about different entries
            
        P_t_2 = P_t_1.copy()#progress time
        P_t_1 = P_t.copy()
        P_A = P_A_new.copy()
        P_B = P_B_new.copy()
        improvement = abs(P_t-P_t_2).max()
        if improvement<precision:
            print('ending after ',t,' iterations')
            break

    return P_A,P_B,P_t
def cavity_C1(J,T,max_iter = 10,precision = 1e-4):
    def probability_upd(i,cond,theta,P_t_2,P_A,P_B):
        cav_neigh = np.array(js[i])[cond]

        A = cavity(P_A[cav_neigh,i].toarray(),inter = np.array(interaction[i])[cond],T = T,theta=theta)
        P_B_temp = np.where(epsilon[cav_neigh,i].toarray(),P_B[cav_neigh,i].toarray(),P_A[cav_neigh,i].toarray())
        #selects either P_A[j,i] or P_B[j,i] depending on epsilon[j,i]=1
        B = cavity(P_B_temp,inter = np.array(interaction[i])[cond],T = T,theta=theta)
        return (1-P_t_2)*A+P_t_2*B
    J_transpose = J.transpose()


    #create network properties
    epsilon = make_epsilon(J)
    js = J_transpose.tolil().rows
    interaction = J_transpose.tolil().data
    #R = epsilon.multiply(J_transpose)
    row = J.tocoo().row#link to 
    col = J.tocoo().col#link from
    #data= J_transpose.tocoo().data#interaction strenght from col to row
    Ks = np.array([len(neigh) for neigh in js])  # in degree of each gene
    avg_degree = np.mean(Ks)



    trj = [] 


    P_A =J.copy()
    P_B= J.copy()
    P_A.data =np.ones(J.nnz)#np.random.rand(J.nnz)#prob of node activation
    P_B.data =np.ones(J.nnz)#np.random.rand(J.nnz)
    P_t_1 = np.random.rand(J.shape[0])#P^{t-1}
    P_t_2 = np.random.rand(J.shape[0])#P^{t-2}
    P_t = np.zeros(J.shape[0])
    P_A_new = P_A.copy()
    P_B_new = P_B.copy()
    #print(len(list(zip(row,col))))
    for t in range(max_iter):
        # here it is a time step of marginal and cavity
        for i in range(J.shape[0]):
            cond = np.array([True]*len(js[i]))#this is for non-cavity probabilities, predecessors. of i 
            P_t[i] = probability_upd(i,cond,0,P_t_2[i],P_A,P_B)

        for i,l in zip(row,col):

            cond = (np.array(js[i])!=l)#this is for cavity probabilities,# predecessors. of i /{l}
            P_A_new[i,l] = probability_upd(i,cond,0,P_A[i,l],P_A,P_B)
            theta = J[l,i]*epsilon[i,l]
            P_B_new[i,l] = probability_upd(i,cond,theta,P_B[i,l],P_A,P_B)# even if P_B depends on P_A, the index we care about different entries
            
        P_t_2 = P_t_1.copy()#progress time
        P_t_1 = P_t.copy()
        P_A = P_A_new.copy()
        P_B = P_B_new.copy()      
        improvement = abs(P_t-P_t_2).max()
        if improvement<precision:
            print('ending after ',t,' iterations')
            break
    return P_A,P_B,P_t
def cavity_C3(J,T,max_iter = 10,precision = 1e-4):
    def probability_upd(i,cond,theta,P_t_2,P_A,P_B):
        cav_neigh = np.array(js[i])[cond]

        A = cavity(P_A[cav_neigh,i].toarray(),inter = np.array(interaction[i])[cond],T = T,theta=theta)
        P_B_temp = np.where(epsilon[cav_neigh,i].toarray(),P_B[cav_neigh,i].toarray(),P_A[cav_neigh,i].toarray())
        #selects either P_A[j,i] or P_B[j,i] depending on epsilon[j,i]=1
        B = cavity(P_B_temp,inter = np.array(interaction[i])[cond],T = T,theta=theta)
        return (1-P_t_2)*A+P_t_2*B
    J_transpose = J.transpose()


    #create network properties
    epsilon = make_epsilon(J)
    js = J_transpose.tolil().rows
    interaction = J_transpose.tolil().data
    #R = epsilon.multiply(J_transpose)

    row = J.tocoo().row#link to 
    col = J.tocoo().col#link from
    #data= J_transpose.tocoo().data#interaction strenght from col to row
    Ks = np.array([len(neigh) for neigh in js])  # in degree of each gene
    avg_degree = np.mean(Ks)

    P_A_2 =J.copy()
    P_B_2= J.copy()
    P_A_2.data =np.ones(J.nnz)#np.random.rand(J.nnz)#prob of node activation
    P_B_2.data =np.ones(J.nnz)#np.random.rand(J.nnz)
    P_A_1 =J.copy()
    P_B_1= J.copy()
    P_A_1.data =np.ones(J.nnz)#np.random.rand(J.nnz)#prob of node activation
    P_B_1.data =np.ones(J.nnz)#np.random.rand(J.nnz)
    P_t_1 = np.random.rand(J.shape[0])#P^{t-1}
    P_t_2 = np.random.rand(J.shape[0])#P^{t-2}
    P_t_3 = np.random.rand(J.shape[0])#P^{t-3}
    P_t = np.zeros(J.shape[0])
    P_A = P_A_1.copy()
    P_B = P_B_1.copy()
    #print(len(list(zip(row,col))))
    for t in range(max_iter):
        # here it is a time step of marginal and cavity
        for i in range(J.shape[0]):
            cond = np.array([True]*len(js[i]))#this is for non-cavity probabilities, predecessors. of i 
            P_t[i] = probability_upd(i,cond,0,P_t_2[i],P_A_1,P_B_1)

        for i,l in zip(row,col):
            P_t_2_cav = P_A_2[i,l]*(1-P_t_3[l])+P_B_2[i,l]*P_t_3[l]
            cond = (np.array(js[i])!=l)#this is for cavity probabilities,# predecessors. of i /{l}
            P_A[i,l] = probability_upd(i,cond,0,P_t_2_cav,P_A_1,P_B_1)
            theta = J[l,i]*epsilon[i,l]
            P_B[i,l] = probability_upd(i,cond,theta,P_t_2_cav,P_A_1,P_B_1)# even if P_B depends on P_A, the index we care about different entries
            
        P_t_3 = P_t_2.copy()#progress time
        P_t_2 = P_t_1.copy()#progress time
        P_t_1 = P_t.copy()
        P_A_2 = P_A_1.copy()
        P_B_2 = P_B_1.copy()        
        P_A_1 = P_A.copy()
        P_B_1 = P_B.copy() 
        improvement = abs(P_t-P_t_2).max()
        if improvement<precision:
            print('ending after ',t,' iterations')
            break
    return P_A,P_B,P_t



def main():
    N = 10000
    k = 3
    T = 1
    kind = 'asymmetric'
    J = make_network(N,k,kind)
    P_A,P_B,P_t_C1 = cavity_C1(J,T,150)
    P_A,P_B,P_t_C2 = cavity_C2(J,T,150)
    P_A,P_B,P_t_C3 = cavity_C3(J,T,150)
    N_replics = 50
    N_iterations = 1e5
    no_gpu = False
    if no_gpu:
        threads = -1 #number of process to be run in parallel, negative if all cores available
        sim = simulation.replics_parallel(J, np.random.rand(J.shape[0]), T, N_replics,N_iterations,threads)
    else:
        sim =  simulation.replics_gpu(csr_gpu(J), cp.random.rand(J.shape[0]), T, N_replics,N_iterations)


    P_t_MC = np.mean(sim,axis = 0)
    dic = {'J':J,'P_t_C1':P_t_C1,'P_t_C2':P_t_C2,'P_t_C3':P_t_C3,'P_t_MC':P_t_MC,'T':T,'N_iterations':N_iterations,'N_replics':N_replics,'J':J}
    save_obj(dic,k,kind)

if __name__ == '__main__':
    main()
