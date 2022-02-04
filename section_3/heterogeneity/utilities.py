import networkx as nx
import numpy as np
from numba import jit
def generate_degree_seq(gamma, N):
    kseq = np.ceil( np.random.pareto(gamma, N))
    cond = kseq > N
    while any(cond):
        temp_seq = np.ceil( np.random.pareto(gamma, np.count_nonzero(cond)))
        kseq[cond] = temp_seq
        cond = kseq > N
    return np.array(kseq, dtype=int)
def make_network(N, gamma,bias):
    seq = generate_degree_seq(gamma, N)
    G = nx.generators.degree_seq.directed_configuration_model(seq, np.random.permutation(seq))
    G = nx.DiGraph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    J = nx.adjacency_matrix(G)
    sign_interaction = np.where(np.random.rand(J.nnz) > bias, 1, -1)  # bias in positive regulation
    J.data = np.ravel(sign_interaction)  # add negative element
    return J


def cavity_trj(J,T,theta,precision=1e-4,J0 = 'auto'):
    J = J.copy()
    J.data = np.where(J.data > 0, 1, -1)
    J_transpose = J.transpose().tolil()
    js = J_transpose.rows  # list of list, structure is [el[i]] where el[i]
    # is the list of  predecessors of gene i ( the index)
    interaction = J_transpose.data  # list of list, structure is [el[i]]
    # where el[i] is the list of  predecessors of gene i (interaction strength with sign)
    Ks = np.array([len(neigh) for neigh in js])  # in degree of each gene
    max_outdegree = max(Ks)
    max_recursions = int((max_outdegree + 1) * (max_outdegree + 2) / 2)
    '''
    if max_recursions > sys.getrecursionlimit():
        print("Warning! maximum degree larger than default recursion limit, I 'll update recursion limit to",
              max_recursions)
        sys.setrecursionlimit(max_recursions)
    '''
    N = J.shape[0]
    if J0 =='auto':
        avg_degree = np.mean(Ks)
        J0 = 1/ np.sqrt(avg_degree)
    return cavity(np.random.rand(N), js, T, interaction, N, Ks, theta,J0,precision)

def cavity(P, js, T, interaction, N, Ks, theta,J0, precision=1e-4, max_iter=50):
    """
    This runs the dynamical cavity without recursive calls. It creates instead a matrix. This works only if couplings are in the form  \pm J.
    If couplings are in a different form, use it cavity_general
     It computes the node activation probability for a  directed network.
    :param P_init: list of floats of length N
    :param T: float
    :param js: list of list, structure is [el[i]] where el[i] is the list of  predecessors of gene i ( the index)
    :param interaction:  list of list, structure is [el[i] for i in range(N)]
            where el[i] is the list of  predecessors of gene i (interaction strength with sign)
    :param theta: float (in units of 1/sqrt(<K>))
    :param max_iter: int
    :param precision: float
    :return: P_new it is a  list of dimensions N which contains the probability of active state for each gene.
    ----NOTES------
    In order to help storing, couplings are taken to be +-1, at the end the local field is rescaled by 1/sqrt(<|J_{ij}|>)
    Even though code runs for any directed network, results  are exact for fully asymmetric networks only.
    """

    if T == 0:
        return cavity_zero_T(P, js, interaction, N, Ks, theta)

    avg_degree = np.mean(Ks)
    trj = []
    for count in range(max_iter):
        P_new = np.zeros(N)
        for i in range(N):
            j = js[i]
            bias = 0
            K = Ks[i]
            if K ==0:
                P_new[i]=0.5
            else:
                inter = interaction[i]
                P_new[i]=cavity_single_numba(P[j],np.array(inter),T,theta,K,J0)
        if max(np.abs(np.array(P) - np.array(P_new))) < precision:
            P = P_new
            #print('finishing after', count, 'iterations')
            break
        if count == max_iter:
            print("Maximum number of repetition reached, but target  precision has not been reached. Precision reached is "+str(max(np.abs(np.array(P) - np.array(P_new)))))

        P = np.array(P_new)
        trj +=[P]
    P = np.array(P)
    return P,trj
@jit(nopython=True)
def cavity_single_numba(P_loc,inter,T,theta,K,J0):
    '''
    Dynamic programming is run using iterative calls. It creates a matrix for this task
    '''
    pos = inter.copy()
    neg = inter.copy()
    pos[pos<0]=0#apply theta function
    neg[neg>0]=0
    m = np.zeros((K+1,K+1))

    h_tilde =np.arange(np.sum(neg),np.sum(pos)+1)
    for ind in range(len(h_tilde)):
        m[K,ind]=1/2*(1+np.tanh((h_tilde[ind]-theta)*J0/2/T))
    offset = np.sum(neg)#this is the offset to map \tilde{h} to index of m matrix
    for l,low,top in list(zip(np.arange(1,K),np.cumsum(neg)[:-1],np.cumsum(pos)[:-1]+1))[::-1]:
        #print(l)
        for h in np.arange(low-offset,top-offset):
            m[l,h] = P_loc[l]*m[l+1,h+inter[l]]+(1-P_loc[l])*m[l+1,h]

        #print(top)
    return P_loc[0]*m[1,inter[0]-offset]+(1-P_loc[0])*m[1,-offset]
