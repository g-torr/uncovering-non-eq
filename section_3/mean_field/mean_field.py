import numpy as np
def mean_field(P, js, T, interaction, N, Ks, theta,J0, precision=1e-4, max_iter=50):
    """
    Run the dynamical cavity with recursive calls.
    :param P_init: list of floats of length N
    :param T: float
    :param J: sparse.csr_matrix
    :param theta: float (in units of 1/sqrt(<K>))
    :param max_iter: int
    :param precision: float
    :return: P_new it is a  list of dimensions N which contains the probability of active state for each gene.
    In order to help storing, couplings are taken to be +-1, bias is then rescaled by 1/sqrt(<|J_{ij}|>)
    """
    avg_degree = np.mean(Ks)
    P_new = np.zeros(N)
    for count in range(max_iter):
        for i,(inter,j) in enumerate(zip(interaction,js)):
            P_new[i] = 0.5*(1+np.tanh((sum(inter*P[j])-theta)*J0/2/T))
        if max(np.abs(np.array(P) - np.array(P_new))) < precision:
            P = P_new
            print('finishing after', count, 'iterations')
            break
        if count == max_iter-1:
            print("Maximum number of repetition reached, but target  precision has not been reached. ")
        P = P_new.copy()

    return P
