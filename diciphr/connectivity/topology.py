from connmat_utils import density, degree, nodestrength, normalize_mat, binarize_mat, is_binary
from diciphr.utils import DiciphrException
import numpy as np
import bct 

#######################################
### BCT Graph Theoretical Functions ###
#######################################
  
def assortativity_bin(connmat):
    return bct.assortativity_bin(binarize_mat(connmat))

def assortativity_wei(connmat):
    if is_binary(connmat):
        raise DiciphrException("Binary input to weighted measure")
    return bct.assortativity_wei(normalize_mat(connmat))

def efficiency_bin(connmat, local=False):
    return bct.efficiency_bin(binarize_mat(connmat), local=local)
    
def efficiency_wei(connmat, local=False):
    if is_binary(connmat):
        raise DiciphrException("Binary input to weighted measure")
    return bct.efficiency_wei(normalize_mat(connmat), local=local)
    
def transitivity_bin(connmat):
    return bct.transitivity_bu(binarize_mat(connmat))

def transitivity_wei(connmat):
    if is_binary(connmat):
        raise DiciphrException("Binary input to weighted measure")
    return bct.transitivity_wu(normalize_mat(connmat))
    
def pathlength_wei(connmat):
    if is_binary(connmat):
        raise DiciphrException("Binary input to weighted measure")
    try:
        ret = bct.charpath(bct.distance_wei(np.linalg.inv(normalize_mat(connmat)))[0])[0]
    except:
        ret = np.nan
    return ret

def pathlength_bin(connmat):
    try:
        ret = bct.charpath(bct.distance_bin(np.linalg.inv(binarize_mat(connmat)))[0])[0]
    except:
        ret = np.nan
    return ret
    
def modularity_louvain_wei(connmat):
    if is_binary(connmat):
        raise DiciphrException("Binary input to weighted measure")
    return np.max([bct.modularity_louvain_und(normalize_mat(connmat))[1] for _iter in range(100)])
    
def modularity_louvain_bin(connmat):
    # check that connmat is wtd
    connmat=binarize_mat(connmat)
    return np.max([bct.modularity_louvain_und(connmat)[1] for _iter in range(100)])

def betweenness_bin(connmat):
    n = len(connmat)
    return bct.betweenness_bin(binarize_mat(connmat))/((n-1)*(n-2))
    
def betweenness_wei(connmat):
    '''
    Node betweenness centrality 
    '''
    G = bct.distance_wei(normalize_mat(connmat))[0]
    n = len(G)
    BC = np.zeros((n,))  # vertex betweenness

    for u in range(n):
        D = np.tile(np.inf, (n,))
        D[u] = 0  # distance from u
        NP = np.zeros((n,))
        NP[u] = 1  # number of paths from u
        S = np.ones((n,), dtype=bool)  # distance permanence
        P = np.zeros((n, n))  # predecessors    
        Q = np.zeros((n,))
        q = n - 1  # order of non-increasing distance

        G1 = G.copy()
        V = [u]
        while True:
            S[V] = 0  # distance u->V is now permanent
            G1[:, V] = 0  # no in-edges as already shortest
            for v in V:
                Q[q] = v
                q -= 1
                W = np.where(G1[v, :])[0]  # neighbors of v
                for w in W:
                    Duw = D[v] + G1[v, w]  # path length to be tested
                    if Duw < D[w]:  # if new u->w shorter than old
                        D[w] = Duw
                        NP[w] = NP[v]  # NP(u->w) = NP of new path
                        P[w, :] = 0
                        P[w, v] = 1  # v is the only predecessor
                    elif Duw == D[w]:  # if new u->w equal to old
                        NP[w] += NP[v]  # NP(u->w) sum of old and new
                        P[w, v] = 1  # v is also predecessor

            if D[S].size == 0:
                break  # all nodes were reached
            if np.isinf(np.min(D[S])):  # some nodes cannot be reached
                Q[:q + 1] = np.where(np.isinf(D))[0]  # these are first in line
                break
            V = np.where(D == np.min(D[S]))[0]

        DP = np.zeros((n,))
        Q = Q.astype(np.int32)
        for w in Q[:n - 1]:
            BC[w] += DP[w]
            for v in np.where(P[w, :])[0]:
                DP[v] += (1 + DP[w]) * NP[v] / NP[w]

    return BC/((n-1)*(n-2))
