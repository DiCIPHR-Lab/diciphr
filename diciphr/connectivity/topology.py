from diciphr.connectivity.connmat_utils import binarize_mat, is_binary
import logging
import numpy as np
import bct 

#######################################
### BCT Graph Theoretical Functions ###
#######################################
  
def assortativity_bin(connmat):
    logging.debug('diciphr.connectivity.topology.assortativity_bin')
    return bct.assortativity_bin(binarize_mat(connmat))

def assortativity_wei(connmat):
    logging.debug('diciphr.connectivity.topology.assortativity_wei')
    if is_binary(connmat):
        raise ValueError("Binary input to weighted measure")
    return bct.assortativity_wei(connmat)

def efficiency_bin(connmat, local=False):
    logging.debug('diciphr.connectivity.topology.efficiency_bin')
    return bct.efficiency_bin(binarize_mat(connmat), local=local)
    
def efficiency_wei(connmat, local=False):
    logging.debug('diciphr.connectivity.topology.efficiency_wei')
    if is_binary(connmat):
        raise ValueError("Binary input to weighted measure")
    return bct.efficiency_wei(connmat, local=local)
    
def transitivity_bin(connmat):
    logging.debug('diciphr.connectivity.topology.transitivity_bin')
    return bct.transitivity_bu(binarize_mat(connmat))

def transitivity_wei(connmat):
    logging.debug('diciphr.connectivity.topology.transitivity_wei')
    if is_binary(connmat):
        raise ValueError("Binary input to weighted measure")
    return bct.transitivity_wu(connmat)
    
def pathlength_wei(connmat):
    logging.debug('diciphr.connectivity.topology.pathlength_wei')
    if is_binary(connmat):
        raise ValueError("Binary input to weighted measure")
    try:
        L = bct.weight_conversion(connmat, 'lengths')
        D, _ = bct.distance_wei(L)
        ret = bct.charpath(D)[0]
    except:
        ret = np.nan
    return ret

def pathlength_bin(connmat):
    logging.debug('diciphr.connectivity.topology.pathlength_bin')
    try:
        ret = bct.charpath(bct.distance_bin(np.linalg.inv(binarize_mat(connmat)))[0])[0]
    except:
        ret = np.nan
    return ret
    
def modularity_louvain_wei(connmat):
    logging.debug('diciphr.connectivity.topology.modularity_louvain_wei')
    if is_binary(connmat):
        raise ValueError("Binary input to weighted measure")
    return np.max([bct.modularity_louvain_und(connmat)[1] for _iter in range(100)])
    
def modularity_louvain_bin(connmat):
    logging.debug('diciphr.connectivity.topology.modularity_louvain_bin')
    # check that connmat is wtd
    connmat=binarize_mat(connmat)
    return np.max([bct.modularity_louvain_und(connmat)[1] for _iter in range(100)])

def betweenness_bin(connmat):
    logging.debug('diciphr.connectivity.topology.betweenness_bin')
    n = len(connmat)
    return bct.betweenness_bin(binarize_mat(connmat))/((n-1)*(n-2))
    
def betweenness_wei(connmat):
    logging.debug('diciphr.connectivity.topology.betweenness_wei')
    G = bct.distance_wei(connmat)[0]
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

# Hemisphere measures 
def intrahemispheric_strength(conn, hemispheres=None, include_diagonal=False, assume_undirected=True, binary=True, nan_policy='omit'):
    """
    Compute the average edge weight of connections within the same hemisphere.

    Parameters
    ----------
    conn : np.ndarray
        Square (N x N) structural connectome (weights can be float; NaNs allowed).
    hemispheres : None | (list/array of length N) | (iterable, iterable), optional
        Hemisphere specification:
        - None: assume first N/2 nodes = left, second N/2 nodes = right (requires N even).
        - 1D labels of length N: any two distinct values (e.g., {0,1}, {'L','R'}).
        - Tuple of two iterables: explicit indices for (left_nodes, right_nodes).
    include_diagonal : bool, default False
        Whether to include self-connections in the average.
    assume_undirected : bool, default True
        If True, only unique edges are counted (upper triangle). If False, all i,j pairs are used.
    nan_policy : {'omit', 'propagate', 'zero'}, default 'omit'
        How to handle NaNs:
        - 'omit': ignore NaNs when averaging (np.nanmean).
        - 'propagate': if any selected value is NaN, result becomes NaN.
        - 'zero': treat NaNs as 0.

    Returns
    -------
    float
        Mean weight of intrahemispheric edges, according to the options.

    Raises
    ------
    ValueError
        If inputs are inconsistent (non-square matrix, invalid hemisphere spec, odd N with default split, etc.).
    """
    # --- Basic checks ---
    if conn.ndim != 2 or conn.shape[0] != conn.shape[1]:
        raise ValueError("`conn` must be a square (N x N) array.")
    N = conn.shape[0]
    if binary:
        conn = (conn>0)*1
    # --- Resolve hemisphere assignment into two boolean masks ---
    if hemispheres is None:
        if N % 2 != 0:
            raise ValueError("Default split requires even N (first N/2 left, second N/2 right).")
        left_idx = np.arange(N // 2)
        right_idx = np.arange(N // 2, N)
    elif isinstance(hemispheres, (tuple, list)) and len(hemispheres) == 2 and \
         hasattr(hemispheres[0], '__iter__') and hasattr(hemispheres[1], '__iter__'):
        left_idx = np.array(list(hemispheres[0]), dtype=int)
        right_idx = np.array(list(hemispheres[1]), dtype=int)
    else:
        labels = np.asarray(hemispheres)
        if labels.shape != (N,):
            raise ValueError("Label-based `hemispheres` must be a 1D array of length N.")
        uniq = np.unique(labels)
        if uniq.size != 2:
            raise ValueError("Label-based `hemispheres` must contain exactly two unique labels.")
        left_idx = np.where(labels == uniq[0])[0]
        right_idx = np.where(labels == uniq[1])[0]

    # Validate indices are disjoint and within range
    all_idx = np.concatenate([left_idx, right_idx])
    if not np.all((all_idx >= 0) & (all_idx < N)):
        raise ValueError("Hemisphere indices out of bounds.")
    if len(np.intersect1d(left_idx, right_idx)) > 0:
        raise ValueError("Left and right hemisphere index sets must be disjoint.")
    # (Optional) ensure coverage—if desired, we can allow unassigned nodes, but here we assume full partition:
    if np.unique(all_idx).size != N:
        raise ValueError("Hemisphere indices must cover all N nodes exactly once.")

    # --- Build intrahemispheric mask ---
    # Create boolean vectors for hemisphere membership
    is_left = np.zeros(N, dtype=bool); is_left[left_idx] = True
    is_right = np.zeros(N, dtype=bool); is_right[right_idx] = True

    same_hemi = np.outer(is_left, is_left) | np.outer(is_right, is_right)
    if not include_diagonal:
        np.fill_diagonal(same_hemi, False)

    # --- Select entries according to undirected/directed choice ---
    if assume_undirected:
        # Upper triangle selection
        k = 0 if include_diagonal else 1
        tri_mask = np.triu(np.ones_like(conn, dtype=bool), k=k)
        select_mask = same_hemi & tri_mask
    else:
        # Use all entries; (i,j) and (j,i) both included
        select_mask = same_hemi

    selected = conn[select_mask]

    # --- Handle NaNs per policy and compute mean ---
    if selected.size == 0:
        # No intrahemispheric edges under provided options
        return np.nan

    if nan_policy == 'omit':
        result = np.nanmean(selected)
    elif nan_policy == 'propagate':
        result = np.mean(selected)  # will be NaN if any NaN present
    elif nan_policy == 'zero':
        result = np.mean(np.nan_to_num(selected, nan=0.0))
    else:
        raise ValueError("`nan_policy` must be one of {'omit','propagate','zero'}.")

    return float(result)

def interhemispheric_strength(conn, hemispheres=None, include_diagonal=False, assume_undirected=True, binary=True, nan_policy='omit'):
    """
    Compute the average edge weight of connections between hemispheres.

    Parameters
    ----------
    conn : np.ndarray
        Square (N x N) structural connectome (weights can be float; NaNs allowed).
    hemispheres : None | (list/array of length N) | (iterable, iterable), optional
        Hemisphere specification:
        - None: assume first N/2 nodes = left, second N/2 nodes = right (requires N even).
        - 1D labels of length N: any two distinct values (e.g., {0,1}, {'L','R'}).
        - Tuple of two iterables: explicit indices for (left_nodes, right_nodes).
    include_diagonal : bool, default False
        Whether to include self-connections in the average (typically False).
    assume_undirected : bool, default True
        If True, only unique edges are counted (upper triangle). If False, all i,j pairs are used.
    nan_policy : {'omit', 'propagate', 'zero'}, default 'omit'
        How to handle NaNs:
        - 'omit': ignore NaNs when averaging (np.nanmean).
        - 'propagate': if any selected value is NaN, result becomes NaN.
        - 'zero': treat NaNs as 0.

    Returns
    -------
    float
        Mean weight of interhemispheric edges, according to the options.

    Raises
    ------
    ValueError
        If inputs are inconsistent (non-square matrix, invalid hemisphere spec, odd N with default split, etc.).
    """
    # --- Basic checks ---
    if conn.ndim != 2 or conn.shape[0] != conn.shape[1]:
        raise ValueError("`conn` must be a square (N x N) array.")
    N = conn.shape[0]
    if binary:
        conn = (conn>0)*1
    # --- Resolve hemisphere assignment into two index arrays ---
    if hemispheres is None:
        if N % 2 != 0:
            raise ValueError("Default split requires even N (first N/2 left, second N/2 right).")
        left_idx = np.arange(N // 2)
        right_idx = np.arange(N // 2, N)
    elif isinstance(hemispheres, (tuple, list)) and len(hemispheres) == 2 and \
         hasattr(hemispheres[0], '__iter__') and hasattr(hemispheres[1], '__iter__'):
        left_idx = np.array(list(hemispheres[0]), dtype=int)
        right_idx = np.array(list(hemispheres[1]), dtype=int)
    else:
        labels = np.asarray(hemispheres)
        if labels.shape != (N,):
            raise ValueError("Label-based `hemispheres` must be a 1D array of length N.")
        uniq = np.unique(labels)
        if uniq.size != 2:
            raise ValueError("Label-based `hemispheres` must contain exactly two unique labels.")
        left_idx = np.where(labels == uniq[0])[0]
        right_idx = np.where(labels == uniq[1])[0]

    # Validate indices: disjoint, in-range, and covering all nodes
    all_idx = np.concatenate([left_idx, right_idx])
    if not np.all((all_idx >= 0) & (all_idx < N)):
        raise ValueError("Hemisphere indices out of bounds.")
    if len(np.intersect1d(left_idx, right_idx)) > 0:
        raise ValueError("Left and right hemisphere index sets must be disjoint.")
    if np.unique(all_idx).size != N:
        raise ValueError("Hemisphere indices must cover all N nodes exactly once.")

    # --- Build interhemispheric mask (left↔right) ---
    is_left = np.zeros(N, dtype=bool); is_left[left_idx] = True
    is_right = np.zeros(N, dtype=bool); is_right[right_idx] = True

    # Different hemispheres: (left, right) OR (right, left)
    cross_lr = np.outer(is_left, is_right)
    cross_rl = np.outer(is_right, is_left)
    different_hemi = cross_lr | cross_rl

    if not include_diagonal:
        np.fill_diagonal(different_hemi, False)

    # --- Select entries according to undirected/directed choice ---
    if assume_undirected:
        # Upper triangle to avoid double counting
        k = 0 if include_diagonal else 1
        tri_mask = np.triu(np.ones_like(conn, dtype=bool), k=k)
        select_mask = different_hemi & tri_mask
    else:
        # Use all (i,j) including both directions
        select_mask = different_hemi

    selected = conn[select_mask]

    # --- Handle NaNs per policy and compute mean ---
    if selected.size == 0:
        return np.nan

    if nan_policy == 'omit':
        result = np.nanmean(selected)
    elif nan_policy == 'propagate':
        result = np.mean(selected)  # becomes NaN if any NaN present
    elif nan_policy == 'zero':
        result = np.mean(np.nan_to_num(selected, nan=0.0))
    else:
        raise ValueError("`nan_policy` must be one of {'omit','propagate','zero'}.")

    return float(result)

