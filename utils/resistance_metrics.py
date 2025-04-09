import numpy as np
import networkx as nx
from scipy.sparse import csgraph
import torch
import torch_geometric as pyg


def effective_resistance_matrix(network):
    """
    Res \in R^{n x n} = (1^T * diag(L^+)) + (diag(L^+)^T * 1) - (2 * L^+)
    """
    L = csgraph.laplacian(np.matrix(nx.adjacency_matrix(network).todense().astype(float)), normed = False)
    Q = np.linalg.pinv(L)
    zeta = np.diag(Q)
    u = np.ones(zeta.shape[0])
    return np.array((np.matrix(u).T * zeta) + (np.matrix(zeta).T * u) - (2 * Q))

def effective_resistance_matrix_3(network):
    """
    Res \in R^{n x n} = (1^T * diag(L^+)) + (diag(L^+)^T * 1) - (2 * L^+)
    """
    L = csgraph.laplacian(np.matrix(nx.adjacency_matrix(network).todense().astype(float)), normed = False)
    pinv = np.linalg.pinv(L)
    pinv_diagonal = np.diagonal(pinv)
    resistance_matrix = pinv_diagonal.unsqueeze(0) + pinv_diagonal.unsqueeze(1) - 2*pinv
    return resistance_matrix

def effective_resistance_matrix_2(network):
    """Calculate effective resistance for each node pair in the network.
    Res(u,v) \in R^1 = (L + 1/n )^+[i,i] + (L + 1/n )^+[j,j] - 2*(L + 1/n )^+[i,j] 

    Parameters:
    ----------
    network: networkx graph.
    """

    n = len(network.nodes)
    L = csgraph.laplacian(np.matrix(nx.adjacency_matrix(network).todense().astype(float)), normed = False)
    Phi = np.ones((n, n)) / n
    Gamma = np.linalg.pinv(L + Phi)

    # calculate resistance for all node pairs
    res = np.array(
        [[Gamma[i, i] + Gamma[j, j] - (2 * Gamma[i, j]) if i != j else 0 for j in range(n)] for i in range(n)])

    return res


def torch_resistance_matrix(ei, ew, L=None, Linv=None, inversion='ones', device='cuda', n=None):
    """
    Torch computation of effective effective resistance matrix.
    $$\mathbf{R} \in \mathbb{R}^{n x n} = (1^T * diag(L^+)) + (diag(L^+)^T * 1) - (2 * L^+)$$
    where L^+ is the pseudoinverse of the Laplacian matrix L if inversion='pinv' or the inverse of L+1.1^T/n if inversion='ones'.
    Args:
        ei (_type_): adjacency edge_index: 2x|E|. E.g. i-th is (ei[])
        ew (_type_): adjacency edge_weight: 1x|E| with the weight of the i-th edge in ei. E.g 1 if Adjacency matrix
        L (_type_, optional): precomputed L. Defaults to None.
        Linv (_type_, optional): precomputed pinv. Defaults to None.
        inversion (str, optional): type of inverse computation: pseudoinverse or inverse of L+1.1^T/n. Defaults to 'pinv'.
        device (str, optional): device to send the Laplacian or the ones matrix if inversion='ones'. Defaults to 'cpu'.
        n (int, optional): number of nodes. Defaults to None.
    Returns:
        torch.tensor: complete pairwise resistance matrix
    """
    if Linv is None:
        if L is None:
            L_ei, Lew  = pyg.utils.get_laplacian(ei, ew)
            L = pyg.utils.to_dense_adj(edge_index = L_ei, edge_attr = Lew)[0] 
            L = L.to(device)

        if inversion =='pinv':
            Linv = torch.linalg.pinv(L, hermitian=True)
        elif inversion =='ones':
            n = L.shape[0] if n is None else n
            #Phi = (torch.ones((n, n), dtype=L.dtype) / n).to(device)
            Phi = 1/n
            #this might be optimized to L+(1/n) instead of Phi to save memory but still not tested
            Linv = torch.linalg.inv(L+Phi)
        else:
            raise ValueError(f"Invalid inversion type: {inversion}. Valid types: ['pinv', 'ones']")
        
    pinv_diagonal = torch.diagonal(Linv)
    # u = torch.ones(pinv_diagonal.shape[0])
    # Next line is equivalent to u.unsqueeze(1) @ pinv_diagonal.unsqueeze(0) +  pinv_diagonal.unsqueeze(1) @ u.unsqueeze(0)- 2*pinv
    return pinv_diagonal.unsqueeze(0) +  pinv_diagonal.unsqueeze(1) - 2*Linv


def biharmonic(network):
    """Calculate biharmonic distance for each node pair in the network.
    Res(u,v) \in R^1 = (L + 1/n )^2+[i,i] + (L + 1/n )^2+[j,j] - 2*(L + 1/n )^2+[i,j] 

    Parameters:
    ----------
    network: networkx graph.
    """

    n = len(network.nodes)
    L = csgraph.laplacian(np.matrix(nx.adjacency_matrix(network).todense().astype(float)), normed=False)
    Phi = np.ones((n, n)) / n
    Gamma = np.linalg.matrix_power(np.linalg.pinv(L + Phi),2)

    # calculate resistance for all node pairs
    res = np.array(
        [[Gamma[i, i] + Gamma[j, j] - (2 * Gamma[i, j]) if i != j else 0 for j in range(n)] for i in range(n)])

    return res

def total_effective_resistance(CT, G=None, filtered=False, mean=False):
    """    
    Implementation of total effective resistance (G.1. or G.2.)

    Args:
        CT (_type_): _description_
        G (_type_, optional): _description_. Defaults to None.
        filtered (bool, optional): _description_. Defaults to False.
        mean (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_

    
    ones = np.ones((R.shape[0],))
    # Perform the operation
    ones.dot(R).dot(ones.T)
    """
    if not filtered:
        pairwise_resistances = CT[np.triu_indices(np.sum(CT.shape[0]),k=1)].ravel().copy()
    else: 
        A = nx.adjacency_matrix(G)
        pairwise_resistances = (CT*A.toarray())[np.triu_indices(np.sum(CT.shape[0]),k=1)].ravel().copy()
    
    total_ER = np.sum(pairwise_resistances) if not mean else np.sum(pairwise_resistances)/CT.shape[0]
    return total_ER

def resistance_diameter(CT, G=None, filtered=False):
    """Resistance diameter (G.3)

    Args:
        CT (_type_): _description_
        G (_type_, optional): _description_. Defaults to None.
        filtered (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    if not filtered:
        pairwise_resistances = CT[np.triu_indices(np.sum(CT.shape[0]),k=1)].ravel().copy()
    else: 
        A = nx.adjacency_matrix(G)
        pairwise_resistances = (CT*A.toarray())[np.triu_indices(np.sum(CT.shape[0]),k=1)].ravel().copy()
    return np.max(pairwise_resistances)


def avg_node_max_distance(CT, G=None, filtered=False):
    """Resistance diameter (G.5)
    
    Args:
        CT (_type_): _description_
        G (_type_, optional): _description_. Defaults to None.
        filtered (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    if filtered:
        A = nx.adjacency_matrix(G)
        CT = (CT*A.toarray()).copy()
        
    node_diameters = np.max(CT, axis=0)
    return np.mean(node_diameters)

def avg_node_total_er(CT, G=None, filtered=False):
    """Avg node total resistance (G.4). Equal to G.2 and prop to G.1.

    Args:
        CT (_type_): _description_
        G (_type_, optional): _description_. Defaults to None.
        filtered (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    if filtered:
        A = nx.adjacency_matrix(G)
        CT = (CT*A.toarray()).copy()
        
    nodes_total_er = np.sum(CT, axis=0)
    return np.mean(nodes_total_er)

def avg_node_mean_er(CT, G=None, filtered=False):
    """Avg node mean resistance (G.4). Prop to G.1 and G2.

    Args:
        CT (_type_): _description_
        G (_type_, optional): _description_. Defaults to None.
        filtered (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    if filtered:
        A = nx.adjacency_matrix(G)
        CT = (CT*A.toarray()).copy()
        
    nodes_total_er = np.mean(CT, axis=0)
    return np.mean(nodes_total_er)
    


## Node metrics

def node_total_er(CT, G=None, filtered=False):
    """Node total ER

    Args:
        CT (_type_): _description_
        G (_type_, optional): _description_. Defaults to None.
        filtered (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    if filtered:
        A = nx.adjacency_matrix(G)
        CT = (CT*A.toarray()).copy()
    return np.sum(CT, axis=0)

def node_mean_er(CT, G=None, filtered=False):
    """Node mean ER

    Args:
        CT (_type_): _description_
        G (_type_, optional): _description_. Defaults to None.
        filtered (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    if filtered:
        A = nx.adjacency_matrix(G)
        CT = (CT*A.toarray()).copy()
    return np.mean(CT, axis=0)

def node_diameters(CT, G=None, filtered=False):
    """node R diameter

    Args:
        CT (_type_): _description_
        G (_type_, optional): _description_. Defaults to None.
        filtered (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    if filtered:
        A = nx.adjacency_matrix(G)
        CT = (CT*A.toarray()).copy()
    return np.max(CT, axis=0)



### Group metrics

def group_total_reff(CT, S):
    """
    Implementation of group effective resistance
    """
    node_total_ers = node_total_er(CT)
    
    # get unique groups
    unique_S = np.unique(S)

    # calculate mean distance for each group --> if different S have different sizes, sum is not the best
    total_er = {}
    for s in unique_S:
        total_er[s] = np.mean(node_total_ers[S == s])
        
    return total_er

def group_reff_diam(CT, S):
    """
    Implementation of resistance diameter
    """
    node_total_ers = node_diameters(CT)
    
    # get unique groups
    unique_S = np.unique(S)

    diams = {}
    for s in unique_S:
        diams[s] = np.max(node_total_ers[S == s])
        
    return diams

def group_avg_reff_diam(CT, S):
    """
    Implementation of average resistance diameter
    """
    node_total_ers = node_diameters(CT)
    
    # get unique groups
    unique_S = np.unique(S)

    diams = {}
    for s in unique_S:
        diams[s] = np.mean(node_total_ers[S == s])
        
    return diams

def group_avg_reff_betweeness(CT, G, S):
    """
    Implementation of average resistance betweenness
    """
    node_total_bet = node_total_er(CT, G, filtered=True)
    
    # get unique groups
    unique_S = np.unique(S)

    betwens = {}
    for s in unique_S:
        betwens[s] = np.mean(node_total_bet[S == s])
        
    return betwens

def group_max_reff_betweeness(CT, G, S):
    """
    Implementation of max resistance betweenness
    """
    node_total_bet = node_total_er(CT, G, filtered=True)
    
    # get unique groups
    unique_S = np.unique(S)

    betwens = {}
    for s in unique_S:
        betwens[s] = np.max(node_total_bet[S == s])
        
    return betwens

def group_std_reff_betweeness(CT, G, S):
    """
    Implementation of std resistance betweenness
    """
    node_total_bet = node_total_er(CT, G, filtered=True)
    
    # get unique groups
    unique_S = np.unique(S)

    betwens = {}
    for s in unique_S:
        betwens[s] = np.std(node_total_bet[S == s])
        
    return betwens

