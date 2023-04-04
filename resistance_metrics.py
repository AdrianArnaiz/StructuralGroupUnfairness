import numpy as np
import networkx as nx

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