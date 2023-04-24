import random
import networkx as nx
import numpy as np
from . import resistance_metrics as reff

def add_strong_link(G, CT=None):
    if CT is None:
        pass #calcuate CT

    A = nx.adjacency_matrix(G).A
    index_to_node = {index: node for index, node in enumerate(G.nodes())}
    G=G.copy()
    
    potential_strong_links = (A+np.eye(A.shape[0])) == 0
    nonzero_mask = CT != 0
    f =nonzero_mask & potential_strong_links
    u,v = np.where((CT==CT[f].min()))
    u, v = u[0], v[0]
    # find the overall index of the maximum
    #u, v = np.where(potential_strong_links)[0][min_idx], np.where(potential_strong_links)[1][min_idx]
    u, v = index_to_node[u], index_to_node[v]
    assert not u==v, "Problem: self-link"
    assert not G.has_edge(u,v), f"Problem: Maximum ER in an existing edge. Edge: {u},{v}"
    G.add_edge(u,v)
    return G

def add_rand_link(G, CT=None):
    if CT is None:
        pass #calcuate CT
    
    G=G.copy()
    u, v = random.sample(list(G.nodes()), k=2)
    while G.has_edge(u,v):
        u, v = random.sample(list(G.nodes()), k=2)
    assert not u==v, "Problem: self-link"
    assert not G.has_edge(u,v), f"Problem: Maximum ER in an existing edge. Edge: {u},{v}"
    G.add_edge(u,v)
    return G
            
            
def add_weak_link(G, CT=None):
    if CT is None:
        pass #calcuate CT
    A = nx.adjacency_matrix(G).A
    index_to_node = {index: node for index, node in enumerate(G.nodes())}
    G=G.copy()
    
    potential_weak_links = (A+np.eye(A.shape[0])) == 0
    #max_idx = np.argmax(cos_res_weak[potential_weak_links])
    #find the overall index of the maximum
    #edge_diameter = np.where(potential_weak_links)[0][max_idx], np.where(potential_weak_links)[1][max_idx]

    edge_diameter = np.unravel_index((CT*potential_weak_links).argmax(), 
                                     A.shape)
    u, v = edge_diameter
    u, v = index_to_node[u], index_to_node[v]
    assert not u==v, "Problem: self-link"
    assert not G.has_edge(u,v), f"Problem: Maximum ER in an existing edge. Edge: {u},{v}"
    G.add_edge(u,v)
    return G


def add_strong_affirmative_link(G, S, CT=None):
    if CT is None:
        CT =  reff.effective_resistance_matrix(G) #calcuate CT

    A = nx.adjacency_matrix(G).A
    index_to_node = {index: node for index, node in enumerate(G.nodes())}
    G=G.copy()
    
    potential_strong_links = (A+np.eye(A.shape[0])) == 0
    nonzero_mask = CT != 0
    S_filter = np.logical_or(S[:, None], S)
    f = nonzero_mask & potential_strong_links & S_filter
    # find the overall index of the maximum
    #u, v = np.where(potential_strong_links)[0][min_idx], np.where(potential_strong_links)[1][min_idx]
    u,v = np.where((CT==CT[f].min()))
    u, v = u[0], v[0]
    u, v = index_to_node[u], index_to_node[v]
    assert not u==v, "Problem: self-link"
    assert not G.has_edge(u,v), f"Problem: Maximum ER in an existing edge. Edge: {u},{v}"
    G.add_edge(u,v)
    return G

def add_weak_affirmative_link(G, S, CT=None):
    if CT is None:
        CT =  reff.effective_resistance_matrix(G) #calcuate CT
    A = nx.adjacency_matrix(G).A
    index_to_node = {index: node for index, node in enumerate(G.nodes())}
    G=G.copy()
    
    potential_links = (A+np.eye(A.shape[0])) == 0
    S_filter = np.logical_or(S[:, None], S)
    potential_affirmative_links = potential_links & S_filter

    edge_diameter = np.unravel_index((CT*potential_affirmative_links).argmax(), 
                                     A.shape)
    u, v = edge_diameter
    u, v = index_to_node[u], index_to_node[v]
    assert not u==v, "Problem: self-link"
    assert not G.has_edge(u,v), f"Problem: Maximum ER in an existing edge. Edge: {u},{v}"
    G.add_edge(u,v)
    return G