import random
import networkx as nx
import numpy as np
from . import resistance_metrics as reff
import utils.resistance_metrics as ermet
import utils.link_addition as rew

from tqdm import tqdm

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


def add_links(G, number_of_edges, S, ratio=1, rand = False):
    def create_dicts(keys):
        return {key: [] for key in keys}
    
    def add_metrics(R, S, G, met_dict):
        total= ermet.group_total_reff(R, S)
        diam= ermet.group_reff_diam(R, S)
        avg_diam = ermet.group_avg_reff_diam(R, S)
        max_bet = ermet.group_max_reff_betweeness(R, G, S)
        avg_bet = ermet.group_avg_reff_betweeness(R, G, S)
        std_bet = ermet.group_std_reff_betweeness(R, G, S)
        #save data
        met_dict["total"].append(total)
        met_dict["diam"].append(diam)
        met_dict["avg_diam"].append(avg_diam)
        met_dict["max_bet"].append(max_bet)
        met_dict["avg_bet"].append(avg_bet)
        met_dict["std_bet"].append(std_bet)
        
    
    metrics = ['total', 'diam', 'avg_diam', 'max_bet', 'avg_bet', 'std_bet']
    strong_data = create_dicts(metrics)
    rand_data = create_dicts(metrics)
    weak_data = create_dicts(metrics)
    aff_weak_data = create_dicts(metrics)
    aff_strong_data = create_dicts(metrics)
    
    G_strong = G.copy()
    G_rand = G.copy()
    G_weak = G.copy()
    
    G_aff_weak= G.copy()
    G_aff_strong = G.copy()
        
    for i in tqdm(range(number_of_edges+1)):        
        #Strong
        cos_res_strong = ermet.effective_resistance_matrix(G_strong)
        add_metrics(cos_res_strong, S, G_strong, strong_data) 

        #Random
        if rand:
            cos_res_rand = ermet.effective_resistance_matrix(G_rand)
            add_metrics(cos_res_rand, S, G_rand, rand_data) 
        
        
        if i % ratio == 0:
            ## ER-Link ======================================
            cos_res_weak = ermet.effective_resistance_matrix(G_weak)
            add_metrics(cos_res_weak, S, G_weak, weak_data) 
            
            ## ERA-Link ======================================
            cos_res_aff_weak = ermet.effective_resistance_matrix(G_aff_weak)
            add_metrics(cos_res_aff_weak, S, G_aff_weak, aff_weak_data)
            
            ## STRONG AA ======================================
            cos_res_aff_strong = ermet.effective_resistance_matrix(G_aff_strong)
            add_metrics(cos_res_aff_strong, S, G_aff_strong, aff_strong_data)
            
            
        #add links
        G_strong = rew.add_strong_link(G_strong, cos_res_strong)
        if rand:
            G_rand = rew.add_rand_link(G_rand, cos_res_rand)
        if i % ratio == 0:
            G_weak = rew.add_weak_link(G_weak, cos_res_weak)
            G_aff_weak = rew.add_weak_affirmative_link(G_aff_weak, CT = cos_res_aff_weak, S=S)
            G_aff_strong = rew.add_strong_affirmative_link(G_aff_strong, CT =  cos_res_aff_strong, S=S)
        
    add_metrics(ermet.effective_resistance_matrix(G_strong), S, G_strong, strong_data)
    add_metrics(ermet.effective_resistance_matrix(G_weak), S, G_weak, weak_data) 
    add_metrics(ermet.effective_resistance_matrix(G_aff_weak), S, G_aff_weak, aff_weak_data)
    add_metrics(ermet.effective_resistance_matrix(G_aff_strong), S, G_aff_strong, aff_strong_data)
    if rand:
        return G_strong, G_weak, G_rand, strong_data, weak_data, rand_data
    else:
        return [G_strong, G_weak, G_aff_weak, G_aff_strong], [strong_data, weak_data, aff_weak_data, aff_strong_data]