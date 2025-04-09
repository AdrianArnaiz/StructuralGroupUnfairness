import networkx as nx
import numpy as np
from scipy.sparse import csgraph

def psudoinverseL(network):
    A = nx.adjacency_matrix(network).todense().astype(float)
    return np.linalg.pinv(csgraph.laplacian(np.matrix(A), normed=False))

def find_evecs(L):
    e, evecs = np.linalg.eig(L.todense())
    idx =e.argsort()
    e = e[idx]
    evecs = evecs[:,idx]
    return e, evecs
