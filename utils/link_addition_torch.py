import random
import torch
import numpy as np
import networkx as nx
from torch_geometric.nn import Node2Vec
from tqdm import tqdm
try:
    from karateclub import DeepWalk
except:
    pass

def get_weakest_link(score, edge_mask, n = None):
    argmax = (score*edge_mask).argmax().item() #detach and cpu on call
    n = score.shape[0] if n is None else n
    u, v = divmod(argmax, n)
    return u, v


def get_random_link(edge_mask, n=None):
    n = edge_mask.shape[0] if n is None else n
    u, v = random.sample(range(n), k=2)
    while u==v or not edge_mask[u,v]:
        u, v = random.sample(range(n), k=2)
    return u, v


def get_strongest_link(score, edge_mask, n = None):
    score[torch.logical_not(edge_mask)]=torch.inf
    argmin = score.argmin().item()
    n = score.shape[0] if n is None else n
    u, v = divmod(argmin, n)
    return u, v

def get_edge_score(GW, model, **kwargs):
    if model == 'ERP':
        return GW.get_effective_resistance()
    elif model == 'deepwalk':
        nxG = nx.from_edgelist(GW.edgelist.T.detach().cpu().numpy())
        dw = DeepWalk(dimensions=kwargs['dim'], walk_length=kwargs['walk_length'], window_size=kwargs['window_size'], workers=kwargs['workers'])
        dw.fit(nxG.copy())
        Z = dw.get_embedding()
        S = np.dot(Z, Z.T)
        np.fill_diagonal(S,0)
        return torch.tensor(S)
    elif model == 'node2vec':
        model = Node2Vec(GW.edgelist, embedding_dim=kwargs['dim'], walk_length=kwargs['walk_length'],
                 context_size=10, walks_per_node=10,
                 num_negative_samples=1, p=1, q=1, sparse=True).to(kwargs['device'])

        sampleloader = model.loader(batch_size=64, shuffle=True, num_workers=1)  # data loader to speed the train 
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)  # initzialize the optimizer 

        for epoch in tqdm(range(1, 15)):
            model.train()  # put model in train model
            total_loss = 0
            
            for pos_rw, neg_rw in sampleloader:
                optimizer.zero_grad()  # set the gradients to 0
                loss = model.loss(pos_rw.to(kwargs['device']), neg_rw.to(kwargs['device']))  # compute the loss for the batch
                loss.backward()
                optimizer.step()  # optimize the parameters
                #total_loss += loss.item()
                #return total_loss / len(loader)
                
        model.eval()
        Z = model(torch.arange(GW.num_nodes, device=kwargs['device']))
        return torch.matmul(Z, Z.T)

    elif model == 'cosine':
        from sklearn.metrics.pairwise import cosine_similarity
        from scipy.sparse import coo_matrix

        edge_index = GW.edgelist.detach().cpu()
        n = GW.num_nodes
        adj_sparse = coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
                         shape=(n, n))

        # Compute cosine similarity using sklearn
        cos_sim = cosine_similarity(adj_sparse)

        return torch.Tensor(cos_sim)

        
        
    else:
        raise ValueError("Model not found")