import numpy as np
import pandas as pd
import networkx as nx
import os

from scipy.sparse import csr_matrix, find
from scipy.sparse.csgraph import connected_components

import torch
import torch_geometric as pyg
import torch_geometric.transforms as T


from utils.GraphWrapper import GraphWrapper

def process_facebook(folder_link, return_adj=True):
    """Code based on "PyGDebias: Graph Datasets and Fairness-Aware Graph Mining Algorithms". 
    Dong, Yushun and Ma, Jing and Chen, Chen and Li, Jundong
    https://github.com/yushundong/PyGDebias/blob/main/dataloading.py

    Modifications to return the biggest connected component of 
    the graph respecting the index of the nodes, important for features, labels and sens
    """
    edges_file=open(folder_link+'/edges.txt')
    edges=[]
    for line in edges_file:
        edges.append([int(one) for one in line.strip('\n').split(' ')])

    feat_file=open(folder_link+'/feat.txt')
    feats=[]
    for line in feat_file:
        feats.append([int(one) for one in line.strip('\n').split(' ')])
    feats=np.array(feats)

    node_to_idx={} #map node original name to index
    for j in range(feats.shape[0]):
        node_to_idx[feats[j][0]]=j

    #change edges name to index
    for i in range(len(edges)):
        edges[i][0] = node_to_idx[edges[i][0]]
        edges[i][1] = node_to_idx[edges[i][1]]
    edges=np.array(edges)
    #change edges shape from (n,2) to (2,n)
    edges=np.transpose(edges)

    #! Get biggest connected component
    # create a sparse matrix from the edges
    n = feats.shape[0]
    adj_matrix = csr_matrix((np.ones(edges.shape[1]), edges), shape=(n, n))
    # get the connected components
    n_components, labels = connected_components(adj_matrix, directed=False)
    # get the idx of the biggest component
    biggest_component = np.argmax(np.bincount(labels))
    #filter the matrix and features to those for the bcc
    adj_matrix = adj_matrix[labels==biggest_component,:][:,labels==biggest_component]
    feats = feats[labels==biggest_component,:]

    #remove index
    feats=feats[:,1:]

    #extract labels and sensitive attributes
    sens=feats[:,264]
    labels=feats[:,220]

    feats=np.concatenate([feats[:,:264],feats[:,266:]],-1)
    feats=np.concatenate([feats[:,:220],feats[:,221:]],-1)

    if return_adj:
        return adj_matrix, feats, labels, sens
    else:
        #r, c, v = find(adj_matrix)
        #edges = np.vstack((r,c))
        edges, ew = pyg.utils.convert.from_scipy_sparse_matrix(adj_matrix)
        return edges, feats, labels, sens



"""def load_pokec(dataset,sens_attr,predict_attr, path="../dataset/pokec/", label_number=1000,sens_number=500,seed=19,test_idx=False):
    #! Code based on https://github.com/donglgcn/FairAC/blob/main/src/utils.py
    print('Loading {} dataset from {}'.format(dataset,path))

    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove("user_id")

    header.remove(sens_attr)
    header.remove(predict_attr)


    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    

    # build graph
    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(os.path.join(path,"{}_relationship.txt".format(dataset)), dtype=int)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    # adj = sparse_mx_to_torch_sparse_tensor(adj)

    import random
    random.seed(seed)
    label_idx = np.where(labels>=0)[0]
    random.shuffle(label_idx)

    idx_train = label_idx[:min(int(0.5 * len(label_idx)),label_number)]
    idx_val = label_idx[int(0.5 * len(label_idx)):int(0.75 * len(label_idx))]
    if test_idx:
        idx_test = label_idx[label_number:]
        idx_val = idx_test
    else:
        idx_test = label_idx[int(0.75 * len(label_idx)):]




    sens = idx_features_labels[sens_attr].values

    sens_idx = set(np.where(sens >= 0)[0])
    idx_test = np.asarray(list(sens_idx & set(idx_test)))
    sens = torch.FloatTensor(sens)
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    random.seed(seed)
    random.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])


    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    

    # random.shuffle(sens_idx)

    return adj, features, labels, idx_train, idx_val, idx_test, sens,idx_sens_train"""




def load_data(dataset, device, dtype=torch.float32, **kwargs):
    """ Load the dataset and return a GraphWrapper object """
    if dataset == "cora":
        transform = T.Compose([T.ToUndirected(), T.LargestConnectedComponents()])
        dataset = pyg.datasets.Planetoid(root='data', name='Cora', transform=transform)
        GW = GraphWrapper(dataset[0].edge_index, labels=dataset[0].y, device=device, dtype=dtype)
        del dataset
        return GW
    
    elif dataset == "pubmed":
        transform = T.Compose([T.ToUndirected(), T.LargestConnectedComponents()])
        dataset = pyg.datasets.Planetoid(root='data', name='PubMed', transform=transform)
        GW = GraphWrapper(dataset[0].edge_index, labels=dataset[0].y, device=device, dtype=dtype)
        del dataset
        return GW
    
    elif dataset == "facebook":
        edges, _, _, sens = process_facebook('data/facebook', return_adj=False)
        return GraphWrapper(edges, sensitive_attr=sens, device=device, dtype=dtype)
    
    elif dataset == "UNC28":
        import pickle as pkl
        # Load sensisitve attr
        sens = pkl.load(open(r'data/unc/UNC28_user_sen.pkl',"rb"))
        sens = [sens[idx] for idx in range(len(sens.keys()))]
        sens = np.array(sens)

        # Load edges
        train_items = pkl.load(open(r'data\unc\UNC28_train_items.pkl',"rb"))
        test_items = pkl.load(open(r'data\unc\UNC28_test_set.pkl',"rb"))
        edgelist = []
        for item in [train_items, test_items]:
            for node, neighs in item.items():
                for node2 in neighs:
                    edgelist.append([node, node2])
        edgelist = np.array(edgelist).T

        #* Get Biggest Connected component
        # create a sparse matrix from the edges
        n = sens.shape[0]
        adj_matrix = csr_matrix((np.ones(edgelist.shape[1]), edgelist), shape=(n, n))
        # get the connected components
        n_components, labels = connected_components(adj_matrix, directed=False)
        # get the idx of the biggest component
        biggest_component = np.argmax(np.bincount(labels))
        #filter the matrix and features to those for the bcc
        adj_matrix = adj_matrix[labels==biggest_component,:][:,labels==biggest_component]
        sens = sens[labels==biggest_component]
        edges, ew = pyg.utils.convert.from_scipy_sparse_matrix(adj_matrix)
        return GraphWrapper(edges, sensitive_attr=sens, device=device, dtype=dtype)

    elif dataset == "cat":
        edges = [(0,1),(0,3),(1,2),(1,3),(1,4),(2,4),(2,11),(3,4),(3,5),
                    (4,5),(5,6),(5,7),(6,7),(6,8),(6,9), (7,9),(7,10),(8,9),
                    (9,10),(10,15),(11,12),(12,13),(13,14),(14,15),(0,11),(8,15)]
        edges = torch.Tensor(edges).T.type(torch.long)
        edges = torch.cat([edges, torch.vstack([edges[1],edges[0]])], dim=1)

        sens = torch.Tensor([0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0]).type(torch.long)
        return GraphWrapper(edges, sensitive_attr=sens, device=device, dtype=dtype)
    
    elif dataset == 'deezer':
        import csv
        edges=[]
        with open('data\deezer\deezer_europe_edges.csv','r') as edges_file:
            reader = csv.reader(edges_file, delimiter=',')
            for row in reader:
                try:
                    edges.append([int(v) for v in row])
                except:
                    print('Excluded:', row)
        edges = np.array(edges)
        bidir_edges = np.vstack([edges, edges[:,[1,0]]])
        bidir_edges = torch.Tensor(bidir_edges).long().T

        sens = []
        with open('data\deezer\deezer_europe_target.csv','r') as edges_file:
            reader = csv.reader(edges_file, delimiter=',')
            for row in reader:
                try:
                    sens.append([int(v) for v in row])
                except:
                    print('Excluded:', row)
        sens = np.array(sens)
        #sort by first column
        sens = torch.Tensor(sens[sens[:,0].argsort()][:,1]).long()

        return GraphWrapper(bidir_edges, sensitive_attr=sens, device=device, dtype=dtype)

    elif dataset == "google":
        #* Read files of edges and features
        PATH = 'data/google/'
        edges = []
        with open(PATH + "google.edges.txt") as edges_file:
            for line in edges_file:
                edges.append([int(one) for one in line.strip("\n").split(" ")])

        feats = []
        with open(PATH + "google.feat.txt") as feats_file:
            for line in feats_file:
                feats.append([int(one) for one in line.strip("\n").split(" ")])

        feat_name = []
        with open(PATH + "google.featnames.txt") as feat_name_file:
            for line in feat_name_file:
                feat_name.append(line.strip("\n").split(" "))
            names = {}
            for name in feat_name:
                if name[1] not in names:
                    names[name[1]] = name[1]  

        #* Map original node id to new ID by order of appearance in the edgelist (thus, we only get non-isolated nodes)
        unique_nodes = np.unique(edges)
        n_nodes = unique_nodes.shape[0]
        node_mapping = {}
        for j, id in enumerate(unique_nodes):
            node_mapping[id] = j

        #* We order the features by the new node ID, so id coincide with the row id in the features
        feats = np.array(feats)
        filter_feats = np.zeros((n_nodes, feats.shape[1]), dtype='O')
        for node in node_mapping:
            filter_feats[node_mapping[node]] = feats[feats[:,0] == node]
        del feats


        #* Remove ID and select sensitive attribute and labels
        filter_feats = filter_feats[:, 1:]
        filter_feats = np.array(filter_feats, dtype=float)
        sens = torch.Tensor(filter_feats[:, 0])
        labels = torch.Tensor(filter_feats[:, 164])
        # Remove sensitive attribute and labels from features
        filter_feats = np.concatenate([filter_feats[:, :164], filter_feats[:, 165:]], -1)
        filter_feats = filter_feats[:, 1:]

        #* Read edges, map by node ID, make bidirectional, remove duplicated edges and transpose to (2,N) shape
        edges = np.array(edges)
        newedges = np.array([[node_mapping[e[0]], node_mapping[e[1]]]for e in edges])
        newedges = np.vstack([newedges, newedges[:,[1,0]]])
        newedges = np.unique(newedges, axis=0).astype(np.int64)

        return GraphWrapper(torch.Tensor(newedges.T).long(), sensitive_attr=sens, device=device)

    elif dataset == "twitter":
        raise NotImplementedError("Twitter dataset not implemented yet")
    elif dataset == "emailarenas":
        raise NotImplementedError("Arenas dataset not implemented yet")
    elif dataset == "pokecz":
        """df = pd.read_csv(f'data{os.sep}pokec_regions{os.sep}region_job.csv', index_col=0)
        pokek = nx.read_edgelist(f'data{os.sep}pokec_regions{os.sep}region_job_relationship.txt')
        pokek = nx.relabel_nodes(pokek, {node:int(node) for node in pokek.nodes()}, copy=True) #str to int
        nx.set_node_attributes(pokek, dict(df['gender']), name='sens')
        nx.set_node_attributes(pokek, dict(df['region']), name='region')

        Gcc = sorted(nx.connected_components(pokek), key=len, reverse=True)
        G = nx.Graph(pokek.subgraph(Gcc[0]).copy())
        del pokek
        del df
        G = nx.relabel_nodes(G, {node:ix for ix, node in enumerate(G.nodes())}, copy=True) # from 0 to n: simplify adding edges

        sensitive_group = nx.get_node_attributes(G, 'sens')
        # sensitive group to numpy array
        sensitive_group = np.array(list(sensitive_group.values()))"""
        raise NotImplementedError("Pokec dataset not implemented yet")
    elif dataset == "pokecn":
        """df = pd.read_csv(f'data{os.sep}pokec_regions{os.sep}region_job_2.csv', index_col=0)
        pokek = nx.read_edgelist(f'data{os.sep}pokec_regions{os.sep}region_job_2_relationship.txt')
        pokek = nx.relabel_nodes(pokek, {node:int(node) for node in pokek.nodes()}, copy=True) #str to int
        nx.set_node_attributes(pokek, dict(df['gender']), name='sens')
        nx.set_node_attributes(pokek, dict(df['region']), name='region')

        Gcc = sorted(nx.connected_components(pokek), key=len, reverse=True)
        G = nx.Graph(pokek.subgraph(Gcc[0]).copy())
        del pokek
        del df
        G = nx.relabel_nodes(G, {node:ix for ix, node in enumerate(G.nodes())}, copy=True) # from 0 to n: simplify adding edges

        sensitive_group = nx.get_node_attributes(G, 'sens')
        # sensitive group to numpy array
        sensitive_group = np.array(list(sensitive_group.values()))"""
        raise NotImplementedError("Pokec dataset not implemented yet")
    
    elif dataset == "SBM":
        sizes = [40, 40, 15, 15]
        #sizes = [150, 150, 50, 50]
        c1 = 0.8
        c2 = 0.8
        c3 = 0.8 
        c4 = 0.95 
        c1_c2 = 0.03
        c1_c3 = 0.01
        c2_c3 = 0.01
        c1_c4 = 0.00
        c2_c4 = 0.00
        c3_c4 = 0.02

        group_by_com = [0,0,1,0]
        noises = [0,0,0,0]
        #noises = [0,0,0,0]
        probs = [[c1, c1_c2, c1_c3, c1_c4],
                [c1_c2, c2, c2_c3, c2_c4],
                [c1_c3, c2_c3, c3, c3_c4],
                [c1_c4, c2_c4, c3_c4, c4]]
        G = nx.stochastic_block_model(sizes, probs, seed=0)

        sens = np.array([])
        for ix, s in enumerate(sizes):
            s_c = np.array([group_by_com[ix]]*s)
            if noises[ix] !=0:
                idx = np.random.choice(s,int(s*noises[ix]),replace=False)
                idx = np.random.choice(range(s)) if len(idx)==0 else idx
                s_c[idx] = 1 - s_c[idx]
            sens = np.hstack([sens, s_c])
        edges = torch.Tensor(list(G.edges)).T.type(torch.long)
        edges = torch.cat([edges, torch.vstack([edges[1],edges[0]])], dim=1)
        return GraphWrapper(edges, sensitive_attr=sens, device=device, dtype=dtype)

    elif dataset == "karate":
        """G = nx.karate_club_graph()
        sensitive_group = np.ones(len(G.nodes()))
        sensitive_group[-4:] = 0"""
        raise NotImplementedError("Karate dataset not implemented yet")
    
    elif dataset == 'edgelist':
        return GraphWrapper(kwargs['edgelist'], sensitive_attr=kwargs['sens'], device=device, dtype=dtype)

    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented yet")