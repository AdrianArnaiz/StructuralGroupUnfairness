import numpy as np

def process_facebook(folder_link, return_tensor_sparse=True):
    """Code based on "PyGDebias: Graph Datasets and Fairness-Aware Graph Mining Algorithms". 
    Dong, Yushun and Ma, Jing and Chen, Chen and Li, Jundong
    https://github.com/yushundong/PyGDebias/blob/main/dataloading.py
    """
    edges_file=open(folder_link+'/edges.txt')
    edges=[]
    for line in edges_file:
        edges.append([int(one) for one in line.strip('\n').split(' ')])

    feat_file=open(folder_link+'/feat.txt')
    feats=[]
    for line in feat_file:
        feats.append([int(one) for one in line.strip('\n').split(' ')])

    feat_name_file = open(folder_link+'/featnames.txt')
    feat_name = []
    for line in feat_name_file:
        feat_name.append(line.strip('\n').split(' '))
    names={}
    for name in feat_name:
        if name[1] not in names:
            names[name[1]]=name[1]
        if 'gender' in name[1]:
            print(name)

    #print(feat_name)
    feats=np.array(feats)

    node_mapping={}
    for j in range(feats.shape[0]):
        node_mapping[feats[j][0]]=j

    feats=feats[:,1:]

    #print(feats.shape)
    #for i in range(len(feat_name)):
    #    print(i, feat_name[i], feats[:,i].sum())

    sens=feats[:,264]
    labels=feats[:,220]

    feats=np.concatenate([feats[:,:264],feats[:,266:]],-1)

    feats=np.concatenate([feats[:,:220],feats[:,221:]],-1)

    edges=np.array(edges)
    #edges=torch.tensor(edges)
    #edges=torch.stack([torch.tensor(one) for one in edges],0)
    #print(len(edges))

    node_num=feats.shape[0]
    adj=np.zeros([node_num,node_num])


    for j in range(edges.shape[0]):
        adj[node_mapping[edges[j][0]],node_mapping[edges[j][1]]]=1


    idx_train=np.random.choice(list(range(node_num)),int(0.8*node_num),replace=False)
    idx_val=list(set(list(range(node_num)))-set(idx_train))
    idx_test=np.random.choice(idx_val,len(idx_val)//2,replace=False)
    idx_val=list(set(idx_val)-set(idx_test))

    #features = torch.FloatTensor(feats)
    #sens = torch.FloatTensor(sens)
    #idx_train = torch.LongTensor(idx_train)
    #idx_val = torch.LongTensor(idx_val)
    #idx_test = torch.LongTensor(idx_test)
    #labels = torch.LongTensor(labels)

    #features=torch.cat([features,sens.unsqueeze(-1)],-1)
    #adj=mx_to_torch_sparse_tensor(adj,return_tensor_sparse)

    features = feats
    return adj, features, labels, idx_train, idx_val, idx_test, sens
