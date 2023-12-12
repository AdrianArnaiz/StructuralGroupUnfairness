import torch



#* Graph Metrics
def total_er(R):
    return torch.sum(R)/2

def avg_total_er(R):
    return torch.sum(R)/2/R.shape[0]

def total_er_filtered(R):
    return R.shape[0] - 1

def diam_ER(R):
    return torch.max(R)


#* Node metrics
def node_total_er(R):
    return torch.sum(R, axis=0)

def node_diam(R):
    return torch.max(R, axis=0)[0]

def node_betwenness(R, filter):
    return torch.sum(R*filter, axis=0)



#* Group metrics
def group_total_er(R, group):
    return R[group].sum(axis=1).mean()

def group_diam(R, group):
    return R[group].max()

def group_avg_diam(R, group):
    return R[group].max(axis=1)[0].mean()

def group_betwenness(R, group, filter):
    return R[group]*filter[group]


def group_er_matrix(R, groups):
    unique_groups = torch.unique(groups)
    H = torch.zeros((unique_groups.shape[0], unique_groups.shape[0]))
    for group1 in unique_groups:
        for group2 in unique_groups:
            H[group1.int(), group2.int()] = R[groups == group1, :][:, groups == group2].sum()
    return H

def get_group_metrics(R, group, filter):
    return {
        'total_er': group_total_er(R, group).cpu().item(),
        'diameter': group_diam(R, group).cpu().item(),
        'avg_diam': group_avg_diam(R, group).cpu().item(),
        'avg_betw': group_betwenness(R, group, filter).sum(axis=1).mean().cpu().item(),
        'std_betw': group_betwenness(R, group, filter).sum(axis=1).std().cpu().item()
    }