import numpy as np
import networkx as nx
from typing import Optional

import torch
import torch_geometric as pyg

class GraphWrapper:
    def __init__(self,
                 edgelist: [torch.Tensor],
                 laplacian: Optional[torch.Tensor] = None,
                 pinv: Optional[torch.Tensor] = None,
                 features: Optional[torch.Tensor] = None,
                 sensitive_attr: Optional[torch.Tensor] = None,
                 labels: Optional[torch.Tensor] = None,
                 device: str = 'cpu',
                 dtype: torch.dtype = torch.float32):
        """Wrapper for a graph with a Laplacian matrix, assuming totally connected graph

        Args:
            edgelist (torch.Tensor]): _description_
            laplacian (Optional[torch.Tensor]): _description_. Defaults to None.
            pinv (Optional[torch.Tensor]): _description_. Defaults to None.
            features (Optional[torch.Tensor]): _description_. Defaults to None.
            sensitive_attr (Optional[torch.Tensor]): _description_. Defaults to None.
            device (str, optional): _description_. Defaults to 'cpu'.
            dtype (torch.dtype, optional): DTYPE of LINV and PINV, L must be 32 bigger. Defaults to torch.float32.
        """

        # ? Change devie to only count for effective resistance computation?
        # ? Then we save edge_mask, sens and features in cpu

        # ? Also, set dtype as a parameter? maybe we can use float 16

        #! inverse only works with high precission

        # Set Device
        self.device = device
        self.dtype = dtype

        # Set edgelist
        self.edgelist = edgelist
        assert self.edgelist.shape[0] == 2, f'edgelist is not of shape (2, num_edges): {self.edgelist.shape}'
        assert self.edgelist.shape[1] > 0, f'edgelist is empty: {self.edgelist.shape}'
        assert isinstance(self.edgelist, torch.Tensor), f'edgelist is not a torch.Tensor: {type(self.edgelist)}'

        if laplacian is None:
            L_ei, L_ew  = pyg.utils.get_laplacian(self.edgelist)#, ew)
            self.laplacian = pyg.utils.to_dense_adj(edge_index = L_ei, edge_attr = L_ew)[0].to(self.device)
            del L_ei
            del L_ew
        else:
            self.laplacian = laplacian.type(self.dtype).to(self.device)

        self.num_nodes = int(self.laplacian.shape[0])
        self.mode = 'exact' if self.num_nodes < 10000 else 'woodbury'
        print(self.mode)
        self.num_edges = int(self.laplacian.diagonal().sum()/2)


        if pinv is None:
            self.pinv = torch.linalg.inv(self.laplacian+(1/self.num_nodes)).type(self.dtype)

        #potentially remove to free memory and always compute on the fly->  L != 0    
        self.edge_mask = self.compute_edge_mask()

    
        self.sens = torch.Tensor(sensitive_attr).to(self.device) if not sensitive_attr is None else None
        self.features = torch.Tensor(features).to(self.device) if not features is None else None
        self.labels = torch.Tensor(labels).to(self.device) if not labels is None else None

        self.safety_checks()

    
    def compute_edge_mask(self) -> torch.Tensor:
        """
        Compute an edge mask tensor for the graph.

        Returns:
            torch.Tensor: A binary n x n tensor where the i, j entry is 1 if {i, j} IS NOT an edge in the graph
                        or if it represents a self-loop. Otherwise, it is 0.

        The edge mask helps identify which pairs of nodes are NOT connected by an edge in the graph. It is
        derived by taking the logical NOT of the Laplacian matrix, where non-zero entries represent self-loops
        or edges in the graph.

        Example:
        If there is no edge between nodes 1 and 2, the edge mask tensor will have a value of 1 at [1, 2].

        """
        return torch.logical_not(self.laplacian.bool())
    
    def get_effective_resistance(self) -> torch.Tensor:
        """ Return an n x n tensor where i,j entry is the effective resistance between i and j

        The effective resistance between i and j is the i,j entry of the inverse of the laplacian.
        """
        Linv_diag = torch.diag(self.pinv)
        return Linv_diag.unsqueeze(0) +  Linv_diag.unsqueeze(1) - 2*self.pinv


    def update_edge_mask(self, i : int, j : int):
        """ Update the edge mask after adding the edge {i,j} """
        self.edge_mask[i,j] = 0
        self.edge_mask[j,i] = 0

    def update_laplacian(self, i : int, j : int):
        """ Update the Laplacian after adding the edge {i,j} """
        self.laplacian[i,i] += 1
        self.laplacian[j,j] += 1
        self.laplacian[i,j] -= 1
        self.laplacian[j,i] -= 1

    def update_edge_list(self, i : int, j : int):
        """ Update the edge list after adding the edge {s,t} """
        self.edgelist = torch.cat([self.edgelist, torch.tensor([[i,j],[j,i]])], dim=1)

    def update_pseudo_inverse(self, i : int, j : int, mode: str = 'exact'):
        """ Update the pseudo inverse after adding the edge {i,j} """
       
        if mode == 'woodbury':
            #! Update the pseudoinverse of the Laplacian with Woodbury's formula after adding the edge (i,j)
            v = self.pinv[:,i] - self.pinv[:,j]
            effective_resistance = v[i] - v[j]
            self.pinv = self.pinv - (1/(1+effective_resistance))*torch.outer(v, v)
        elif mode == 'exact':
            self.pinv = torch.linalg.inv(self.laplacian+(1/self.num_nodes))
        else:
            raise ValueError(f'Invalid mode: {mode}')
        
    
    def is_edge(self, i : int, j : int) -> bool:
        """ Return True if the edge {i,j} exists in the graph """
        return not self.edge_mask[i,j]

    def add_link(self, i, j, mode: str = 'exact'):
        if self.is_edge(i,j):
            raise ValueError(f'Edge {i,j} already exists')
        
        self.update_edge_mask(i, j)
        self.update_edge_list(i, j)
        self.update_laplacian(i, j)
        self.update_pseudo_inverse(i, j, mode=self.mode)
        self.num_edges += 1
        #update_pinv(i,j)

        #self.safety_checks()



    def safety_checks(self) -> bool:
        """ Check that the graph is connected, undirected, has no self loops,
            the number of links on edge_list is the same as the number of edges on the laplacian
        """
        assert self.laplacian.shape[0] == self.laplacian.shape[1], f'L not Square: {self.laplacian.shape[0]} != {self.laplacian.shape[1]}'
        assert self.laplacian.shape[0] == self.num_nodes, f'Mismatching number of nodes: {self.laplacian.shape[0]} != {self.num_nodes}'
        assert np.unique(self.edgelist.ravel()).shape[0] == self.num_nodes, f'Mismatching number of nodes: {np.unique(self.edgelist.ravel()).shape} != {self.num_nodes}'
        assert (torch.logical_not(self.laplacian.bool()) == self.edge_mask).all(), f'edge_mask is not the logical not of the laplacian'
        assert self.num_edges == ((~self.edge_mask).sum().item()-self.num_nodes)/2, f'Mismatching number of edges: {self.num_edges} != {((~self.edge_mask).sum().item()-self.num_nodes)/2}'
        #Check edges are symmetric
        assert np.allclose(self.laplacian.detach().cpu(), self.laplacian.detach().cpu().T), f'laplacian is not symmetric'
        assert self.laplacian.detach().cpu().diagonal().sum() == self.edgelist.shape[1], f'Mismatching edges: {self.laplacian.diagonal().sum()} != {self.edgelist.shape[1]}'
        assert self.laplacian.detach().cpu().diagonal().sum()//2 == self.num_edges, f'Mismatching edges: {self.laplacian.diagonal().sum()} != {self.num_edges}'
        if not self.sens is None:
            assert self.sens.shape[0] == self.num_nodes, f'Mismatching number of S and nodes: {self.sens.shape[0]} != {self.num_nodes}'
        if not self.features is None:
            assert self.features.shape[0] == self.num_nodes, f'Mismatching number of F and nodes: {self.features.shape[0]} != {self.num_nodes}'
        return True