import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from ipdb import set_trace


def index_sum(agg_size, source, idx, cuda):
    """
        source is N x hid_dim [float]
        idx    is N           [int]
        
        Sums the rows source[.] with the same idx[.];
    """
    tmp = torch.zeros((agg_size, source.shape[1]))
    tmp = tmp.cuda() if cuda else tmp
    res = torch.index_add(tmp, 0, idx, source)
    return res


class EGNN_layer(nn.Module):
    def __init__(self, in_dim, hid_dim, batch_norm, cuda=True):
        super().__init__()
        self.hid_dim=hid_dim
        self.cuda = cuda
        self.batch_norm = batch_norm
        self.bn_node_h = nn.BatchNorm1d(hid_dim)
        
        # computes messages based on hidden representations -> [0, 1]
        self.f_e = nn.Sequential(
            nn.Linear(in_dim*2+1, hid_dim), nn.SiLU(),
            nn.Linear(hid_dim, hid_dim), nn.SiLU())
        
        # preducts "soft" edges based on messages 
        self.f_inf = nn.Sequential( 
            nn.Linear(hid_dim, 1),
            nn.Sigmoid()) 
        
        # updates hidden representations -> [0, 1]
        self.f_h = nn.Sequential(
            nn.Linear(in_dim+hid_dim, hid_dim), nn.SiLU(),
            nn.Linear(hid_dim, hid_dim))
    
    def forward(self, h, x, e):
        if self.batch_norm:
            h = self.bn_node_h(h)  # batch normalization

        e_st, e_end = e[0], e[1]
        dists = torch.norm(x[e_st] - x[e_end], dim=1).reshape(-1, 1)
        
        # compute messages
        tmp = torch.hstack([h[e_st], h[e_end], dists])
        m_ij = self.f_e(tmp)
        
        # predict edges
        e_ij = self.f_inf(m_ij)
        
        # average e_ij-weighted messages  
        # m_i is num_nodes x hid_dim
        m_i = index_sum(h.shape[0], e_ij*m_ij, e[0], self.cuda)
        
        # update hidden representations
        h += self.f_h(torch.hstack([h, m_i]))

        return h, e
    

class EGNN(nn.Module):
    def __init__(self, in_dim, hid_dim, n_layers, batch_norm, cuda=True):
        super().__init__()
        self.hid_dim=hid_dim
        
        self.emb = nn.Linear(in_dim, hid_dim) 
        self.layers = nn.ModuleList([EGNN_layer(hid_dim, hid_dim,batch_norm, cuda=cuda) for _ in range(n_layers)])

        if cuda: self.cuda()
        self.cuda = cuda
    
    def forward(self, h, x, e):
        h = self.emb(h)

        for conv in self.layers:
            h, e = conv(h, x, e)
        
        return h, e