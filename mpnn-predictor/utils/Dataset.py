"""
Create on Sep 7,2021

@Email:nannan.sun@wowjoy.cn
"""
import networkx as nx
import torch
import torch.utils.data as data
import numpy as np
import argparse
import time
import os
import sys
import pandas as pd
from .graph_create import Grapher_create
from .Data_process import nodes_trans,edges_trans

class dataset(data.Dataset):
    def __init__(self,Data,ids,vertex_transform = nodes_trans,edge_transform = edges_trans,target_transform = None,
                 e_representation = "chem_graph"):

        self.Data = Data
        self.ids = ids
        self.vertex_transform = vertex_transform
        self.edge_transform = edge_transform
        self.target_transform = target_transform
        self.e_representation = e_representation
    def __getitem__(self, item):
       # print(item)
        ids = self.ids[item]
        data = self.Data[ids]
        g,target = Grapher_create(data,ids)
        if self.vertex_transform is not None:
            h = self.vertex_transform(g)
            #print("h",len(h))
        if self.edge_transform is not None:
            g,e = self.edge_transform(g,self.e_representation)
            #print("g",g.shape)
        return (g,h,e),target
    def __len__(self):
        return (len(self.ids))









