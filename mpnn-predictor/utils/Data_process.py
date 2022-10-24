"""
Create on Sep 7,2021
@author:nanan
@Email:nanan.sun@wowjoy.cn
"""
import os
import sys
import pandas as pd
import rdkit
import torch
import networkx as nx
import numpy as np
import shutil
import rdkit.Chem as Chem
elem_list = ['H','C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V',
                  'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni',
                  'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi',
                  'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce', 'Gd', 'Ga', 'Cs', 'unknown']

#print("elem_list",len(elem_list))
Bond_type =[rdkit.Chem.rdchem.BondType.SINGLE, rdkit.Chem.rdchem.BondType.DOUBLE, rdkit.Chem.rdchem.BondType.TRIPLE, \
        rdkit.Chem.rdchem.BondType.AROMATIC]
def reader_data(Data_Path):
    Data = {}
    if Data_Path.endswith(".xlsx"):
        Infos = pd.read_excel(Data_Path)
        for idx, smile in enumerate(Infos["Smiles"].value):
            activity = Infos["Activity"].values[idx]
            try:
                mol = Chem.MolFromSmiles(smile)
                smiles = Chem.MolToSmiles(mol)
                Data.update({idx: {"smiles": smiles, "activity":activity}})
            except:
                print("Parsing incorrectly-----")
    else:
        with open(Data_Path, "r")as fr:
            for idx, line in enumerate(fr.read().splitlines()):
                lig,activity = line.split(",")[0],line.split(",")[-1]
                try:
                    mol = Chem.MolFromSmiles(lig)
                    smiles = Chem.MolToSmiles(mol)
                    Data.update({idx: {"smiles": smiles, "activity": activity}})
                except:
                    print("Parsing incorrectly---------")
    return Data
def onehot_encoding(x,allowble_list):
    if x not in allowble_list:
        x = allowble_list[-1]
    return list(map(lambda s: int(x==s),allowble_list))
def nodes_trans(g,hydrogen=False):
    h = []
    for n, d in g.nodes_iter(data = True):
        h_t = []
        h_t += onehot_encoding(d["atom_type"],elem_list)
        h_t += [int(d["atom_number"])]
        h_t += [int(d["acceptor"])]
        h_t += [int(d["donor"])]
        h_t += [int(d["aromatic"])]
        h_t += onehot_encoding(d["hybridization"],[rdkit.Chem.rdchem.HybridizationType.SP, rdkit.Chem.rdchem.HybridizationType.SP2, rdkit.Chem.rdchem.HybridizationType.SP3])
        h_t += onehot_encoding(d["ExplicitValence"],[1,2,3,4,5,6])
        h_t += onehot_encoding(d["ImplicitValence"],[0,1,2,3,4,5])
        h_t += onehot_encoding(d["Degree"],[0,1,2,3,4,5])
        if hydrogen:
            h_t.append(int(d["num_h"]))
        h.append(h_t)
    return h


def edges_trans(g,e_representation = "chem_graph"):
    remove_edges = []
    e = {}
    for n1,n2, d in g.edges_iter(data = True):
        e_t = []
        if e_representation == "chem_graph":
            if d["bond_type"] is None:
                remove_edges.append((n1,n2))
            else:
                e_t += onehot_encoding(d["bond_type"],Bond_type)
                e_t += [int(d["conjugated"])]
                e_t += [int(d["IsInring"])]
        else:
            print("Incorrect Edge representaion transform")
            quit()
        if e_t:
            e.update({(n1,n2):e_t})
    for edge in remove_edges:
        g.remove_edge(*edge)
    return nx.to_numpy_matrix(g),e

def collate_g(batch):
    """
    1.The shape of original g,h,e is [num_node,num_node],[num_node,chemical_prop],{(node_idx,node_idx):edeg_prop,(node_idx,node_idx):edge_prop} respectively

    2.batch if len(batch)=16,[((g,h,e),target),((g,h,e),target),...()],
    maxing input_i[1],that is h, has been as size selection
    """
    #print("batch",len(batch))
    #for (input_i,target_i) in batch:
    #    print("input_i",input_i[2])
    #import IPython
    #IPython.embed()
    batch_sizes = np.max(np.array([[len(input_i[1]),len(input_i[1][0]),len(input_i[2]),len(list(input_i[2].values())[0])]
                                   if input_i[2] else
                                   [len(input_i[1]),len(input_i[1][0]),0,0]
                                   for (input_i,target_i) in batch]),axis=0)
    #print("batch_size",batch_size)
    ####recreate g,h,e
    ###adjacency matrix shape
    g = np.zeros((len(batch),batch_sizes[0],batch_sizes[0]))
    ###node shape
    h = np.zeros((len(batch),batch_sizes[0],batch_sizes[1]))
    ###edge shape
    e = np.zeros((len(batch),batch_sizes[0],batch_sizes[0],batch_sizes[-1]))

    target = np.zeros((len(batch),len(batch[0][1])))
    for i in range(len(batch)):
        num_nodes = len(batch[i][0][1])
        g[i,0:num_nodes,0:num_nodes]=batch[i][0][0]
        h[i,0:num_nodes,:] = batch[i][0][1]
        ###for edge
        for edge in batch[i][0][-1].keys():
            e[i,edge[0],edge[-1],:]=batch[i][0][-1][edge]
            e[i,edge[-1],edge[0],:] = batch[i][0][-1][edge]
        target[i,:]=batch[i][-1]
    g = torch.FloatTensor(g)
    h = torch.FloatTensor(h)
    e = torch.FloatTensor(e)
    target = torch.FloatTensor(target)
    return g,h,e,target

def save_checkpoint(state, is_best, directory):

    if not os.path.isdir(directory):
        os.makedirs(directory)
    checkpoint_file = os.path.join(directory, 'checkpoint.pth')
    best_model_file = os.path.join(directory, 'model_best.pth')
    torch.save(state, checkpoint_file)
    if is_best:
        shutil.copyfile(checkpoint_file, best_model_file)








