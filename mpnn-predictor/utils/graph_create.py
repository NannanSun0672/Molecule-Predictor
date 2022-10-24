import numpy as np
import networkx as nx
import rdkit
from rdkit import RDConfig
import rdkit.Chem as Chem
from rdkit.Chem import ChemicalFeatures

import os
def Grapher_create(Data_infos,ids):
    label = [Data_infos["activity"]]
    grapher = nx.Graph(tag = ids)
    smiles = Data_infos["smiles"]
    m = Chem.MolFromSmiles(smiles)
    m = Chem.AddHs(m)
    fdef_name = os.path.join(RDConfig.RDDataDir,"BaseFeatures.fdef")
    factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
    feats = factory.GetFeaturesForMol(m)
    ####Create node
    for i in range(0,m.GetNumAtoms()):
        atom_i = m.GetAtomWithIdx(i)
        grapher.add_node(i,atom_type = atom_i.GetSymbol(),atom_number = atom_i.GetAtomicNum(),
                         acceptor = 0,donor = 0,aromatic = atom_i.GetIsAromatic(),
                         hybridization = atom_i.GetHybridization(),num_h = atom_i.GetTotalNumHs(),
                         ExplicitValence = atom_i.GetExplicitValence(),ImplicitValence = atom_i.GetImplicitValence(),
                         Degree = atom_i.GetDegree()
                         )
    for i in range(0,len(feats)):
        if feats[i].GetFamily()=="Donor":
            node_list = feats[i].GetAtomIds()
            for j in node_list:
                grapher.node[j]["donor"] = 1
        elif feats[i].GetFamily() =="Acceptor":
            node_list = feats[i].GetAtomIds()
            for j in node_list:
                grapher.node[j]["acceptor"]=1
    #### Read Egeds
    for i in  range(0,m.GetNumAtoms()):
        for j in range(0,m.GetNumAtoms()):
            e_ij = m.GetBondBetweenAtoms(i,j)
            if e_ij is not None:
                grapher.add_edge(i,j,bond_type = e_ij.GetBondType(),conjugated  = e_ij.GetIsConjugated(),IsInring = e_ij.IsInRing())
            else:
                grapher.add_edge(i,j,bond_type = None,conjugated = None,inring = None)
    #import IPython
    #IPython.embed()
    return grapher,label

if __name__ == "__main__":
    grapher_infos = {"smiles":"CCC1C2CC3CC1C23 CC[C@H]1[C@H]2C[C@H]3C[C@@H]1[C@@H]23","activity":6.4089353929735}
    idx = 1
    grapher,label = Grapher_create(grapher_infos, idx)
    print("grapher",grapher)






