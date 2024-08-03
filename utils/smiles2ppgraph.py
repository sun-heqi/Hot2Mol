import os
import random

import dgl
import numpy as np
import torch
from rdkit import Chem
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures, AllChem
from ipdb import set_trace
from dgl import DGLError

MAX_NUM_PP_GRAPHS = 8

def sample_probability(elment_array, plist, N):
    Psample = []
    n = len(plist)
    index = int(random.random() * n)
    mw = max(plist)
    beta = 0.0
    for i in range(N):
        beta = beta + random.random() * 2.0 * mw
        while beta > plist[index]:
            beta = beta - plist[index]
            index = (index + 1) % n
        Psample.append(elment_array[index])

    return Psample


def six_encoding(atom):
    # actually seven
    orgin_phco = [0, 0, 0, 0, 0, 0, 0, 0]
    for j in atom:
        orgin_phco[j] = 1
    return torch.HalfTensor(orgin_phco[1:])


def cal_euclidean_dist(start_phco, end_phco):
    start_coord = np.array(start_phco[2:])
    end_coord = np.array(end_phco[2:])
    dist = np.linalg.norm(start_coord - end_coord)
    return dist

def smiles_code_(smiles, g, e_list):
    smiles = smiles
    dgl = g
    e_elment = e_list
    mol = AllChem.MolFromSmiles(smiles)
    atom_num = mol.GetNumAtoms()
    atom_index_list = []
    smiles_code = np.zeros((atom_num, MAX_NUM_PP_GRAPHS))
    for elment_i in range(len(e_elment)):  ##定位这个元素在第几个药效团
        elment = e_elment[elment_i]
        for e_i in range(len(elment)):
            e_index = elment[e_i]
            for atom in mol.GetAtoms():  ##定位这个原子在分子中的索引
                if e_index == atom.GetIdx():
                    list_ = ((dgl.ndata['type'])[elment_i]).tolist()
                    for list_i in range(len(list_)):
                        if list_[list_i] == 1:
                            smiles_code[atom.GetIdx(), elment_i] = 1.0
    return smiles_code


def smiles2ppgraph(smiles:str):
    '''
    :param smiles: a molecule
    :return: (pp_graph, mapping)
        pp_graph: DGLGraph, the corresponding **random** pharmacophore graph
        mapping: np.Array ((atom_num, MAX_NUM_PP_GRAPHS)) the mapping between atoms and pharmacophore features
    '''

    mol = AllChem.MolFromSmiles(smiles)
    smiles = Chem.MolToSmiles(mol)
    mol = AllChem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    atom_index_list = []
    pharmocophore_all = []

    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    feats = factory.GetFeaturesForMol(mol)
    for f in feats:
        phar = f.GetFamily()
        pos = f.GetPos()
        atom_index = f.GetAtomIds()
        atom_index = tuple(sorted(atom_index))
        atom_type = f.GetType()
        mapping = {'Aromatic': 1, 'Hydrophobe': 2, 'PosIonizable': 3,
                   'Acceptor': 4, 'Donor': 5, 'LumpedHydrophobe': 6}
        phar_index = mapping.setdefault(phar, 7)
        # pharmocophore_ = [phar_index, atom_index]  # some pharmacophore feature
        pharmocophore_ = [phar_index, atom_index, pos.x, pos.y, pos.z]  # some pharmacophore feature
        pharmocophore_all.append(pharmocophore_)  # all pharmacophore features within a molecule
        atom_index_list.append(atom_index)  # atom indices of one pharmacophore feature
    random.shuffle(pharmocophore_all)
    num = [3, 4, 5, 6, 7]
    num_p = [0.086, 0.0864, 0.389, 0.495, 0.0273]  # P(Number of Pharmacophore points)
    num_ = sample_probability(num, num_p, 1)

    type_list = []
    size_ = []
    
    ## The randomly generated clusters are obtained,
    # and the next step is to perform a preliminary merging of these randomly generated clusters with identical elements
    if len(pharmocophore_all) >= int(num_[0]):
        mol_phco = pharmocophore_all[:int(num_[0])]
    else:
        mol_phco = pharmocophore_all

    for pharmocophore_all_i in range(len(mol_phco)):
        for pharmocophore_all_j in range(len(mol_phco)):
            if mol_phco[pharmocophore_all_i][1] == mol_phco[pharmocophore_all_j][1] \
                    and mol_phco[pharmocophore_all_i][0] != mol_phco[pharmocophore_all_j][0]:
                index_ = [min(mol_phco[pharmocophore_all_i][0], mol_phco[pharmocophore_all_j][0]),
                          max(mol_phco[pharmocophore_all_i][0], mol_phco[pharmocophore_all_j][0])]
                mol_phco[pharmocophore_all_j] = [index_, mol_phco[pharmocophore_all_i][1], \
                                                 mol_phco[pharmocophore_all_i][2], mol_phco[pharmocophore_all_i][3], \
                                                 mol_phco[pharmocophore_all_i][4]]
                mol_phco[pharmocophore_all_i] = [index_, mol_phco[pharmocophore_all_i][1], \
                                                 mol_phco[pharmocophore_all_i][2], mol_phco[pharmocophore_all_i][3], \
                                                 mol_phco[pharmocophore_all_i][4]]
            else:
                index_ = mol_phco[pharmocophore_all_i][0]
    unique_index_filter = []
    unique_index = []
    for mol_phco_candidate_single in mol_phco:
        if mol_phco_candidate_single not in unique_index:
            if type(mol_phco[0]) == list:
                unique_index.append(mol_phco_candidate_single)
            else:
                unique_index.append([[mol_phco_candidate_single[0]], mol_phco_candidate_single[1]])
    for unique_index_single in unique_index:
        if unique_index_single not in unique_index_filter:
            unique_index_filter.append(unique_index_single)  ## The following is the order of the pharmacophores by atomic number
    sort_index_list = []
    for unique_index_filter_i in unique_index_filter:  ## Collect the mean of the participating elements
        sort_index = sum(unique_index_filter_i[1]) / len(unique_index_filter_i[1])
        sort_index_list.append(sort_index)
    sorted_id = sorted(range(len(sort_index_list)), key=lambda k: sort_index_list[k])
    unique_index_filter_sort = []
    for index_id in sorted_id:
        unique_index_filter_sort.append(unique_index_filter[index_id])

    position_matrix = np.zeros((len(unique_index_filter_sort), len(unique_index_filter_sort)))
    e_list = []
    for mol_phco_i in range(len(unique_index_filter_sort)):
        mol_phco_i_elment = list(unique_index_filter_sort[mol_phco_i][1])
        if type(unique_index_filter_sort[mol_phco_i][0]) == list:
            type_list.append(six_encoding(unique_index_filter_sort[mol_phco_i][0]))
        else:
            type_list.append(six_encoding([unique_index_filter_sort[mol_phco_i][0]]))
        size_.append(len(mol_phco_i_elment))
        e_list.append(mol_phco_i_elment)
        for mol_phco_j in range(len(unique_index_filter_sort)):
            mol_phco_j_elment = list(unique_index_filter_sort[mol_phco_j][1])
            if mol_phco_i_elment == mol_phco_j_elment:
                position_matrix[mol_phco_i, mol_phco_j] = 0
            else:
                # directly calculate distance between pharmacophores
                position_matrix[mol_phco_i, mol_phco_j] =  cal_euclidean_dist(unique_index_filter_sort[mol_phco_i], unique_index_filter_sort[mol_phco_j])

    weights = []
    u_list = []
    v_list = []
    phco_single = []
    for u in range(position_matrix.shape[0]):
        for v in range(position_matrix.shape[1]):
            if u != v:
                u_list.append(u)
                v_list.append(v)
                if position_matrix[u, v] >= position_matrix[v, u]:
                    weights.append(position_matrix[v, u])
                else:
                    weights.append(position_matrix[u, v])

    g = dgl.DGLGraph()
    g.add_nodes(len(unique_index_filter_sort))
    coords = np.array([x[2:] for x in unique_index_filter_sort])
    g.ndata['coord'] = torch.tensor(coords, dtype=torch.float32)
    u_list_tensor = torch.tensor(u_list)
    v_list_tensor = torch.tensor(v_list)
    g.add_edges(u_list_tensor, v_list_tensor)
    g.edata['dist'] = torch.HalfTensor(weights)
    type_list_tensor = torch.stack(type_list)    
    g.ndata['type'] = type_list_tensor
    g.ndata['size'] = torch.HalfTensor(size_)
    smiles_code_res = smiles_code_(smiles, g, e_list)

    return g, smiles_code_res
