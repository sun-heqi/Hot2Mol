import re
from typing import List

import dgl
import numpy as np
import torch
from rdkit import Chem
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem

from utils.smiles2ppgraph import smiles2ppgraph
from ogb.utils.features import atom_to_feature_vector
from ipdb import set_trace


MAX_NUM_PP_GRAPHS = 8 


class Tokenizer:
    NUM_RESERVED_TOKENS = 32
    SPECIAL_TOKENS = ('<sos>', '<eos>', '<pad>', '<mask>', '<sep>', '<unk>')
    SPECIAL_TOKENS += tuple([f'<t_{i}>' for i in range(len(SPECIAL_TOKENS), 32)])  # saved for future use

    PATTEN = re.compile(r'\[[^\]]+\]'
                        # only some B|C|N|O|P|S|F|Cl|Br|I atoms can omit square brackets
                        r'|B[r]?|C[l]?|N|O|P|S|F|I'
                        r'|[bcnops]'
                        r'|@@|@'
                        r'|%\d{2}'
                        r'|.')
    
    ATOM_PATTEN = re.compile(r'\[[^\]]+\]'
                             r'|B[r]?|C[l]?|N|O|P|S|F|I'
                             r'|[bcnops]')
    
    @staticmethod
    def gen_vocabs(smiles_list):
        smiles_set = set(smiles_list)
        vocabs = set()

        for a in tqdm(smiles_set):
            vocabs.update(re.findall(Tokenizer.PATTEN, a))

        special_tokens = list(Tokenizer.SPECIAL_TOKENS)
        vocabs = special_tokens + sorted(vocabs - set(special_tokens), key=lambda x: (len(x), x)) + list(vocabs)

        return vocabs

    @staticmethod
    def gen_prop_change(property_condition):
        property_condition = [f'QEPPI_change_{item}' for item in property_condition]
        vocabs = list(set(property_condition))
        
        return vocabs

    def __init__(self, vocabs):
        self.vocabs = vocabs
        self.i2s = {i: s for i, s in enumerate(vocabs)}
        self.s2i = {s: i for i, s in self.i2s.items()}

    def __len__(self):
        return len(self.vocabs)
    
    def parse(self, smiles, return_atom_idx=False):
        l = []
        if return_atom_idx:
            atom_idx=[]
        for i, s in enumerate(('<sos>', *re.findall(Tokenizer.PATTEN, smiles), '<eos>')):
            if s not in self.s2i:
                a = 3  
            else:
                a = self.s2i[s]
            l.append(a)
            
            if return_atom_idx and re.fullmatch(Tokenizer.ATOM_PATTEN, s) is not None:
                atom_idx.append(i)
        if return_atom_idx:
            return l, atom_idx
        return l

    def get_text(self, predictions):
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.tolist()

        smiles = []
        for p in predictions:
            s = []
            for i in p:
                c = self.i2s[i]
                if c == '<eos>':
                    break
                s.append(c)
            smiles.append(''.join(s))

        return smiles
    
    def print_all_tokens(self):
        for token in self.vocabs:
            print(token)


def run_test_tokenizer():
    smiles = ['CCNC(=O)NInc1%225cpppcc2nc@@nc(N@c3ccc(O[C@@H+5]c4cccc(F)c4)c(Cl)c3)c2c1']
    tokenizer = Tokenizer(Tokenizer.gen_vocabs(smiles))
    print(tokenizer.parse(smiles[0]))
    print(tokenizer.get_text([tokenizer.parse(smiles[0])]))


def get_random_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)

    # clear isotope
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))

    rsmiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=False, doRandom=True)

    return rsmiles


def construct_input_graph(mol):
    """
    Constructs an 3D graph from input molecule.

    Parameters:
    - mol: A molecule object from RDKit.

    Returns:
    - A DGL graph containing atom features, edge distances, and atom coordinates.
    """
    # Add hydrogen atoms and perform 3D embedding
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)

    conf = mol.GetConformer()
    positions = conf.GetPositions()
    atom_coords = torch.tensor(positions, dtype=torch.float)

    # Build a list of atom features
    atom_features_list = [atom_to_feature_vector(atom) for atom in mol.GetAtoms()]
    atom_features = torch.tensor(np.array(atom_features_list), dtype=torch.float)

    # Build features and connections for edges
    bond_features = []
    src_nodes = []
    dst_nodes = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        distance = torch.norm(torch.tensor(positions[start]) - torch.tensor(positions[end]), p=2)
        bond_features.append(distance)
        src_nodes += [start, end]
        dst_nodes += [end, start]
        bond_features.append(distance)  # Add the same feature for the reverse direction edge
    
    # Create a DGL graph
    input_graph = dgl.graph((src_nodes, dst_nodes))
    input_graph.ndata['h'] = atom_features
    input_graph.edata['dist'] = torch.tensor(bond_features).unsqueeze(1)
    input_graph.ndata['coord'] = atom_coords

    return input_graph


def build_intervals(delta_QEPPI, step=0.1):
    min_val, max_val = min(delta_QEPPI), max(delta_QEPPI)

    start_map_interval = {}
    interval_str = '({}, {}]'.format(round(-step/2, 2), round(step/2, 2))
    intervals = [interval_str]
    start_map_interval[-step/2] = interval_str

    positives = step/2
    while positives < max_val:
        interval_str = '({}, {}]'.format(round(positives, 2), round(positives+step, 2))
        intervals.append(interval_str)
        start_map_interval[positives] = interval_str
        positives += step
    interval_str = '({}, inf]'.format(round(positives, 2))
    intervals.append(interval_str)
    start_map_interval[float('inf')] = interval_str

    negatives = -step/2
    while negatives > min_val:
        interval_str = '({}, {}]'.format(round(negatives-step, 2), round(negatives, 2))
        intervals.append(interval_str)
        negatives -= step
        start_map_interval[negatives] = interval_str
    interval_str = '(-inf, {}]'.format(round(negatives, 2))
    intervals.append(interval_str)
    start_map_interval[float('-inf')] = interval_str

    return intervals, start_map_interval


def encode_delta(value, tokenizer):
    intervals_processed = []
    for interval in tokenizer.vocabs:
        if not interval.startswith('QEPPI_change_'):
            continue
        bounds_str = interval.replace('QEPPI_change_(', '').rstrip(']')
        lower_bound, upper_bound = bounds_str.split(', ')
        lower_bound = float('-inf') if lower_bound == '-inf' else float(lower_bound)
        upper_bound = float('inf') if upper_bound == 'inf' else float(upper_bound)
        intervals_processed.append((lower_bound, upper_bound, interval))
    
    for lower_bound, upper_bound, interval in intervals_processed:
        if lower_bound < value <= upper_bound:
            return interval
    
    return None


def encode_delta_onehot(delta_encoded, tokenizer):
    one_hot_encoding = np.zeros(len(tokenizer.vocabs))

    if delta_encoded is None or delta_encoded not in tokenizer.s2i:
        return one_hot_encoding

    index = tokenizer.s2i[delta_encoded]
    one_hot_encoding[index] = 1

    return one_hot_encoding.tolist()


class SemiSmilesDataset(Dataset):

    def __init__(self, data, tokenizer: Tokenizer, tokenizer_prop: Tokenizer,
                 use_random_input_smiles=False, use_random_target_smiles=False, rsmiles=None):

        super().__init__()
        
        self.src_smiles_list = data['Source_Mol'].tolist()
        self.tar_smiles_list = data['Target_Mol'].tolist()
        self.prop_change = data['Delta_QEPPI'].tolist()  

        self.tokenizer_prop = tokenizer_prop
        self.tokenizer = tokenizer
        self.mask_token = tokenizer.SPECIAL_TOKENS.index('<mask>')

        self.vocab_size = len(tokenizer)
        self.len = len(data)
        
        self.use_random_input_smiles = use_random_input_smiles
        self.use_random_target_smiles = use_random_target_smiles
        self.rsmiles = rsmiles
        
        if rsmiles is None and (use_random_input_smiles or use_random_target_smiles):
            print('WARNING: The result of rdkit.Chem.MolToSmiles(..., doRandom=True) is NOT reproducible '
                  'because this function does not provide a way to control its random seed.')

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        src_smiles = self.src_smiles_list[item]
        src_mol = Chem.MolFromSmiles(src_smiles)
        tar_smiles = self.tar_smiles_list[item]
        tar_mol = Chem.MolFromSmiles(tar_smiles)
        
        # clear isotope
        for atom in src_mol.GetAtoms():
            atom.SetIsotope(0)
        src_mol = Chem.MolFromSmiles(Chem.MolToSmiles(src_mol))
        for atom in tar_mol.GetAtoms():
            atom.SetIsotope(0)
        tar_mol = Chem.MolFromSmiles(Chem.MolToSmiles(tar_mol))
        
        src_csmiles = Chem.MolToSmiles(src_mol, isomericSmiles=False, canonical=True, doRandom=False)
        tar_csmiles = Chem.MolToSmiles(tar_mol, isomericSmiles=False, canonical=True, doRandom=False)
        
        src_rsmiles = Chem.MolToSmiles(src_mol, isomericSmiles=False, canonical=False, doRandom=True)
        tar_rsmiles = Chem.MolToSmiles(tar_mol, isomericSmiles=False, canonical=False, doRandom=True)
        
        src_smiles = src_rsmiles if self.use_random_input_smiles else src_csmiles
        tar_smiles = tar_rsmiles if self.use_random_target_smiles else tar_csmiles
        
        src_seq = self.tokenizer.parse(src_smiles)
        src_seq = torch.LongTensor(src_seq)
        tar_seq, atom_idx = self.tokenizer.parse(tar_smiles, return_atom_idx=True)
        tar_seq = torch.LongTensor(tar_seq)

        # construct 3D molecule graph
        input_graph = construct_input_graph(src_mol)

        # construct 3D pharmacophore graph
        pp_graph, mapping = smiles2ppgraph(tar_smiles)
        pp_graph.ndata['h'] = \
            torch.cat((pp_graph.ndata['type'], pp_graph.ndata['size'].reshape(-1, 1)), dim=1).float()
        pp_graph.edata['h'] = pp_graph.edata['dist'].reshape(-1, 1).float()

        # encode delta into one-hot encodings
        delta = self.prop_change[item]
        delta_encoded = encode_delta(delta, self.tokenizer_prop)
        delta_onehot =  encode_delta_onehot(delta_encoded, self.tokenizer_prop)

        mapping = torch.FloatTensor(mapping)
        mapping[:,pp_graph.num_nodes():] = -100  # torch cross entropy loss ignores -100 by default
        
        mapping_ = torch.ones(tar_seq.shape[0], MAX_NUM_PP_GRAPHS)*-100
        mapping_[atom_idx,:] = mapping

        return src_seq, input_graph, pp_graph, mapping_, tar_seq, delta_onehot

    @staticmethod
    def collate_fn(batch):
        pad_token = Tokenizer.SPECIAL_TOKENS.index('<pad>')

        src_seqs, src_graphs, pp_graphs, mappings, tar_seqs, delta_onehot, *other_descriptors = list(zip(*batch))

        src_seqs = \
            pad_sequence(src_seqs, batch_first=True, padding_value=pad_token)
        input_mask = (src_seqs==pad_token).bool()
        
        src_graphs = dgl.batch(src_graphs)
        pp_graphs = dgl.batch(pp_graphs)

        mappings = pad_sequence(mappings, batch_first=True, padding_value=-100)  # torch cross entropy loss ignores -100 by default, but we do not use cross_entropy_loss acctually
        
        tar_seqs = pad_sequence(tar_seqs, batch_first=True, padding_value=pad_token)

        return src_seqs, src_graphs, input_mask, pp_graphs, mappings, tar_seqs, torch.tensor(delta_onehot)


if __name__ == '__main__':
    run_test_tokenizer()
