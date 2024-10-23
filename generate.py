import argparse
import pickle
import os
import random
from pathlib import Path

import numpy as np
import torch
import dgl
import rdkit
from rdkit import RDLogger
from tqdm.auto import tqdm

from model.Hot2Mol import Hot2Mol
from utils.file_utils import load_phar_file


RDLogger.DisableLog('rdApp.*')

def load_model(model_path, tokenizer_path):
    with open(tokenizer_path+'/tokenizer_r_iso.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open(tokenizer_path+'/tokenizer_delta_qeppi.pkl', 'rb') as f:
        tokenizer_prop = pickle.load(f)

    model_params = {
        "max_len": 128,
        "pp_v_dim": 7 + 1,
        "pp_e_dim": 1,
        "pp_encoder_n_layer": 4,
        "hidden_dim": 384,
        "n_layers": 8,
        "ff_dim": 1024,
        "n_head": 8,
        'device': 'cuda:1'
    }

    model = Hot2Mol(model_params, tokenizer, tokenizer_prop)
    states = torch.load(model_path, map_location='cpu')
    print(model.load_state_dict(states['model'], strict=False))

    return model, tokenizer, tokenizer_prop

def format_smiles(smiles):
    mol = rdkit.Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    smiles = rdkit.Chem.MolToSmiles(mol, isomericSmiles=True)

    return smiles

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('input_path', type=Path, help='the input file path. If it is a directory, then every file '
                                                      'ends with `.edgep` or `.posp` will be processed')
    parser.add_argument('output_dir', type=Path, help='the output directory')
    parser.add_argument('model_path', type=Path, help='the weights file (xxx.pth)')
    parser.add_argument('tokenizer_path', type=str)

    parser.add_argument('--n_mol', type=int, default=1000, help='number of generated molecules for each '
                                                                 'pharmacophore file')
    parser.add_argument('--device', type=str, default='cpu', help='`cpu` or `cuda`')
    parser.add_argument('--filter', action='store_true', help='whether to save only the unique valid molecules')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=-1)

    args = parser.parse_args()

    if args.seed != -1:
        seed_torch(args.seed)

    if args.input_path.is_dir():
        files = list(args.input_path.glob('*.posp')) + list(args.input_path.glob('*.edgep'))
    else:
        assert args.input_path.suffix in ('.edgep', '.posp')
        files = [args.input_path]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, tokenizer_prop = load_model(args.model_path, args.tokenizer_path)
    model.eval()
    model.to(args.device)

    for file in files:
        output_path = args.output_dir / f'{file.stem}_{args.n_mol}_result.txt'

        g = load_phar_file(file)

        g_batch = [g] * args.batch_size
        g_batch = dgl.batch(g_batch).to(args.device)
        n_epoch = (args.n_mol + args.batch_size - 1) // args.batch_size
        
        res = []
        for i in tqdm(range(n_epoch)):
            res.extend(tokenizer.get_text(model.generate(g_batch)))
        res = res[:args.n_mol]

        if args.filter:
            res = [format_smiles(i) for i in res]
            res = [i for i in res if i]
            res = list(set(res))

        output_path.write_text('\n'.join(res))

    print('done')