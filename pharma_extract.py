import os
import argparse
from rdkit import Chem
from rdkit import Geometry
from rdkit import RDConfig
from rdkit.Chem import AllChem
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.Pharm3D import Pharmacophore
import random
from collections import defaultdict

def extract_pharmacophore_features(pdb_filename, specific_residues):
    random.seed(123)
    desired_families = {'Donor', 'Acceptor', 'Hydrophobe', 'LumpedHydrophobe', 'Aromatic', 'PosIonizable'}
    feature_mapping = {
        'Donor': 'HDON',
        'Acceptor': 'HACC',
        'Hydrophobe': 'HYBL',
        'LumpedHydrophobe': 'LHYBL',
        'Aromatic': 'AROM',
        'PosIonizable': 'POSC'
    }

    # Load the molecule from the PDB file
    mol = Chem.MolFromPDBFile(pdb_filename)
    if mol is None:
        raise ValueError(f"Failed to load molecule from {pdb_filename}")
    
    # Load the feature factory
    fdef_file = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    featfact = ChemicalFeatures.BuildFeatureFactory(fdef_file)
    all_features = featfact.GetFeaturesForMol(mol)

    # Collect pharmacophore features for the specified residues
    results = defaultdict(list)
    total_sampled_features = 0
    max_features = 8

    for res_name, res_num in specific_residues:
        specific_atom_indices = []
        for atom in mol.GetAtoms():
            pdb_info = atom.GetPDBResidueInfo()
            if pdb_info and pdb_info.GetResidueNumber() == res_num:
                specific_atom_indices.append(atom.GetIdx())

        # Filter features related to the current residue
        specific_features = [feat for feat in all_features if feat.GetFamily() in desired_families and any(atom_idx in specific_atom_indices for atom_idx in feat.GetAtomIds())]

        # Calculate the number of features to sample for this residue
        remaining_features = max_features - total_sampled_features
        num_to_sample = min(remaining_features, 3, len(specific_features))

        # Randomly select the required number of features
        selected_features = random.sample(specific_features, num_to_sample)
        total_sampled_features += len(selected_features)

        for feat in selected_features:
            pos = feat.GetPos()
            mapped_family = feature_mapping.get(feat.GetFamily(), feat.GetFamily())
            results[f"{res_name} {res_num}"].append((mapped_family, pos.x, pos.y, pos.z))

        if total_sampled_features >= max_features:
            break

    return results

def main():
    parser = argparse.ArgumentParser(description='Extract pharmacophore features from specific residues in a PDB file.')
    parser.add_argument('pdb_file', type=str, help='Path to the PDB file of the target protein within the protein-protein interaction (PPI) complex.')
    parser.add_argument('residues', type=str, nargs='+', help='Residues in the format RESIDUE_NAME RESIDUE_NUMBER (e.g., LYS 7). Maximum of 3 residues allowed.')
    parser.add_argument('output_file', type=str, help='Path to the output .posp file.')

    args = parser.parse_args()

    if len(args.residues) // 2 > 3:
        raise ValueError("A maximum of 3 residues are allowed.")
    
    pdb_filename = args.pdb_file
    specific_residues = [(args.residues[i], int(args.residues[i+1])) for i in range(0, len(args.residues), 2)]
    output_file = args.output_file
    
    features = extract_pharmacophore_features(pdb_filename, specific_residues)
    
    with open(output_file, 'w') as f:
        for res, feats in features.items():
            for feat in feats:
                f.write(f"{feat[0]} {feat[1]:.3f} {feat[2]:.3f} {feat[3]:.3f}\n")

if __name__ == "__main__":
    main()
