
# Hot2Mol



## Overview

This repository contains the PyTorch implementation of *Designing target-specific PPI inhibitors with hot-spot-guided deep generative model*. 

Protein-protein interactions (PPIs) play crucial roles in cellular functions and represent compelling targets for drug discovery. However, developing effective small molecule inhibitors for PPIs is challenging due to their flat and wide interfaces. To address this challenge, we propose Hot2Mol, a deep learning framework designed to generate PPI inhibitors by mimicking the pharmacophores of hot-spot residues, thereby achieving high affinity and selectivity. Hot2Mol utilizes E(n)-equivariant graph neural networks to accurately encode 3D molecular structures and pharmacophore patterns. A conditional transformer is used to generate molecules while optimizing drug-like properties. 


![model_framework.png](pics%2Fmodel_framework.png)

## Requirements
- python==3.8
- torch==1.12.1+cu116
- rdkit==2023.9.2
- dgl-cuda11.1=0.9.1
- fairseq==0.10.2
- numpy==1.23.0
- pandas==2.0.3
- tqdm==4.65.0
- einops==0.7.0


### Creating a new environment in conda
We recommend using `conda` to manage the environment. 

```bash
conda env create -f environment.yml
conda activate Hot2Mol
```

## Training

The training process with default parameters requires a GPU card with at least 10GB of memory.

Run `train.py` using the following command:
```bash
python train.py <output_dir> --device cuda:0 --show_progressbar
```
- the `output_dir` is the directory you want to store the trained model



## Using a trained Hot2Mol model to generate molecules


### Prepare the pharmacophore hypotheses

Hot2Mol only requires a pharmacophore hypothesis as input. The hypothesis can be constructed by sampling pharmacophores from hot-spot residues at PPI interfaces. The hot-spot residues may be computed using docking methods like HawkDock, or obatained from literatures.


First of all, you need some pharmacophore hypotheses. A pharmacophore is defined as a set of chemical features and their spatial information that is necessary for a drug to bind to a target and there are many ways to acquire one. 

If you have a biochemistry background, we strongly encourage you to build it yourself by stacking active ligands or analyzing the receptor structure. There are also many tools available. 
And you can always adjust the input hypothesis according to the results.

Apart from building it yourself, you can also acquire them by searching the literature or just randomly sampling 3-6 pharmacophore elements from a reference ligand to build some hypotheses and filtering the generated molecules afterwards.


### Format the hypotheses

The pharmacophore hypotheses need to be converted to a fully-connected graph and should be provided in one of the two formats:

- the `.posp` format where the type of the pharmacophore points and the 3d positions are provided, see `data/IL-2:IL-2R.posp` for example. 

**Pharmacophore types** supported by default:
- AROM: aromatic ring
- POSC: cation
- HACC: hydrogen bond acceptor
- HDON: hydrogen bond donor
- HYBL: hydrophobic group (ring)
- LHYBL: hydrophobic group (non-ring)

The 3d position in `.posp` files will first be used to calculate the Euclidean distances between each point and then the distances will be mapped to the shortest-path-based distances.


### Generate

Use the `generate.py` to generate molecules.

usage:
```text
python generate.py [-h] [--n_mol N_MOL] [--device DEVICE] [--filter] [--batch_size BATCH_SIZE] [--seed SEED] input_path output_dir model_path tokenizer_path

positional arguments:
  input_path            the input file path. If it is a directory, then every file ends with `.posp` will be processed
  output_dir            the output directory
  model_path            the weights file (xxx.pth)
  tokenizer_path        the saved tokenizers (tokenizer_r_iso.pkl, tokenizer_delta_qeppi.pkl)

optional arguments:
  -h, --help            show this help message and exit
  --n_mol N_MOL         number of generated molecules for each pharmacophore file
  --device DEVICE       `cpu` or `cuda`, default:'cpu'
  --filter              whether to save only the unique valid molecules
  --batch_size BATCH_SIZE
  --seed SEED
```

The output is a `.txt` file containing the generated SMILES. It takes about 30 seconds to generate 10,000 molecules using a single 2080Ti, and about 10 minutes if using CPUs.

To run generation on the demo input:
```bash
python generate.py ./data/IL-2:IL-2R.posp ./results ./pretrained_model/epoch32.pth ./pretrained_model --filter --device cuda:0 --seed 123
```

**We provide the weights file acquired using `train.py` in the [release page](https://github.com/sun-heqi/Hot2Mol/releases/tag/v1.0).** Please unzip it in the root directory.

**The current model only support a maximum of 8 pharmacophore points in a single hypotheis.** If you want to increase the maximum number, a possible way is to re-train the model with increased number of randomly selected pharmacophore elements and a larger `MAX_NUM_PP_GRAPHS`.


## Acknowledgements
This implementation is inspired and partially based on earlier works [1], [2]. Thanks for giving us inspirations!


## References

* [1] Zhu, Huimin, et al. "A pharmacophore-guided deep learning approach for bioactive molecular generation." Nature Communications 14.1 (2023): 6234.
    
* [2] Yoshida, Shuhei, et al. "Peptide-to-small molecule: a pharmacophore-guided small molecule lead generation strategy from high-affinity macrocyclic peptides." Journal of Medicinal Chemistry 65.15 (2022): 10655-10673.   

