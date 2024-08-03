
# Hot2Mol


This is a repository of our paper "Designing target-specific PPI inhibitors with hot-spot-guided deep generative model". 

Protein-protein interactions (PPIs) play crucial roles in cellular functions and represent compelling targets for drug discovery. However, developing effective small molecule inhibitors for PPIs is challenging due to their flat and wide interfaces. To address this challenge, we propose Hot2Mol, a deep learning framework designed to generate PPI inhibitors by mimicking the pharmacophores of hot-spot residues, thereby achieving high affinity and selectivity.


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

```bash
conda env create -f environment.yml
conda activate Hot2Mol
```

## Training

Run `train.py` using the following command:
```bash
python train.py <output_dir> --device cuda:0 --show_progressbar
```



## Using a trained Hot2Mol model to generate molecules


### Prepare the pharmacophore hypotheses

Hot2Mol requires a pharmacophore hypothesis as input. Construct the hypothesis by sampling pharmacophores from hot-spot residues on the target protein of the PPI complex. Hot-spot residues can be computed using docking methods like [HawkDock](http://cadd.zju.edu.cn/hawkdock/), or obatained from literatures.

A pharmacophore hypothesis should be provided in `.posp` format, which includes the type of pharmacophore feature in the first column and 3D coordinates in the last three columns. See `data/IL-2:IL-2R.posp` for an example.

**Supported pharmacophore types**:
- AROM: aromatic ring
- POSC: cation
- HACC: hydrogen bond acceptor
- HDON: hydrogen bond donor
- HYBL: hydrophobic group (ring)
- LHYBL: hydrophobic group (non-ring)


### Build the hypotheses

Use the `pharma_extract.py` to generate the pharmacophore hypothesis.

usage:
```text
python pharma_extract.py [-h] pdb_file residues output_file

positional arguments:
  pdb_file            Path to the PDB file of the target protein within the PPI complex.
  residues            Residues in the format RESIDUE_NAME RESIDUE_NUMBER (e.g., LYS 7).
  output_file         Path to the output .posp file.
```

The output is a `.posp` file containing the pharmacophore hypotesis. 

**Example**

To build pharmacophore hypothesis for the demo input:
```bash
python pharma_extract.py data/pdbfile.pdb LEU 48 ILE 19 ALA 14 IL-2:IL-2R.posp
```

### Generate

Use `generate.py` to generate molecules.

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

The output is a `.txt` file containing the generated SMILES strings.

**Example**

To run generation on the demo input:
```bash
python generate.py ./data/IL-2:IL-2R.posp ./results ./pretrained_model/epoch32.pth ./pretrained_model --filter --device cuda:0 --seed 123
```

**Note:** The weights file acquired using `train.py` is available on the [release page](https://github.com/sun-heqi/Hot2Mol/releases/tag/v1.0).

The current model supports a maximum of 8 pharmacophore features in a single hypothesis. If you wish to increase this limit, you can retrain the model with a higher number of randomly selected pharmacophore elements and a larger `MAX_NUM_PP_GRAPHS`.


## Acknowledgements
This implementation is inspired and partially based on earlier works [1], [2]. Thank you for the inspiration!


## References

* [1] Zhu, Huimin, et al. "A pharmacophore-guided deep learning approach for bioactive molecular generation." Nature Communications 14.1 (2023): 6234.
    
* [2] Yoshida, Shuhei, et al. "Peptide-to-small molecule: a pharmacophore-guided small molecule lead generation strategy from high-affinity macrocyclic peptides." Journal of Medicinal Chemistry 65.15 (2022): 10655-10673.   

