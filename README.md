
# Hot2Mol


This is a repository of our paper "De novo design of drug-like protein-protein interaction inhibitors using hot-spot-guided generative deep learning". 

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
Installation typically takes within 5 minutes.

```bash
conda env create -f environment.yml
conda activate Hot2Mol
```

## Training

Unzip the `data.zip` folder to get the training and validation data.

Run `train.py` using the following command:
```bash
python train.py <output_dir> --device cuda:0 --show_progressbar
```



## Use a pre-trained Hot2Mol model to generate molecules

### Prepare model weights

Download the model files from the [release page](https://github.com/sun-heqi/Hot2Mol/releases/tag/v1.0).
Save to `Hot2Mol/pretrained_model` folder


### Prepare pharmacophore hypotheses

Hot2Mol requires a pharmacophore hypothesis of "hot-spot" residues as input. Construct the hypothesis by sampling pharmacophores from the top-3 hot-spot residues identified using docking methods like [HawkDock](http://cadd.zju.edu.cn/hawkdock/), which computes residue-wise binding energy with MM/GBSA. Hot-spot residues can also be obtained from relevant literature.

A pharmacophore hypothesis should be provided in `.posp` format, which includes the type of pharmacophore feature in the first column and spatial coordinates in the last three columns. See `data/1z92_IL2R.posp` for an example.

**Supported pharmacophore types**:
- AROM: aromatic ring
- POSC: cation
- HACC: hydrogen bond acceptor
- HDON: hydrogen bond donor
- HYBL: hydrophobic group (ring)
- LHYBL: hydrophobic group (non-ring)


### Build the hypotheses

Use the `pharma_extract.py` to generate a pharmacophore hypothesis for hot-spot residues. 

usage:
```text
python pharma_extract.py [-h] pdb_file residues output_file

positional arguments:
  pdb_file            Path to the PDB file of the PPI chain that includes the selected hot-spot residues.
  residues            List of residues in the format RESIDUE_NAME RESIDUE_NUMBER (e.g., LYS 7)
  output_file         Path to the output .posp file.
```

The output is a `.posp` file containing the pharmacophore hypotesis. 


To build pharmacophore hypothesis for the demo input:
```bash
python pharma_extract.py data/1z92_IL2R.pdb ARG 36 LEU 42 HIE 120 data/1z92_IL2R.posp
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

The output is a `.txt` file containing the generated SMILES strings. Using a single 3090Ti, it takes approximately 30 seconds to generate 10,000 molecules.


To run generation on the demo input:
```bash
python generate.py data/1z92_IL2R.posp results pretrained_model/epoch32.pth pretrained_model --n_mol 1000 --filter --device cuda:0 --seed 123
```


The current model allows for a maximum of 8 pharmacophore features in a single hypothesis. To increase this limit, you can retrain the model with a greater number of randomly selected pharmacophore elements and a larger `MAX_NUM_PP_GRAPHS`.



## Acknowledgements
This implementation is inspired and partially built on earlier works [1], [2].


## References

* [1] Zhu, Huimin, et al. "A pharmacophore-guided deep learning approach for bioactive molecular generation." Nature Communications 14.1 (2023): 6234.
    
* [2] Yoshida, Shuhei, et al. "Peptide-to-small molecule: a pharmacophore-guided small molecule lead generation strategy from high-affinity macrocyclic peptides." Journal of Medicinal Chemistry 65.15 (2022): 10655-10673.   

