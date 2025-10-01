# Drug-Target Interaction Datasets

## Downloaded Datasets

### 1. DAVIS Dataset
- **Location**: `data/raw/davis/`
- **Description**: Kinase inhibitor bioactivities
- **Size**: 442 proteins × 72 compounds
- **Target**: Kinase proteins
- **Values**: Kd (dissociation constant)

### 2. KIBA Dataset
- **Location**: `data/raw/kiba/`
- **Description**: Kinase Inhibitor BioActivities
- **Size**: 229 proteins × 2111 compounds
- **Target**: Kinase proteins
- **Values**: KIBA scores (combined from Ki, Kd, IC50)

### 3. BindingDB
- **Location**: `data/raw/bindingdb/`
- **Description**: Large-scale binding affinity database
- **Note**: Requires registration at https://www.bindingdb.org/
- **Size**: >1M interactions (full dataset)

### 4. ChEMBL Sample
- **Location**: `data/raw/chembl/`
- **Description**: Molecular properties and bioactivities
- **Use**: Additional features for molecules

### 5. PDB Structures
- **Location**: `data/raw/pdb_structures/`
- **Description**: 3D protein structures
- **Use**: Structure-based protein representations

## Data Organization

```
data/
├── raw/              # Original downloaded data
│   ├── davis/        # DAVIS dataset
│   ├── kiba/         # KIBA dataset
│   ├── bindingdb/    # BindingDB data
│   ├── chembl/       # ChEMBL molecular data
│   └── pdb_structures/ # PDB protein structures
├── processed/        # Preprocessed data
│   ├── molecular_graphs/  # Converted molecular graphs
│   ├── protein_features/  # Protein representations
│   └── interaction_data/  # Processed DTI pairs
└── splits/          # Train/val/test splits
    ├── davis/
    ├── kiba/
    └── bindingdb/
```

## Usage

1. Run `python src/download_data.py` to download all datasets
2. Use notebooks in `notebooks/` for preprocessing
3. Processed data will be saved in `data/processed/`
