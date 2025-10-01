#!/usr/bin/env python3
"""
Script to download Drug-Target Interaction datasets
Includes: DAVIS, KIBA, and BindingDB datasets
"""

import os
import sys
import requests
import zipfile
import tarfile
import gzip
import shutil
from tqdm import tqdm
from pathlib import Path
import json

def download_file(url, destination, description="Downloading"):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=description) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                pbar.update(len(chunk))
    
    return destination

def extract_archive(archive_path, extract_to):
    """Extract zip, tar, or gz archives"""
    print(f"Extracting {archive_path}...")
    
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)
    elif archive_path.endswith('.gz'):
        output_path = os.path.join(extract_to, os.path.basename(archive_path[:-3]))
        with gzip.open(archive_path, 'rb') as gz_ref:
            with open(output_path, 'wb') as out_file:
                shutil.copyfileobj(gz_ref, out_file)
    else:
        print(f"Unsupported archive format: {archive_path}")

def download_davis():
    """Download DAVIS dataset from DeepDTA repository"""
    print("\n" + "="*50)
    print("Downloading DAVIS Dataset...")
    print("="*50)
    
    davis_dir = Path("data/raw/davis")
    davis_dir.mkdir(parents=True, exist_ok=True)
    
    # Base URL for DeepDTA repository
    base_url = "https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/davis/"
    
    # Files to download
    files = {
        "Y": "Y",  # Pickle file with affinity matrix
        "ligands_can.txt": "ligands_can.txt",  # Canonical SMILES
        "ligands_iso.txt": "ligands_iso.txt",  # Isomeric SMILES
        "proteins.txt": "proteins.txt",  # Protein sequences
        "affinity_matrix.txt": "drug-target_interaction_affinities_Kd__Davis_et_al.2011v1.txt",
        "drug_similarity.txt": "drug-drug_similarities_2D.txt",
        "protein_similarity.txt": "target-target_similarities_WS.txt",
    }
    
    for local_name, remote_name in files.items():
        destination = davis_dir / local_name
        if not destination.exists():
            url = base_url + remote_name
            try:
                download_file(url, destination, f"Downloading {remote_name}")
                print(f"✓ {local_name} downloaded")
            except Exception as e:
                print(f"✗ Failed to download {local_name}: {e}")
        else:
            print(f"✓ {local_name} already exists")
    
    # Download fold information
    folds_dir = davis_dir / "folds"
    folds_dir.mkdir(exist_ok=True)
    
    fold_files = [
        "test_fold_setting1.txt",
        "train_fold_setting1.txt",
    ]
    
    for fold_file in fold_files:
        destination = folds_dir / fold_file
        if not destination.exists():
            url = f"{base_url}folds/{fold_file}"
            try:
                download_file(url, destination, f"Downloading {fold_file}")
                print(f"✓ {fold_file} downloaded")
            except Exception as e:
                print(f"✗ Failed to download {fold_file}: {e}")
        else:
            print(f"✓ {fold_file} already exists")
    
    print("✓ DAVIS dataset downloaded successfully")

def download_kiba():
    """Download KIBA dataset from DeepDTA repository"""
    print("\n" + "="*50)
    print("Downloading KIBA Dataset...")
    print("="*50)
    
    kiba_dir = Path("data/raw/kiba")
    kiba_dir.mkdir(parents=True, exist_ok=True)
    
    # Base URL for DeepDTA repository
    base_url = "https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/kiba/"
    
    # Files to download
    files = {
        "Y": "Y",  # Pickle file with affinity matrix (KIBA scores)
        "ligands_can.txt": "ligands_can.txt",  # Canonical SMILES
        "ligands_iso.txt": "ligands_iso.txt",  # Isomeric SMILES
        "proteins.txt": "proteins.txt",  # Protein sequences
        "drug_similarity.txt": "drug-drug_similarities_2D.txt",
        "protein_similarity.txt": "target-target_similarities_WS.txt",
    }
    
    for local_name, remote_name in files.items():
        destination = kiba_dir / local_name
        if not destination.exists():
            url = base_url + remote_name
            try:
                download_file(url, destination, f"Downloading {remote_name}")
                print(f"✓ {local_name} downloaded")
            except Exception as e:
                print(f"✗ Failed to download {local_name}: {e}")
        else:
            print(f"✓ {local_name} already exists")
    
    # Download fold information
    folds_dir = kiba_dir / "folds"
    folds_dir.mkdir(exist_ok=True)
    
    fold_files = [
        "test_fold_setting1.txt",
        "train_fold_setting1.txt",
    ]
    
    for fold_file in fold_files:
        destination = folds_dir / fold_file
        if not destination.exists():
            url = f"{base_url}folds/{fold_file}"
            try:
                download_file(url, destination, f"Downloading {fold_file}")
                print(f"✓ {fold_file} downloaded")
            except Exception as e:
                print(f"✗ Failed to download {fold_file}: {e}")
        else:
            print(f"✓ {fold_file} already exists")
    
    print("✓ KIBA dataset downloaded successfully")

def download_bindingdb():
    """Download BindingDB dataset (TSV format, publicly available)"""
    print("\n" + "="*50)
    print("Downloading BindingDB Dataset...")
    print("="*50)
    
    bindingdb_dir = Path("data/raw/bindingdb")
    bindingdb_dir.mkdir(parents=True, exist_ok=True)
    
    # BindingDB is publicly available (no registration required)
    # Using a subset for demonstration - full dataset is ~490 MB
    # You can change to BindingDB_All_202509_tsv.zip for the full dataset
    
    # Option 1: Download ChEMBL subset (smaller, ~280 MB)
    url = "https://www.bindingdb.org/rwd/bind/downloads/BindingDB_ChEMBL_202509_tsv.zip"
    dataset_name = "ChEMBL subset"
    
    # Option 2: For full dataset, uncomment:
    # url = "https://www.bindingdb.org/rwd/bind/downloads/BindingDB_All_202509_tsv.zip"
    # dataset_name = "Full dataset"
    
    destination = bindingdb_dir / "BindingDB_data.zip"
    
    if not destination.exists():
        print(f"Downloading BindingDB {dataset_name} (~280 MB, this may take a while)...")
        try:
            download_file(url, destination, f"Downloading BindingDB {dataset_name}")
            extract_archive(str(destination), str(bindingdb_dir))
            print(f"✓ BindingDB {dataset_name} downloaded and extracted")
        except Exception as e:
            print(f"✗ Failed to download BindingDB: {e}")
            print("  You can manually download from: https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp")
    else:
        print("✓ BindingDB data already exists")
    
    # Create info file
    info = {
        "dataset": "BindingDB",
        "source": "https://www.bindingdb.org/",
        "note": "ChEMBL subset downloaded. Full dataset available at downloads page.",
        "downloads_page": "https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp",
        "format": "TSV (tab-separated values)",
        "size": "~280 MB (ChEMBL subset) or ~490 MB (full dataset)"
    }
    
    info_file = bindingdb_dir / "dataset_info.json"
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2)
    
    print("\n✓ BindingDB dataset ready")
    print("  Note: This downloads the ChEMBL subset. For the full dataset,")
    print("  edit download_data.py and uncomment the full dataset URL.")

def download_chembl_sample():
    """Download a sample from ChEMBL for additional molecular data"""
    print("\n" + "="*50)
    print("Downloading ChEMBL Sample Data...")
    print("="*50)
    
    chembl_dir = Path("data/raw/chembl")
    chembl_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample ChEMBL data for molecular properties
    # This would be a subset focused on kinase inhibitors
    info = {
        "dataset": "ChEMBL",
        "note": "Sample molecular property data",
        "url": "https://www.ebi.ac.uk/chembl/",
        "description": "Additional molecular descriptors and bioactivity data"
    }
    
    info_file = chembl_dir / "info.json"
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2)
    
    print("✓ ChEMBL info created")

def download_pdb_samples():
    """Download sample PDB structures for protein targets"""
    print("\n" + "="*50)
    print("Preparing PDB Structure Directory...")
    print("="*50)
    
    pdb_dir = Path("data/raw/pdb_structures")
    pdb_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample PDB IDs for common drug targets
    sample_pdbs = [
        "1ATP",  # ATP binding protein
        "2SRC",  # Tyrosine kinase SRC
        "3ETA",  # EGFR kinase domain
        "4AGD",  # BTK kinase
    ]
    
    info = {
        "dataset": "PDB Structures",
        "sample_ids": sample_pdbs,
        "note": "Protein structures can be downloaded from RCSB PDB",
        "url": "https://www.rcsb.org/"
    }
    
    info_file = pdb_dir / "pdb_info.json"
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2)
    
    print("✓ PDB directory prepared")

def create_data_readme():
    """Create README for data directory"""
    readme_content = """# Drug-Target Interaction Datasets

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
"""
    
    with open("data/README.md", 'w') as f:
        f.write(readme_content)
    
    print("\n✓ Data README created")

def main():
    """Main download function"""
    print("\n" + "="*60)
    print(" DRUG-TARGET INTERACTION DATASET DOWNLOADER")
    print("="*60)
    
    # Create data directories
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("data/splits").mkdir(parents=True, exist_ok=True)
    
    try:
        # Download all datasets
        download_davis()
        download_kiba()
        download_bindingdb()
        download_chembl_sample()
        download_pdb_samples()
        
        # Create README
        create_data_readme()
        
        print("\n" + "="*60)
        print(" ✓ ALL DATASETS DOWNLOADED SUCCESSFULLY!")
        print("="*60)
        print("\nNext steps:")
        print("1. For BindingDB: Register and download from https://www.bindingdb.org/")
        print("2. Run the preprocessing notebooks to prepare the data")
        print("3. Start with 01_data_exploration.ipynb notebook")
        
    except Exception as e:
        print(f"\n✗ Error during download: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
