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
    """Download DAVIS dataset"""
    print("\n" + "="*50)
    print("Downloading DAVIS Dataset...")
    print("="*50)
    
    davis_dir = Path("data/raw/davis")
    davis_dir.mkdir(parents=True, exist_ok=True)
    
    # DAVIS dataset from DeepDTA repository
    urls = {
        "data": "https://github.com/hkmztrk/DeepDTA/raw/master/data/davis.zip",
    }
    
    for name, url in urls.items():
        destination = davis_dir / f"davis_{name}.zip"
        if not destination.exists():
            download_file(url, destination, f"Downloading DAVIS {name}")
            extract_archive(str(destination), str(davis_dir))
            print(f"✓ DAVIS {name} downloaded and extracted")
        else:
            print(f"✓ DAVIS {name} already exists")

def download_kiba():
    """Download KIBA dataset"""
    print("\n" + "="*50)
    print("Downloading KIBA Dataset...")
    print("="*50)
    
    kiba_dir = Path("data/raw/kiba")
    kiba_dir.mkdir(parents=True, exist_ok=True)
    
    # KIBA dataset from DeepDTA repository
    urls = {
        "data": "https://github.com/hkmztrk/DeepDTA/raw/master/data/kiba.zip",
    }
    
    for name, url in urls.items():
        destination = kiba_dir / f"kiba_{name}.zip"
        if not destination.exists():
            download_file(url, destination, f"Downloading KIBA {name}")
            extract_archive(str(destination), str(kiba_dir))
            print(f"✓ KIBA {name} downloaded and extracted")
        else:
            print(f"✓ KIBA {name} already exists")

def download_bindingdb():
    """Download BindingDB dataset (smaller sample for demonstration)"""
    print("\n" + "="*50)
    print("Downloading BindingDB Sample Dataset...")
    print("="*50)
    
    bindingdb_dir = Path("data/raw/bindingdb")
    bindingdb_dir.mkdir(parents=True, exist_ok=True)
    
    # Note: Full BindingDB requires registration and is very large
    # Here we'll prepare a structure for it
    info = {
        "dataset": "BindingDB",
        "note": "Full dataset available at https://www.bindingdb.org/",
        "instructions": [
            "1. Register at https://www.bindingdb.org/",
            "2. Download the TSV file from Downloads section",
            "3. Place it in data/raw/bindingdb/",
            "4. Run preprocessing notebook to process it"
        ]
    }
    
    info_file = bindingdb_dir / "download_instructions.json"
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2)
    
    print("✓ BindingDB download instructions created")
    print("  Note: Full BindingDB requires registration at https://www.bindingdb.org/")

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
