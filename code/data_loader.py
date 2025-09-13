import os
import logging
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from tqdm import tqdm

class LRIDataset(Dataset):
    def __init__(self, lri_data, ligands_dict, receptors_dict):
        # Filter out samples with missing residue features
        self.lri_data = []
        for item in lri_data:
            l, r, label = item
            if (ligands_dict.get(l) and ligands_dict[l].get('residue') is not None and
                receptors_dict.get(r) and receptors_dict[r].get('residue') is not None):
                self.lri_data.append(item)

        self.ligands_dict = ligands_dict
        self.receptors_dict = receptors_dict

        logging.info(f"Valid samples: {len(self.lri_data)}")

    def __len__(self):
        return len(self.lri_data)

    def __getitem__(self, item):
        l, r, label = self.lri_data[item]
        ligand_data = self.ligands_dict[l]
        receptor_data = self.receptors_dict[r]

        return (
            ligand_data['global'],
            ligand_data['residue'],
            receptor_data['global'],
            receptor_data['residue'],
            label
        )


class GetLRIDataset:
    def __init__(self, data_dir, residue_dir):
        # Verify file existence
        required_files = [
            os.path.join(data_dir, "related.csv"),
            os.path.join(data_dir, "ligand_global_features.csv"),
            os.path.join(data_dir, "receptor_global_features.csv")
        ]
        for f in required_files:
            if not os.path.exists(f):
                raise FileNotFoundError(f"File not found: {f}")

        # Store residue feature directory
        self.residue_dir = residue_dir

        # Load features and matrix
        self.ligands_dict = self.load_features(
            global_path=os.path.join(data_dir, "ligand_global_features.csv"),
            is_ligand=True
        )
        self.receptors_dict = self.load_features(
            global_path=os.path.join(data_dir, "receptor_global_features.csv"),
            is_ligand=False
        )
        self.matrix = self.load_matrix(os.path.join(data_dir, "related.csv"))

        # Validate ID matching
        self.validate_id_matching()

        # Initialize dataset
        self.init_lri()

    def load_features(self, global_path, is_ligand):
        """Load global features and fuse with residue features"""
        # Read global feature CSV
        global_df = pd.read_csv(global_path, index_col=0)
        features_dict = {}

        # Iterate over each protein
        for protein_id in tqdm(global_df.index, desc=f"Loading {'ligand' if is_ligand else 'receptor'} features"):
            # Global features
            global_feat = global_df.loc[protein_id].values.astype(np.float32)

            # Residue features
            residue_path = os.path.join(self.residue_dir, f"{protein_id}.npy")
            if not os.path.exists(residue_path):
                logging.warning(f"Residue feature file does not exist: {residue_path}, will use global features only")
                residue_feat = None
            else:
                residue_feat = np.load(residue_path)  # Shape: [seq_len, 1024]

            # Store original features
            features_dict[protein_id] = {
                'global': torch.tensor(global_feat, dtype=torch.float32),
                'residue': torch.tensor(residue_feat, dtype=torch.float32) if residue_feat is not None else None
            }

        return features_dict

    def load_matrix(self, path):
        """Load interaction matrix"""
        matrix = pd.read_csv(path, index_col=0)
        # Ensure all values are numeric
        matrix = matrix.apply(pd.to_numeric, errors='coerce').fillna(0)
        return matrix

    def validate_id_matching(self):
        """Validate ID consistency between feature files and matrix"""
        # Get ligand and receptor IDs from matrix
        matrix_ligands = set(self.matrix.index)
        matrix_receptors = set(self.matrix.columns)

        # Get IDs from feature dictionaries
        feature_ligands = set(self.ligands_dict.keys())
        feature_receptors = set(self.receptors_dict.keys())

        # Check for missing ligands
        missing_ligands = matrix_ligands - feature_ligands
        if missing_ligands:
            logging.warning(f"Warning: {len(missing_ligands)} ligands missing from feature files, will be ignored")
            # Update matrix to keep only existing ligands
            self.matrix = self.matrix.loc[list(feature_ligands & matrix_ligands)]

        # Check for missing receptors
        missing_receptors = matrix_receptors - feature_receptors
        if missing_receptors:
            logging.warning(f"Warning: {len(missing_receptors)} receptors missing from feature files, will be ignored")
            # Update matrix to keep only existing receptors
            self.matrix = self.matrix.loc[:, list(feature_receptors & matrix_receptors)]

        logging.info(f"Updated interaction matrix shape: {self.matrix.shape}")

    def init_lri(self):
        """Initialize LRI dataset"""
        # Get positive and negative samples
        stacked = self.matrix.stack()
        positive = stacked[stacked == 1].index.tolist()
        positive = [(*idx, 1) for idx in tqdm(positive, desc="Processing positive samples")]

        # Balance negative samples
        negative = stacked[stacked == 0].index.tolist()
        random.shuffle(negative)
        negative = [(*idx, 0) for idx in negative[:len(positive)]]

        # Combine and shuffle data
        self.lri_data = positive + negative
        random.shuffle(self.lri_data)
        self.size = len(self.lri_data)

        logging.info(f"Dataset statistics: Total samples={self.size}, Positive samples={len(positive)}, Negative samples={len(negative)}")

    def get_kfold_dataloader(self, batch_size=32, n_splits=5):
        """Five-fold cross-validation data loader"""
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Prepare features and labels
        ligands = [x[0] for x in self.lri_data]
        receptors = [x[1] for x in self.lri_data]
        labels = [x[2] for x in self.lri_data]

        # Convert to numpy arrays for KFold
        ligands = np.array(ligands)
        receptors = np.array(receptors)
        labels = np.array(labels)

        fold_loaders = []
        for train_idx, test_idx in kf.split(labels):
            # Create dataset for each fold
            train_set = [(ligands[i], receptors[i], labels[i]) for i in train_idx]
            test_set = [(ligands[i], receptors[i], labels[i]) for i in test_idx]

            # Create data loaders
            train_loader = DataLoader(
                dataset=LRIDataset(train_set, self.ligands_dict, self.receptors_dict),
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                collate_fn=collate_func,
                num_workers=4
            )
            test_loader = DataLoader(
                dataset=LRIDataset(test_set, self.ligands_dict, self.receptors_dict),
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                collate_fn=collate_func,
                num_workers=4
            )

            fold_loaders.append((train_loader, test_loader))

        return fold_loaders


def collate_func(batch):
    lig_global, lig_residue, rec_global, rec_residue, labels = zip(*batch)

    # Process global features
    lig_global = torch.stack(lig_global)
    rec_global = torch.stack(rec_global)

    # Process residue features - need to pad to same length
    max_lig_len = max(r.shape[0] for r in lig_residue)
    max_rec_len = max(r.shape[0] for r in rec_residue)

    padded_lig_residue = []
    for res in lig_residue:
        pad_len = max_lig_len - res.shape[0]
        padded = F.pad(res, (0, 0, 0, pad_len), value=0)
        padded_lig_residue.append(padded)

    padded_rec_residue = []
    for res in rec_residue:
        pad_len = max_rec_len - res.shape[0]
        padded = F.pad(res, (0, 0, 0, pad_len), value=0)
        padded_rec_residue.append(padded)

    lig_residue = torch.stack(padded_lig_residue)
    rec_residue = torch.stack(padded_rec_residue)

    # Create attention masks
    lig_mask = torch.zeros(lig_residue.shape[0], max_lig_len, dtype=torch.bool)
    for i, res in enumerate(lig_residue):
        lig_mask[i, :res.shape[0]] = 1

    rec_mask = torch.zeros(rec_residue.shape[0], max_rec_len, dtype=torch.bool)
    for i, res in enumerate(rec_residue):
        rec_mask[i, :res.shape[0]] = 1

    labels = torch.tensor(labels, dtype=torch.long)

    return {
        'lig_global': lig_global,
        'lig_residue': lig_residue,
        'lig_mask': lig_mask,
        'rec_global': rec_global,
        'rec_residue': rec_residue,
        'rec_mask': rec_mask,
        'labels': labels
    }