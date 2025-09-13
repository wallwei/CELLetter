import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import json
import logging
from GLORI import EnhancedDecoderWithResidues, GetLRIDataset, collate_func

# ===== Hardcoded Paths =====
MODEL_PATH = "/data/NAS_DATA_Department_Of_Sciences/tmp/ww/cell/BERT/out/five-fold-cross-validation/auc/dataset1/best_model_repeat1_fold4.pth"
DATA_DIR = "/data/wuwei/cell/BERT/dataset/dataset1/features_1"
RESIDUE_DIR = "/data/wuwei/cell/BERT/dataset/dataset1/residue_features_1"
MATRIX_PATH = "/data/wuwei/cell/BERT/dataset/dataset1/related.csv"
OUTPUT_PATH = "/data/wuwei/cell/BERT/out/five-fold-cross-validation/auc/dataset1/predictions.csv"
THRESHOLD = 0.9999  # Probability threshold
# ===========================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger()


def load_features(global_path, residue_dir, is_ligand=True):
    """Load global features and fuse with residue features"""
    global_df = pd.read_csv(global_path, index_col=0)
    features_dict = {}

    for protein_id in tqdm(global_df.index, desc=f"Loading {'ligand' if is_ligand else 'receptor'} features"):
        global_feat = global_df.loc[protein_id].values.astype(np.float32)

        residue_path = os.path.join(residue_dir, f"{protein_id}.npy")
        if not os.path.exists(residue_path):
            logging.warning(f"Residue feature file does not exist: {residue_path}")
            residue_feat = None
        else:
            residue_feat = np.load(residue_path)

        features_dict[protein_id] = {
            'global': torch.tensor(global_feat, dtype=torch.float32),
            'residue': torch.tensor(residue_feat, dtype=torch.float32) if residue_feat is not None else None
        }

    return features_dict

class PredictionDataset(Dataset):
    def __init__(self, matrix_path, ligands_dict, receptors_dict):
        self.matrix = pd.read_csv(matrix_path, index_col=0)
        self.ligands = self.matrix.index.tolist()
        self.receptors = self.matrix.columns.tolist()

        self.ligands_dict = ligands_dict
        self.receptors_dict = receptors_dict

        # Collect all pairs with label 0
        self.zero_pairs = []
        self.missing_pairs = []

        # Get all positions with value 0 in the matrix
        zero_indices = np.where(self.matrix.values == 0)

        for row_idx, col_idx in zip(*zero_indices):
            ligand = self.ligands[row_idx]
            receptor = self.receptors[col_idx]

            if (ligand in ligands_dict and ligands_dict[ligand].get('residue') is not None and
                    receptor in receptors_dict and receptors_dict[receptor].get('residue') is not None):
                self.zero_pairs.append((row_idx, col_idx, ligand, receptor))
            else:
                self.missing_pairs.append((row_idx, col_idx, ligand, receptor))

        logger.info(f"Total samples with label 0 to predict: {len(self.zero_pairs)}, Missing feature samples: {len(self.missing_pairs)}")

    def __len__(self):
        return len(self.zero_pairs)

    def __getitem__(self, idx):
        lig_idx, rec_idx, ligand, receptor = self.zero_pairs[idx]
        ligand_data = self.ligands_dict[ligand]
        receptor_data = self.receptors_dict[receptor]

        return (
            ligand_data['global'],
            ligand_data['residue'],
            receptor_data['global'],
            receptor_data['residue'],
            lig_idx, rec_idx
        )

def predict_collate_func(batch):
    lig_global, lig_residue, rec_global, rec_residue, lig_idx, rec_idx = zip(*batch)

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

    return {
        'lig_global': lig_global,
        'lig_residue': lig_residue,
        'lig_mask': lig_mask,
        'rec_global': rec_global,
        'rec_residue': rec_residue,
        'rec_mask': rec_mask,
        'lig_indices': torch.tensor(lig_idx, dtype=torch.long),
        'rec_indices': torch.tensor(rec_idx, dtype=torch.long)
    }

def predict_with_hardcoded_paths():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model architecture
    model = EnhancedDecoderWithResidues().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    logger.info(f"Loaded model: {MODEL_PATH}")

    # Load feature data
    logger.info("Loading feature data...")
    ligands_dict = load_features(os.path.join(DATA_DIR, "ligand_global_features.csv"), RESIDUE_DIR, is_ligand=True)
    receptors_dict = load_features(os.path.join(DATA_DIR, "receptor_global_features.csv"), RESIDUE_DIR, is_ligand=False)

    # Create prediction dataset
    pred_dataset = PredictionDataset(MATRIX_PATH, ligands_dict, receptors_dict)

    # Create data loader
    pred_loader = DataLoader(
        pred_dataset,
        batch_size=32,
        shuffle=False,
        drop_last=False,
        collate_fn=predict_collate_func,  # Use the new collate function
        num_workers=4
    )

    # Store all prediction results
    all_results = []

    # Perform prediction
    with torch.no_grad():
        for batch in tqdm(pred_loader, desc="Predicting"):
            inputs = {k: v.to(device) for k, v in batch.items() if k not in ['lig_indices', 'rec_indices']}
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()

            lig_indices = batch['lig_indices'].numpy()
            rec_indices = batch['rec_indices'].numpy()

            for i in range(len(probs)):
                if probs[i] > THRESHOLD:
                    all_results.append([lig_indices[i], rec_indices[i], probs[i]])

    logger.info(f"Ignored {len(pred_dataset.missing_pairs)} samples with missing features")

    # Convert to DataFrame and sort
    results_df = pd.DataFrame(all_results, columns=['ligand_index', 'receptor_index', 'probability'])
    results_df = results_df.sort_values(by=['ligand_index', 'receptor_index'])

    # Save results
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Predictions saved to: {OUTPUT_PATH} (Total predictions: {len(results_df)})")

    return results_df


# Directly run prediction
if __name__ == "__main__":
    # No need for argument parsing, directly call the prediction function
    predict_with_hardcoded_paths()