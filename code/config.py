import torch

# Data paths
DATA_DIR = './dataset/dataset1/features_1'
RESIDUE_DIR = './dataset/dataset1/residue_features_1'
RESULTS_DIR = "/data/NAS_DATA_Department_Of_Sciences/tmp/ww/cell/BERT/out/five-fold-cross-validation/auc/dataset1"

# Training parameters
BATCH_SIZE = 32
NUM_EPOCHS = 100
N_SPLITS = 5
N_REPEATS = 1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 15
REDUCE_LR_PATIENCE = 10
REDUCE_LR_FACTOR = 0.5

# Model parameters
GLOBAL_DIM = 1024
RESIDUE_DIM = 1024
REDUCE_DIM = 512
NUM_EXPERTS = 4

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Random seed
SEED = 42