# CELLetter: Leveraging Large Language Model and Dual-Stream Network to Identify Context-Specific Ligand-Receptor Interactions for Cell-Cell Communication Analysis
CELLetter is a deep learning framework for predicting cell-cell communication (CCC) by identifying ligand-receptor interactions (LRI) from protein sequence and expression data. The system combines global and residue-level features using a Mixture of Experts (MoE) architecture to achieve high-precision LRI prediction and cell communication scoring.

## Overveiw of CELLetter
<img src="https://github.com/wallwei/CELLetter/blob/8864e6b2bc4d06c1bb56eeef199a2d95ec77c31c/Fig1.pdf" width = 50%>

## Workflow of L-R prediction
<img src="https://github.com/wallwei/CELLetter/blob/8864e6b2bc4d06c1bb56eeef199a2d95ec77c31c/Fig2.pdf" width = 50%>

## Pipeline of CCC inference
<img src="https://github.com/wallwei/CELLetter/blob/8864e6b2bc4d06c1bb56eeef199a2d95ec77c31c/Fig3.pdf" width = 50%>

## Model Architecture
1.​​ Global Feature Processor​​: MLP with residual connections

2. Residue Feature Processor​​: Mixture of Experts with 1D convolutional networks
3. 
​​3. Fusion Gate​​: Adaptive weighting between global and local features

​​4. Classifier​​: Multi-layer perceptron for final prediction

## Features
Dual Feature Extraction​​: Combines global protein features with residue-level local features

​​Advanced Architecture​​: Utilizes Mixture of Experts (MoE) with gating mechanisms

## Installation
### Prerequisites
Python 3.8+
PyTorch 2.0+
CUDA-capable GPU (recommended)

## Usage Pipeline
### 1. Feature Extraction
```python
    feature_embedding/global_feature.py \
    --input datasets/protein_sequences.fasta \
    --output datasets/global_features/
```
### 2. Model Training
```python
    --data_dir datasets/global_features/ \
    --residue_dir datasets/residue_features/ \
    --matrix_path datasets/interaction_matrix.csv \
    --output_dir results/training/
```
### 3. Prediction
```python
    --model_path results/training/best_model.pth \
    --data_dir datasets/global_features/ \
    --residue_dir datasets/residue_features/ \
    --matrix_path datasets/interaction_matrix.csv \
    --output results/predictions.csv
```

### 4. Cell Communication Scoring
```
python code/CCC_score/compute_communication.py \
    --lri_file results/predictions.csv \
    --expr_matrix datasets/expression_matrix.csv \
    --meta_file datasets/cell_metadata.csv \
    --output results/communication_matrix.csv
```

## Outputs
--predictions.csv: Predicted ligand-receptor pairs
--communication_matrix.csv: Cell-type communication strength matrix
--top3_ligand_receptor_pairs.csv: Top contributing LRIs
