# CELLetter: Leveraging Large Language Model and Dual-Stream Network to Identify Context-Specific Ligand-Receptor Interactions for Cell-Cell Communication Analysis
CELLetter is a deep learning framework for predicting cell-cell communication (CCC) by identifying ligand-receptor interactions (LRI) from protein sequence and expression data. The system combines global and residue-level features using a Mixture of Experts (MoE) architecture to achieve high-precision LRI prediction and cell communication scoring.

## Overveiw of CELLetter
<img src="https://github.com/wallwei/CELLetter/blob/3901030074cb7c371f972693ad826f715afa2880/Fig1_01.png" width = 50%>
(A) Interacting L-R prediction. Protein sequences of ligands and receptors are encoded using ProstT5 to learn their initial embeddings. These embeddings are then processed through a novel feature learning model composed of a dual-stream architecture, a gated mechanism, and an interaction strategy. The learned features are taken as inputs of an MLP to find potential interacting L-R pairs. (B) CCC inference. The experimentally validated and predicted interactions are merged and filtered to construct a comprehensive set based on their scRNA-seq data. Next, TF activities are estimated using pySCENIC. And cellular crosstalk is scored based on L-R geometric mean expression across cell types with weight of the global TF activity. Finally, the constructed CCC network is visualized to reveal cellular signaling patterns.

## Workflow of L-R prediction
<img src="https://github.com/wallwei/CELLetter/blob/3901030074cb7c371f972693ad826f715afa2880/Fig2_01.png" width = 50%>
(1) Feature embedding. The protein sequences of ligands and receptors are encoded using ProstT5 to generate both global features (sequence-level) and local features (residue-level). (2) Feature extraction and dimensionality reduction. The global features are extracted and reduced by comnbining an MLP with residual connections, while the residue-level features are refined to capture diverse local sequence patterns by a CNN-MoE module comprising multiple 1D-CNN experts with dilated convolutions. (3) Feature fusion. A gating mechanism is utilized to dynamically combine the refined global and local features, thereby producing a context-aware representation for each L-R. (4) Feature interaction. The absolute difference and element-wise product operations are used to construct interaction-aware features from the fused representations. (5) Classification. Both context-aware and interaction-aware representation are concatenated and fed into an MLP classifier to predict the interaction probability

## Pipeline of CCC inference
<img src="https://github.com/wallwei/CELLetter/blob/3901030074cb7c371f972693ad826f715afa2880/Fig3_01.png" width = 50%>
The workflow consists of four main components: (1) L-R Filtering. The experimentally validated and predicted interacting L-R pairs are merged and filtered using scRNA-seq data. (2) TF activity computation. TF activities are inferred using the pySCENIC toolkit, which combines co-expression network analysis with motif enrichment to estimate intracellular signaling potentials. (3) CCC scoring. Cellular communication strength is assessed by integrating L-R expression with global TF activity; (4) CCC visualization: The inferred CCC is visualized to highlight key communication pathways using heatmap, bubble plot, stacked bar chart and circos plot.

## Model Architecture
1.​​ Global Feature Processor​​: MLP with residual connections

2. Residue Feature Processor​​: Mixture of Experts with 1D convolutional networks

​​3. Fusion Gate​​: Adaptive weighting between global and local features

​​4. Classifier​​: Multi-layer perceptron for final prediction

## Features
Dual Feature Extraction​​: Combines global protein features with residue-level local features

​​Advanced Architecture​​: Utilizes Mixture of Experts (MoE) with gating mechanisms

## Installation
### Prerequisites
--Python==3.9

--numpy==1.24.2

--pandas==2.0.1

--torch==2.4.1

--pySCENIC==0.12.0

--CUDA-capable GPU (recommended)

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
