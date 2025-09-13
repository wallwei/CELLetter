import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidueMoE(nn.Module):
    def __init__(self, residue_dim, reduce_dim, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(residue_dim, reduce_dim * 2, 3, dilation=2, padding=2),
                nn.BatchNorm1d(reduce_dim * 2),
                nn.GELU(),
                nn.Conv1d(reduce_dim * 2, reduce_dim, 5, padding=2),
                nn.BatchNorm1d(reduce_dim),
                nn.GELU()
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(residue_dim, num_experts),
            nn.Softmax(dim=-1)
        )
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        # x: [batch, residue_dim, seq_len]
        gate_weights = self.gate(x)  # [batch, num_experts]

        # Process each expert
        expert_outputs = []
        for expert in self.experts:
            out = expert(x)  # [batch, reduce_dim, seq_len]
            expert_outputs.append(self.pool(out))  # [batch, reduce_dim, 1]

        # Combine expert outputs
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch, num_experts, reduce_dim, 1]
        weighted_output = torch.einsum('be,be...->b...', gate_weights, expert_outputs)
        return weighted_output.squeeze(-1)  # [batch, reduce_dim]


class GlobalFeatureProcessor(nn.Module):
    def __init__(self, global_dim=1024, reduce_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(global_dim, reduce_dim * 2),
            nn.BatchNorm1d(reduce_dim * 2),
            nn.GELU(),
            nn.Linear(reduce_dim * 2, reduce_dim),
        )
        self.shortcut = nn.Linear(global_dim, reduce_dim) if global_dim != reduce_dim else nn.Identity()

    def forward(self, x):
        # Input x shape: [batch, global_dim]
        residual = self.shortcut(x)
        x = self.mlp(x)
        return F.gelu(x + residual)  # Activation after residual connection


class EnhancedDecoderWithResidues(nn.Module):
    def __init__(self, global_dim=1024, residue_dim=1024, reduce_dim=512):
        super().__init__()

        # Global feature processing
        self.lig_global_processor = GlobalFeatureProcessor(global_dim, reduce_dim)
        self.rec_global_processor = GlobalFeatureProcessor(global_dim, reduce_dim)

        # Residue feature processing with MoE
        self.lig_residue_moe = ResidueMoE(residue_dim, reduce_dim)
        self.rec_residue_moe = ResidueMoE(residue_dim, reduce_dim)

        # Feature fusion gating mechanism
        self.fusion_gate = nn.Sequential(
            nn.Linear(reduce_dim, reduce_dim),
            nn.Sigmoid()
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(reduce_dim * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, 2)
        )

    def forward(self, batch):
        # Extract required features from batch dictionary
        lig_global = self.lig_global_processor(batch['lig_global'])
        rec_global = self.rec_global_processor(batch['rec_global'])
        ligand_residues = batch['lig_residue'].permute(0, 2, 1)  # [batch, channels, seq_len]
        receptor_residues = batch['rec_residue'].permute(0, 2, 1)

        # Process residue features
        lig_res = self.lig_residue_moe(ligand_residues).squeeze(-1)
        rec_res = self.rec_residue_moe(receptor_residues).squeeze(-1)

        # Gated fusion: global features control residue features
        lig_gate = self.fusion_gate(lig_global)
        lig_combined = lig_gate * lig_global + (1 - lig_gate) * lig_res

        rec_gate = self.fusion_gate(rec_global)
        rec_combined = rec_gate * rec_global + (1 - rec_gate) * rec_res

        # Feature difference
        diff_feat = torch.abs(lig_combined - rec_combined)

        # Feature product
        product_feat = lig_combined * rec_combined

        # Final concatenation
        combined = torch.cat([lig_combined, rec_combined, diff_feat, product_feat], dim=1)
        return self.classifier(combined)