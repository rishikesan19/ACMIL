import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# ============================================================================
# POSITIONAL ENCODING
# ============================================================================
class PositionalEncoding(nn.Module):
    """Sequential Positional Encoding for pseudo-bag instances."""
    def __init__(self, embed_dim, max_len=2048):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# ============================================================================
# FEATURE NORMALIZATION (from PAMIL)
# ============================================================================
def normalize_features(features, norm_type='instance'):
    """Normalize features as in PAMIL."""
    if norm_type == 'instance':
        # L2 normalization per instance
        return F.normalize(features, p=2, dim=-1)
    elif norm_type == 'batch':
        # Batch normalization
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True) + 1e-6
        return (features - mean) / std
    else:
        return features

# ============================================================================
# MLFF MODULE 
# ============================================================================
class MLFF(nn.Module):
    """Multi-Level Feature Fusion."""
    def __init__(self, embed_dim):
        super(MLFF, self).__init__()
        self.embed_dim = embed_dim
        # Add layer norm for stability (from PAMIL)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, memory_bank):
        """
        memory_bank: List of dicts with 'token' and 'pred' keys
        Returns: F_bar_k_set tensor of shape [K-1, embed_dim] or None
        """
        if len(memory_bank) < 2:
            return None
        
        # Extract top token and prediction
        f_top = memory_bank[0]['token']  # Shape: [embed_dim]
        y_top = memory_bank[0]['pred']   # Shape: [] (scalar)
        
        # Extract other tokens and predictions
        f_k_tokens = [item['token'] for item in memory_bank[1:]]
        y_k_preds = [item['pred'] for item in memory_bank[1:]]
        
        # Equation 6: Element-wise multiplication
        f_hat_top = y_top * f_top
        f_hat_k_list = [y_k * f_k for y_k, f_k in zip(y_k_preds, f_k_tokens)]
        
        if not f_hat_k_list:
            return None
        
        f_hat_k_tensor = torch.stack(f_hat_k_list)  # Shape: [K-1, embed_dim]
        
        # Equation 7: Calculate weights using sum of features
        f_hat_top_sum = f_hat_top.sum()
        f_hat_k_sums = f_hat_k_tensor.sum(dim=1)  # Shape: [K-1]
        
        # Combine all sums for softmax
        all_sums = torch.cat([f_hat_top_sum.unsqueeze(0), f_hat_k_sums])
        softmax_weights = F.softmax(all_sums, dim=0)
        
        # Extract relative weights W_k (Equation 7)
        w_top = softmax_weights[0]
        w_k = softmax_weights[1:] / (w_top + 1e-8)
        
        # Equation 8: Weighted combination with normalization
        f_bar_list = []
        for i, (w, f_k) in enumerate(zip(w_k, f_k_tokens)):
            f_bar = self.norm(f_top + w * f_k)  # Add layer norm
            f_bar_list.append(f_bar)
        
        if not f_bar_list:
            return None
        
        return torch.stack(f_bar_list, dim=0)  # Shape: [K-1, embed_dim]

# ============================================================================
# MEMORY-BASED CROSS-ATTENTION 
# ============================================================================
class MemoryBasedCrossAttention(nn.Module):
    """Memory-based Cross-Attention."""
    def __init__(self, embed_dim, dropout=0.1, heads=8):
        super(MemoryBasedCrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim, heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        # Add FFN for better representation (from PAMIL)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, u_t, adaptive_memory_bank, f_bar_k_set):
        """
        u_t: [1, 1, embed_dim]
        adaptive_memory_bank: List of dicts
        f_bar_k_set: [K-1, embed_dim] or None
        """
        # Collect memory tokens
        mem_tokens = [item['token'] for item in adaptive_memory_bank]
        
        if not mem_tokens and (f_bar_k_set is None or f_bar_k_set.numel() == 0):
            return self.norm2(u_t + self.ffn(u_t))
        
        # Combine memory tokens and refined features
        kv_source = []
        if mem_tokens:
            kv_source.append(torch.stack(mem_tokens, dim=0))
        if f_bar_k_set is not None and f_bar_k_set.numel() > 0:
            kv_source.append(f_bar_k_set)
        
        kv_tokens = torch.cat(kv_source, dim=0).unsqueeze(0)  # [1, num_tokens, embed_dim]
        
        # Apply multi-head attention with residual
        attn_output, _ = self.multihead_attn(u_t, kv_tokens, kv_tokens)
        u_t = self.norm1(u_t + attn_output)
        
        # Apply FFN with residual
        u_t = self.norm2(u_t + self.ffn(u_t))
        
        return u_t

# ============================================================================
# SPFF MODULE 
# ============================================================================
class SPFF(nn.Module):
    """Self-Progressive Feature Fusion."""
    def __init__(self, input_dim, embed_dim, K, dropout, n_classes, heads=8):
        super(SPFF, self).__init__()
        self.K = K
        self.embed_dim = embed_dim
        self.n_classes = n_classes
        
        # Feature projection with layer norm (from PAMIL)
        self.fc_in = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.position_encoding = PositionalEncoding(embed_dim, max_len=2048)
        
        # Learnable initial token
        self.initial_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.normal_(self.initial_token, std=0.02)
        
        # Transformer for feature aggregation
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation=F.gelu,
            batch_first=True,
            norm_first=True  # Pre-LN for stability (from PAMIL)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_encoder_layer, 
            num_layers=2,
            norm=nn.LayerNorm(embed_dim)
        )
        
        # Memory-based cross-attention
        self.mca = MemoryBasedCrossAttention(embed_dim, dropout, heads=heads)
        
        # Multi-level feature fusion
        self.mlff = MLFF(embed_dim=embed_dim)
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # MLP for pseudo-label prediction - outputs single positive probability
        self.mlp_pseudo_label = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)  # Single output for binary-style prediction
        )

    def forward(self, v_t, adaptive_memory_bank):
        """
        v_t: [M, input_dim] - pseudo-bag features
        adaptive_memory_bank: List of dicts with 'token' and 'pred'
        Returns: (F_t, y_hat_t) where F_t is token and y_hat_t is pseudo-label prediction
        """
        # Normalize input features (from PAMIL)
        v_t = normalize_features(v_t, norm_type='instance')
        
        # Embed and add positional encoding
        v_t_embed = self.fc_in(v_t.unsqueeze(0))  # [1, M, embed_dim]
        v_t_with_pos = self.position_encoding(v_t_embed)
        
        # Add class token
        cls_token = self.initial_token.expand(v_t_with_pos.shape[0], -1, -1)
        transformer_input = torch.cat([cls_token, v_t_with_pos], dim=1)
        
        # Create attention mask to prevent padding interference
        mask = torch.zeros((transformer_input.shape[1], transformer_input.shape[1]), 
                          device=transformer_input.device).bool()
        
        # Transformer encoding
        transformer_output = self.transformer_encoder(transformer_input, src_key_padding_mask=None)
        u_t = transformer_output[:, 0:1, :]  # Extract class token
        
        # Apply MLFF to get refined features
        f_bar_k_set = self.mlff(adaptive_memory_bank)
        
        # Memory-based cross-attention
        F_t = self.mca(u_t, adaptive_memory_bank, f_bar_k_set)
        F_t = self.norm(F_t).squeeze(0)  # [1, embed_dim]
        
        # Predict pseudo-label (binary probability for positive class)
        logits = self.mlp_pseudo_label(F_t)  # [1, 1]
        y_hat_t = torch.sigmoid(logits).squeeze()  # Scalar in [0, 1]
        
        return F_t.squeeze(0), y_hat_t  # Return: [embed_dim], scalar

# ============================================================================
# LINEAR ATTENTION & EIRL 
# ============================================================================
class LinearAttention(nn.Module):
    """Linear Attention for explicit relations (LAM)."""
    def __init__(self, embed_dim, dropout=0.1):
        super(LinearAttention, self).__init__()
        self.attention_fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, x):
        """
        x: [batch_size, num_tokens, embed_dim]
        Returns: (h, attention_weights)
        """
        scores = self.attention_fc(x).squeeze(-1)  # [batch_size, num_tokens]
        attn = F.softmax(scores, dim=1)
        h = torch.bmm(attn.unsqueeze(1), x).squeeze(1)  # [batch_size, embed_dim]
        return h, attn

class EIRL(nn.Module):
    """Explicit-Implicit Representation Learning."""
    def __init__(self, embed_dim, n_classes, dropout, heads=8):
        super(EIRL, self).__init__()
        
        # Explicit relation via Linear Attention
        self.lam = LinearAttention(embed_dim, dropout)
        self.mlp_er = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, n_classes)
        )
        
        # Implicit relation via Multi-head Self-Attention
        self.msa = nn.MultiheadAttention(embed_dim, heads, dropout=dropout, batch_first=True)
        self.h_ir_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.normal_(self.h_ir_token, std=0.02)
        self.mlp_ir = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, n_classes)
        )
        
        # Add normalization layers
        self.norm_er = nn.LayerNorm(embed_dim)
        self.norm_ir = nn.LayerNorm(embed_dim)

    def forward(self, F_all):
        """
        F_all: [T, embed_dim] - all pseudo-bag tokens
        Returns: explicit and implicit representations and predictions
        """
        # Normalize input
        F_all = normalize_features(F_all, norm_type='instance')
        
        # Add batch dimension
        F_all_batch = F_all.unsqueeze(0)  # [1, T, embed_dim]
        
        # Explicit relations
        h_ER, A_ER = self.lam(F_all_batch)
        h_ER = self.norm_er(h_ER)
        Y_hat_ER = self.mlp_er(h_ER)
        
        # Implicit relations
        h_ir_token = self.h_ir_token.expand(F_all_batch.size(0), -1, -1)
        h_IR, A_IR_raw = self.msa(h_ir_token, F_all_batch, F_all_batch)
        h_IR = self.norm_ir(h_IR.squeeze(1))
        Y_hat_IR = self.mlp_ir(h_IR)
        
        return h_ER, Y_hat_ER, A_ER.squeeze(0), h_IR, Y_hat_IR, A_IR_raw.squeeze(1)

# ============================================================================
# DDM 
# ============================================================================
class DDM(nn.Module):
    """Dynamic Decision-Making."""
    def __init__(self, embed_dim, n_classes):
        super(DDM, self).__init__()
        self.weight_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, 2)
        )

    def forward(self, h_ER, Y_hat_ER, h_IR, Y_hat_IR):
        """
        Dynamically weight explicit and implicit predictions.
        """
        # Get confidence scores
        conf_ER, _ = F.softmax(Y_hat_ER, dim=1).max(dim=1, keepdim=True)
        conf_IR, _ = F.softmax(Y_hat_IR, dim=1).max(dim=1, keepdim=True)
        
        # Weight by confidence
        H_ER = conf_ER * h_ER
        H_IR = conf_IR * h_IR
        
        # Calculate dynamic weights
        weights = F.softmax(self.weight_net(torch.cat([H_ER, H_IR], dim=-1)), dim=1)
        W_ER = weights[:, 0].unsqueeze(1)
        W_IR = weights[:, 1].unsqueeze(1)
        
        # Combine predictions
        Y_hat_DM = W_ER * Y_hat_ER + W_IR * Y_hat_IR
        
        return Y_hat_DM

# ============================================================================
# MAIN OODML MODEL
# ============================================================================
class OODML(nn.Module):
    def __init__(self, input_dim=1024, n_classes=3, K=5, embed_dim=512,
                 pseudo_bag_size=512, tau=1.0, dropout=0.3, heads=8, use_ddm=True):
        super(OODML, self).__init__()
        self.K = K
        self.pseudo_bag_size = pseudo_bag_size
        self.tau = tau
        self.n_classes = n_classes
        self.use_ddm = use_ddm
        
        # Modules
        self.spff = SPFF(input_dim, embed_dim, K, dropout, n_classes, heads)
        self.eirl = EIRL(embed_dim, n_classes, dropout, heads)
        
        if self.use_ddm:
            self.ddm = DDM(embed_dim, n_classes)
    
    def forward(self, all_feats, bag_label=None):
        """
        all_feats: [N, input_dim] - all instance features
        bag_label: scalar - bag label (0, 1, or 2 for 3-class)
        """
        # Randomly shuffle and create pseudo-bags
        perm = torch.randperm(all_feats.shape[0], device=all_feats.device)
        shuffled_feats = all_feats[perm]
        
        # Dynamic pseudo-bag size (from PAMIL)
        actual_pseudo_bag_size = min(self.pseudo_bag_size, shuffled_feats.shape[0])
        pseudo_bags = list(shuffled_feats.split(actual_pseudo_bag_size, dim=0))
        
        # Initialize adaptive memory bank
        adaptive_memory_bank = []
        pseudo_bag_tokens = []
        pseudo_label_preds = []
        
        # Process each pseudo-bag
        for v_t in pseudo_bags:
            if v_t.shape[0] == 0:
                continue
            
            # Get pseudo-bag token and prediction
            F_t, y_hat_t = self.spff(v_t, adaptive_memory_bank)
            
            pseudo_bag_tokens.append(F_t)
            pseudo_label_preds.append(y_hat_t)
            
            # Update memory bank (Equation 5 in paper)
            adaptive_memory_bank.append({
                'token': F_t.detach(),
                'pred': y_hat_t.detach()
            })
            
            # Sort by prediction confidence and keep top K
            adaptive_memory_bank = sorted(
                adaptive_memory_bank,
                key=lambda x: x['pred'].item() if isinstance(x['pred'], torch.Tensor) else x['pred'],
                reverse=True
            )[:self.K]
        
        if not pseudo_bag_tokens:
            # Return dummy outputs if no pseudo-bags
            empty_shape = (1, self.n_classes)
            zeros = torch.zeros(empty_shape, device=all_feats.device, requires_grad=True)
            return {
                "Y_hat_ER": zeros,
                "Y_hat_IR": zeros,
                "Y_hat_DM": zeros,
                "pseudo_label_preds": torch.tensor([]),
                "drpl_pseudo_labels": torch.tensor([])
            }
        
        # Stack all pseudo-bag tokens
        F_all = torch.stack(pseudo_bag_tokens, dim=0)  # [T, embed_dim]
        
        # EIRL: Get explicit and implicit representations
        h_ER, Y_hat_ER, A_ER, h_IR, Y_hat_IR, A_IR = self.eirl(F_all)
        
        # DDM: Dynamic decision making
        if self.use_ddm:
            Y_hat_DM = self.ddm(h_ER, Y_hat_ER, h_IR, Y_hat_IR)
        else:
            Y_hat_DM = 0.5 * Y_hat_ER + 0.5 * Y_hat_IR
        
        # DRPL: Generate pseudo-labels based on bag label
        with torch.no_grad():
            if bag_label is not None and bag_label.item() > 0:  # Positive bag (class 1 or 2)
                # For 3-class, positive bags are class 1 (atypical) or 2 (malignant)
                # Use the probability of being positive (non-benign)
                prob_ER = F.softmax(Y_hat_ER, dim=1)
                prob_IR = F.softmax(Y_hat_IR, dim=1)
                
                # Sum probabilities of positive classes (1 and 2)
                pos_prob_ER = prob_ER[:, 1:].sum(dim=1, keepdim=True)  # [1, 1]
                pos_prob_IR = prob_IR[:, 1:].sum(dim=1, keepdim=True)  # [1, 1]
                
                # Expand to match number of pseudo-bags
                pos_prob_ER = pos_prob_ER.expand(-1, len(pseudo_bag_tokens)).squeeze(0)
                pos_prob_IR = pos_prob_IR.expand(-1, len(pseudo_bag_tokens)).squeeze(0)
                
                # Normalize attention scores for stability (from PAMIL)
                A_ER_norm = A_ER / (A_ER.sum() + 1e-8)
                A_IR_norm = A_IR / (A_IR.sum() + 1e-8)
                
                # Weight by attention scores (Equation 12)
                weighted_scores = A_ER_norm * pos_prob_ER + A_IR_norm * pos_prob_IR
                drpl_y_t = torch.sigmoid(weighted_scores / self.tau)
            else:  # Negative bag (class 0)
                drpl_y_t = torch.zeros(len(pseudo_bag_tokens), device=all_feats.device)
        
        return {
            "Y_hat_ER": Y_hat_ER,
            "Y_hat_IR": Y_hat_IR,
            "Y_hat_DM": Y_hat_DM,
            "pseudo_label_preds": torch.stack(pseudo_label_preds),
            "drpl_pseudo_labels": drpl_y_t,
        }

