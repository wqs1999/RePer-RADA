import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.mlp import MLP


def build_multi_resolution_schedule(dataset_name='ETTh1', seq_len=672, dom_period=None):
    """
    Build dataset-adaptive multi-resolution pooling schedule (Section 3.3).
    
    The schedule is derived from the dominant period P_dom:
    - Short-term:  k_s ∈ {1, P_dom/6, P_dom/3}
    - Mid-term:    k_s ∈ {P_dom, 2*P_dom}
    - Long-term:   k_s ∈ {4*P_dom, 7*P_dom}
    
    For hourly data with P_dom=24: {1, 4, 8, 24, 168} (Table 1).
    """
    if dom_period is None:
        if dataset_name in ['ETTh1', 'ETTh2', 'ECL', 'electricity']:
            dom_period = 24
        elif dataset_name in ['ETTm1', 'ETTm2']:
            dom_period = 96
        elif dataset_name in ['weather', 'Weather']:
            dom_period = 144
        elif dataset_name in ['Solar', 'solar', 'Solar-Energy']:
            dom_period = 144
        elif dataset_name in ['traffic', 'Traffic']:
            dom_period = 24
        else:
            dom_period = 24

    kernels = [1]
    candidates = [
        max(1, dom_period // 6),
        max(1, dom_period // 3),
        dom_period,
        2 * dom_period,
        4 * dom_period,
        7 * dom_period
    ]
    for k in candidates:
        if k not in kernels and k <= seq_len // 2:
            kernels.append(k)
    kernels = sorted(list(set(kernels)))
    if len(kernels) > 5:
        step = len(kernels) / 5.0
        idxs = [min(int(i * step), len(kernels) - 1) for i in range(5)]
        kernels = sorted(list(set([kernels[i] for i in idxs])))
    return kernels


class RelativePeriodicEncoding(nn.Module):
    """
    Relative-Periodic Joint Temporal Encoding (Section 3.2).
    Implements Equations (1)-(7).
    """

    def __init__(self, max_offset_d, period_list, hidden_dim, basis_dim=64):
        super().__init__()
        self.max_offset_d = max_offset_d
        self.period_list = period_list
        self.num_periods = len(period_list)
        self.hidden_dim = hidden_dim
        self.basis_dim = basis_dim

        # Relative temporal encoding: W_rel ∈ R^{(2D+1) x d} (Equations 1-2)
        self.relative_embed = nn.Embedding(2 * max_offset_d + 1, hidden_dim)

        # Periodic encoding basis projection (Equations 3-4)
        self.period_basis_proj = nn.Linear(2, basis_dim)

        # MLP for period fusion weights: omega(i) ∈ R^P
        self.period_weight_mlp = nn.Sequential(
            nn.Linear(hidden_dim, basis_dim),
            nn.ReLU(),
            nn.Linear(basis_dim, self.num_periods)
        )

        # Project periodic features to hidden_dim
        self.period_to_hidden = nn.Linear(basis_dim, hidden_dim)

    def forward(self, seq_len, device):
        """
        Returns:
            T_joint: [seq_len, hidden_dim] — Equation (5)
            e_per:   [seq_len, basis_dim]   — periodic encoding before projection
        """
        i_ref = seq_len - 1
        time_indices = torch.arange(seq_len, device=device, dtype=torch.float32)

        # === Relative Encoding (Equations 1-2) ===
        offsets = (time_indices - i_ref).long()
        clipped = torch.clamp(offsets, -self.max_offset_d, self.max_offset_d)
        embed_idx = clipped + self.max_offset_d
        T_rel = self.relative_embed(embed_idx)  # [seq_len, hidden_dim]

        # === Periodic Encoding (Equations 3-4) ===
        period_features = []
        for p in self.period_list:
            angle = 2 * np.pi * time_indices / p
            sin_val = torch.sin(angle)
            cos_val = torch.cos(angle)
            basis = torch.stack([sin_val, cos_val], dim=-1)  # [seq_len, 2]
            projected = self.period_basis_proj(basis)        # [seq_len, basis_dim]
            period_features.append(projected)

        period_stack = torch.stack(period_features, dim=1)  # [seq_len, num_periods, basis_dim]

        # Context-aware fusion weights
        omega = self.period_weight_mlp(T_rel)  # [seq_len, num_periods]
        omega = F.softmax(omega, dim=-1)

        # Weighted periodic encoding
        e_per = torch.sum(omega.unsqueeze(-1) * period_stack, dim=1)  # [seq_len, basis_dim]

        # Project to hidden_dim
        T_per = self.period_to_hidden(e_per)  # [seq_len, hidden_dim]

        # === Joint Encoding (Equation 5) ===
        T_joint = T_rel + T_per

        return T_joint, e_per


class ResolutionAwareWeighting(nn.Module):
    """
    Resolution-Aware Dynamic Weight Aggregation (Section 3.4).
    Implements Equations (13)-(15).
    """

    def __init__(self, num_scales, basis_dim):
        super().__init__()
        self.num_scales = num_scales
        self.weight_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(basis_dim, max(1, basis_dim // 2)),
                nn.ReLU(),
                nn.Linear(max(1, basis_dim // 2), 1)
            ) for _ in range(num_scales)
        ])

    def forward(self, e_per_list):
        """
        e_per_list: list of [basis_dim] tensors (periodic context per scale)
        Returns: [num_scales] softmax-normalized weights w_s
        """
        scores = []
        for s in range(self.num_scales):
            z_s = self.weight_mlps[s](e_per_list[s]).squeeze(-1)
            scores.append(z_s)

        scores = torch.stack(scores, dim=0)  # [num_scales]
        weights = F.softmax(scores, dim=0)   # [num_scales] — Equation (14)
        return weights


class SimpleTransformerBackbone(nn.Module):
    """
    Lightweight Transformer backbone as LLM substitute.
    Uses standard PyTorch TransformerEncoder for compatibility.
    """

    def __init__(self, hidden_dim=128, num_layers=2, num_heads=4, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, inputs_embeds=None, **kwargs):
        """Mimic LlamaForCausalLM interface."""
        out = self.encoder(inputs_embeds)
        out = self.norm(out)
        return (out,)


class Model(nn.Module):
    """
    RPTF: Relative-Periodic Temporal Fusion forecasting model.
    
    Architecture pipeline (Section 3.1):
        Input X ∈ R^{B×L×N}
            │
            ├── Normalize → X̃
            │
            ├── Multi-Resolution Decomposition (Section 3.3, Eq. 8)
            │   ├── Scale 0: X̃_0 = X̃ (k=1)
            │   ├── Scale 1: X̃_1 = Pool(X̃, 4)
            │   ├── Scale 2: X̃_2 = Pool(X̃, 8)
            │   ├── Scale 3: X̃_3 = Pool(X̃, 24)
            │   └── Scale 4: X̃_4 = Pool(X̃, 168)
            │
            ├── For each scale s (Algorithm 1):
            │   ├── Temporal Encoding (Eq. 1-7) → T^joint_s, e_s
            │   ├── Tokenize + Encode (Eq. 9) → H_s
            │   ├── Concat(H_s, e_s⊗1) (Eq. 10) → H'_s
            │   ├── Transformer Forward (Eq. 12) → Y_s
            │   └── Upsample to L_0 (Eq. 11)
            │
            ├── Dynamic Weighting (Eq. 13-14)
            │   └── w_s = softmax(f_s(e_per_s))
            │
            └── Weighted Aggregation (Eq. 15)
                └── Y = Σ_s w_s · Y_s
    
    Training objective (Equations 16-19):
        L_total = λ·Σ_s L_s + (1-λ)·L_agg - η·Σ_s w_s·log(w_s)
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.token_len = configs.token_len
        self.configs = configs

        if configs.use_multi_gpu:
            self.device = f"cuda:{configs.local_rank}"
        elif configs.gpu == 'cpu':
            self.device = 'cpu'
        else:
            self.device = f"cuda:{configs.gpu}"

        # Multi-Resolution Settings (Section 3.3)
        self.pooling_kernels = getattr(configs, 'pooling_kernels', [1, 4, 8, 24, 168])
        self.num_scales = len(self.pooling_kernels)
        self.lambda_scale = getattr(configs, 'lambda_scale', 0.5)
        self.eta_entropy = getattr(configs, 'eta_entropy', 0.01)

        # Relative-Periodic encoding (Section 3.2)
        self.max_offset_d = getattr(configs, 'max_offset_d', 96)
        self.basis_dim = getattr(configs, 'basis_dim', 64)
        self.period_list = getattr(configs, 'period_list', [24])

        print(f"[RPTF] Device: {self.device}")
        print(f"[RPTF] Pooling kernels: {self.pooling_kernels}")
        print(f"[RPTF] Period list: {self.period_list}")
        print(f"[RPTF] Num scales: {self.num_scales}")

        # LLM backbone (SimpleTransformer for CPU compatibility)
        hidden_dim = getattr(configs, 'llm_hidden_dim', 128)
        num_layers = getattr(configs, 'llm_num_layers', 2)
        self.llama = SimpleTransformerBackbone(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=4,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1
        )
        self.hidden_dim = hidden_dim

        # Freeze backbone parameters
        for param in self.llama.parameters():
            param.requires_grad = False

        # Encoder / Decoder (shared across scales)
        if configs.mlp_hidden_layers == 0:
            print("[RPTF] Using linear encoder/decoder")
            self.encoder = nn.Linear(self.token_len, self.hidden_dim)
            self.decoder = nn.Linear(self.hidden_dim, self.token_len)
        else:
            print("[RPTF] Using MLP encoder/decoder")
            self.encoder = MLP(self.token_len, self.hidden_dim,
                               configs.mlp_hidden_dim, configs.mlp_hidden_layers,
                               configs.dropout, configs.mlp_activation)
            self.decoder = MLP(self.hidden_dim, self.token_len,
                               configs.mlp_hidden_dim, configs.mlp_hidden_layers,
                               configs.dropout, configs.mlp_activation)

        # === Paper modules ===
        # 1. Relative-Periodic Joint Temporal Encoding (Section 3.2)
        self.temporal_encoding = RelativePeriodicEncoding(
            max_offset_d=self.max_offset_d,
            period_list=self.period_list,
            hidden_dim=self.hidden_dim,
            basis_dim=self.basis_dim
        )

        # 2. Scale-specific context aggregation MLPs (Equation 6)
        self.scale_context_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.basis_dim),
                nn.ReLU(),
                nn.Linear(self.basis_dim, 1)
            ) for _ in range(self.num_scales)
        ])

        # 3. Resolution-Aware Dynamic Weighting (Section 3.4)
        self.resolution_weighting = ResolutionAwareWeighting(
            num_scales=self.num_scales,
            basis_dim=self.basis_dim
        )

        # Projection for concatenated embeddings [2d] → [d] (Equation 10)
        self.concat_proj = nn.Linear(2 * self.hidden_dim, self.hidden_dim)

    def nonoverlap_avg_pool(self, x, kernel_size):
        """Non-overlapping average pooling (Equation 8)."""
        B, L, N = x.shape
        if kernel_size == 1:
            return x
        L_s = L // kernel_size
        if L_s == 0:
            return x
        x = x[:, :L_s * kernel_size, :]
        x = x.view(B, L_s, kernel_size, N)
        return x.mean(dim=2)

    def linear_upsample(self, y_s, target_length):
        """Linear upsampling (Equation 11)."""
        y_s = y_s.permute(0, 2, 1)
        y_up = F.interpolate(y_s, size=target_length, mode='linear', align_corners=True)
        return y_up.permute(0, 2, 1)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """Inference: uses finest scale (s=0) for efficiency."""
        B, L, N = x_enc.shape
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        Y_0 = self._process_scale(x_enc, 0, B, L, N)
        Y_0 = Y_0 * stdev + means
        return Y_0

    def _process_scale(self, x_norm, scale_idx, B, L, N):
        """Single-scale processing (Algorithm 1)."""
        k_s = self.pooling_kernels[scale_idx]

        # Step 1: Pool (Equation 8)
        x_s = x_norm if k_s == 1 else self.nonoverlap_avg_pool(x_norm, k_s)
        L_s = x_s.shape[1]

        # Step 2: Temporal encoding (Equations 1-7)
        T_joint_s, e_per_s = self.temporal_encoding(L_s, x_norm.device)

        # Scale-specific context aggregation (Equation 6)
        scores = self.scale_context_mlps[scale_idx](T_joint_s).squeeze(-1)
        alpha = F.softmax(scores, dim=0)
        e_s = torch.sum(alpha.unsqueeze(-1) * T_joint_s, dim=0)

        # Step 3: Tokenize and encode (Equation 9)
        x_s_perm = x_s.permute(0, 2, 1)
        x_s_flat = x_s_perm.reshape(B * N, L_s)

        # Split into fixed-length tokens (pad to multiple of token_len)
        token_num = (L_s + self.token_len - 1) // self.token_len
        padded_len = token_num * self.token_len
        if padded_len > L_s:
            pad_len = padded_len - L_s
            padding = torch.zeros(B * N, pad_len, device=x_s_flat.device, dtype=x_s_flat.dtype)
            x_s_padded = torch.cat([x_s_flat, padding], dim=-1)
        else:
            x_s_padded = x_s_flat
        x_s_tokens = x_s_padded.view(B * N, token_num, self.token_len)

        H_s = self.encoder(x_s_tokens)  # [B*N, token_num, hidden_dim]

        # Step 4: Concatenate with temporal context (Equation 10)
        e_s_broadcast = e_s.view(1, 1, -1).expand(B * N, token_num, -1)
        H_prime_s = torch.cat([H_s, e_s_broadcast], dim=-1)
        H_prime_s = self.concat_proj(H_prime_s)

        # Step 5: Transformer forward (Equation 12)
        outputs = self.llama(inputs_embeds=H_prime_s)[0]

        # Step 6: Decode and reshape
        Y_s_tokens = self.decoder(outputs)
        Y_s_flat = Y_s_tokens.view(B * N, -1)
        Y_s_flat = Y_s_flat[:, :L_s]
        Y_s = Y_s_flat.view(B, N, L_s)
        Y_s = Y_s.permute(0, 2, 1)

        # Step 7: Upsample (Equation 11)
        if L_s < L:
            Y_s = self.linear_upsample(Y_s, L)
        elif L_s > L:
            Y_s = Y_s[:, :L, :]

        return Y_s

    def forward_with_loss(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, target=None):
        """Training with multi-scale loss (Equations 16-19)."""
        B, L, N = x_enc.shape

        # Normalize
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc_norm = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc_norm = x_enc_norm / stdev

        target_norm = (target - means) / stdev if target is not None else None

        # Multi-scale processing
        scale_preds = []
        e_per_list = []

        for s in range(self.num_scales):
            Y_s = self._process_scale(x_enc_norm, s, B, L, N)
            scale_preds.append(Y_s)

            k_s = self.pooling_kernels[s]
            x_s = x_enc_norm if k_s == 1 else self.nonoverlap_avg_pool(x_enc_norm, k_s)
            L_s = x_s.shape[1]
            _, e_per_s = self.temporal_encoding(L_s, x_enc.device)
            e_per_agg = e_per_s.mean(dim=0)
            e_per_list.append(e_per_agg)

        preds_stack = torch.stack(scale_preds, dim=0)

        # Resolution-aware weighting (Equations 13-14)
        weights = self.resolution_weighting(e_per_list)

        # Weighted aggregation (Equation 15)
        Y_agg = torch.sum(weights.view(-1, 1, 1, 1) * preds_stack, dim=0)
        Y_agg_denorm = Y_agg * stdev + means

        if target_norm is not None:
            # Scale-specific losses (Equation 16)
            criterion = nn.MSELoss()
            scale_losses = {}
            for s in range(self.num_scales):
                loss_s = criterion(scale_preds[s], target_norm)
                scale_losses[f'scale_{s}'] = loss_s

            # Aggregated loss (Equation 17)
            loss_agg = criterion(Y_agg, target_norm)

            # Combined loss
            loss_scale_sum = sum(scale_losses.values())
            loss = self.lambda_scale * loss_scale_sum + (1 - self.lambda_scale) * loss_agg

            # Entropy regularizer (Equation 18)
            entropy_reg = -self.eta_entropy * torch.sum(weights * torch.log(weights + 1e-8))

            # Total loss (Equation 19)
            total_loss = loss + entropy_reg

            return total_loss, Y_agg_denorm, scale_losses, weights

        return Y_agg_denorm
