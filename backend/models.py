"""
INS/GPS Fusion — model architectures.

These class definitions are copied verbatim (architecture-wise) from the
training notebook (INS_GPS_Fusion_Config3_Final.ipynb) so that the saved
state_dicts load without key mismatches. The hyperparameters below are
hardcoded to the values the notebook actually trained with
(FAST_MODE=True): HIDDEN=32, D_MODEL=32, D_STATE=8, N_LAYERS=1.

Do not change these constants unless you re-train with different settings
and re-export the .pkl artifacts to match.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Fixed hyperparameters (must match training run) ─────────────────────
WINDOW = 50
STRIDE = 2
HIDDEN = 32
D_MODEL = 32
D_STATE = 8
N_LAYERS = 1
N_OUT = 2  # (v_N, v_E)

FEATURE_COLS_DESCRIPTION = [
    "yaw_rate", "accel_longitudinal", "accel_lateral", "dt",
    "wheel_speed_FL", "wheel_speed_FR", "wheel_speed_RL", "wheel_speed_RR",
    "vehicle_speed", "steering_angle",
    "heading_sin", "heading_cos", "gps_velocity",
]
N_FEAT = len(FEATURE_COLS_DESCRIPTION)  # 13


# ── Recurrent models ──────────────────────────────────────────────────
class RNNModel(nn.Module):
    def __init__(self, n_feat, hidden=HIDDEN, layers=N_LAYERS, drop=0.15):
        super().__init__()
        self.rnn = nn.RNN(n_feat, hidden, num_layers=layers,
                           batch_first=True, dropout=drop if layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                   nn.Dropout(drop), nn.Linear(hidden, N_OUT))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(self.norm(out[:, -1]))


class LSTMModel(nn.Module):
    def __init__(self, n_feat, hidden=HIDDEN, layers=N_LAYERS, drop=0.15):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=layers,
                             batch_first=True, dropout=drop if layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                   nn.Dropout(drop), nn.Linear(hidden, N_OUT))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.norm(out[:, -1]))


class GRUModel(nn.Module):
    def __init__(self, n_feat, hidden=HIDDEN, layers=N_LAYERS, drop=0.15):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, num_layers=layers,
                           batch_first=True, dropout=drop if layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                   nn.Dropout(drop), nn.Linear(hidden, N_OUT))

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(self.norm(out[:, -1]))


class BiRNNModel(nn.Module):
    def __init__(self, n_feat, hidden=HIDDEN, layers=N_LAYERS, drop=0.15):
        super().__init__()
        self.rnn = nn.RNN(n_feat, hidden, num_layers=layers, batch_first=True,
                           bidirectional=True, dropout=drop if layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden * 2)
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(),
                                   nn.Dropout(drop), nn.Linear(hidden, N_OUT))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(self.norm(out[:, -1]))


class BiLSTMModel(nn.Module):
    def __init__(self, n_feat, hidden=HIDDEN, layers=N_LAYERS, drop=0.15):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=layers, batch_first=True,
                             bidirectional=True, dropout=drop if layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden * 2)
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(),
                                   nn.Dropout(drop), nn.Linear(hidden, N_OUT))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.norm(out[:, -1]))


# ── Transformer (encoder-only, CLS token) ────────────────────────────
class SinusoidalPE(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                         (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerModel(nn.Module):
    def __init__(self, n_feat, d_model=D_MODEL, n_heads=4,
                 n_layers=N_LAYERS, dim_ff=128, drop=0.15):
        super().__init__()
        self.proj = nn.Linear(n_feat, d_model)
        self.cls_tok = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pe = SinusoidalPE(d_model, max_len=WINDOW + 1)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=drop, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model),
                                   nn.Linear(d_model, d_model), nn.GELU(),
                                   nn.Dropout(drop), nn.Linear(d_model, N_OUT))

    def forward(self, x):
        B = x.size(0)
        x = self.proj(x)
        cls = self.cls_tok.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pe(x)
        x = self.encoder(x)
        return self.head(x[:, 0])


# ── Temporal Fusion Transformer (simplified) ─────────────────────────
class GatedResidualNetwork(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden, out_dim)
        self.drop = nn.Dropout(drop)
        self.gate = nn.Linear(out_dim, out_dim)
        self.sig = nn.Sigmoid()
        self.norm = nn.LayerNorm(out_dim)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        res = self.skip(x)
        h = self.elu(self.fc1(x))
        h = self.drop(self.fc2(h))
        g = self.sig(self.gate(h))
        return self.norm(g * h + res)


class TFTModel(nn.Module):
    def __init__(self, n_feat, hidden=HIDDEN, n_heads=2, n_layers=N_LAYERS, drop=0.15):
        super().__init__()
        self.input_grn = GatedResidualNetwork(n_feat, hidden, hidden, drop)
        self.lstm_enc = nn.LSTM(hidden, hidden, batch_first=True)
        self.enrich = GatedResidualNetwork(hidden, hidden, hidden, drop)
        self.attn = nn.MultiheadAttention(hidden, n_heads, dropout=drop, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden)
        self.ff_grn = GatedResidualNetwork(hidden, hidden, hidden, drop)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                   nn.Dropout(drop), nn.Linear(hidden, N_OUT))

    def forward(self, x):
        x = self.input_grn(x)
        x, _ = self.lstm_enc(x)
        x = self.enrich(x)
        a, _ = self.attn(x, x, x)
        x = self.attn_norm(x + a)
        x = self.ff_grn(x)
        return self.head(x[:, -1])


# ── Mamba (Selective SSM / S6) — pure PyTorch, CPU-safe ──────────────
class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=D_STATE):
        super().__init__()
        self.d_model, self.d_state = d_model, d_state
        self.dt_rank = max(1, d_model // 16)
        self.x_proj = nn.Linear(d_model, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)
        A = torch.arange(1, d_state + 1).float().unsqueeze(0).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        B_sz, L, D = x.shape
        N = self.d_state
        x_dbl = self.x_proj(x)
        dt, B_mat, C_mat = x_dbl.split([self.dt_rank, N, N], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log.float())
        dt_A = dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        A_bar = torch.exp(dt_A)
        B_bar = dt.unsqueeze(-1) * B_mat.unsqueeze(2)
        h = torch.zeros(B_sz, D, N, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)
            y_t = (h * C_mat[:, t].unsqueeze(1)).sum(-1)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)
        return y + x * self.D.unsqueeze(0).unsqueeze(0)


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=D_STATE, expand=2, d_conv=4, drop=0.1):
        super().__init__()
        d_inner = d_model * expand
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv - 1, groups=d_inner)
        self.ssm = SelectiveSSM(d_inner, d_state)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        res = x
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :x_in.size(1)].transpose(1, 2)
        x_conv = F.silu(x_conv)
        x_ssm = self.ssm(x_conv)
        x_out = x_ssm * F.silu(z)
        return self.norm(self.drop(self.out_proj(x_out)) + res)


class MambaModel(nn.Module):
    def __init__(self, n_feat, d_model=D_MODEL, d_state=D_STATE,
                 n_layers=N_LAYERS, expand=2, d_conv=4, drop=0.1):
        super().__init__()
        self.proj = nn.Linear(n_feat, d_model)
        self.blocks = nn.ModuleList([MambaBlock(d_model, d_state, expand, d_conv, drop)
                                      for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                   nn.Dropout(drop), nn.Linear(d_model, N_OUT))

    def forward(self, x):
        x = self.proj(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x[:, -1]))


# ── Mamba-2 (Structured State Space Duality) ──────────────────────────
class SSDBlock(nn.Module):
    def __init__(self, d_model, d_state=D_STATE, n_heads=4, expand=2, drop=0.1):
        super().__init__()
        d_inner = d_model * expand
        self.n_heads = n_heads
        head_dim = d_inner // n_heads
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, 4, padding=3, groups=d_inner)
        self.A_log = nn.Parameter(torch.zeros(n_heads))
        self.dt_proj = nn.Linear(head_dim, head_dim, bias=True)
        self.BC_proj = nn.Linear(head_dim, 2 * d_state, bias=False)
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.drop_layer = nn.Dropout(drop)
        self.head_dim = head_dim
        self.d_state = d_state

    def forward(self, x):
        res = x
        B_sz, L, _ = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = F.silu(x_conv)
        x_heads = x_conv.view(B_sz, L, self.n_heads, self.head_dim)
        A = -torch.exp(self.A_log).float()
        ys = []
        for hi in range(self.n_heads):
            xh = x_heads[:, :, hi, :]
            bc = self.BC_proj(xh)
            B_mat, C_mat = bc.split(self.d_state, dim=-1)
            dt = F.softplus(self.dt_proj(xh))
            a_bar = torch.exp(dt * A[hi])
            h = torch.zeros(B_sz, self.head_dim, self.d_state, device=x.device, dtype=x.dtype)
            head_ys = []
            for t in range(L):
                h = a_bar[:, t].unsqueeze(-1) * h + dt[:, t].unsqueeze(-1) * B_mat[:, t].unsqueeze(1) * xh[:, t].unsqueeze(-1)
                head_ys.append((h * C_mat[:, t].unsqueeze(1)).sum(-1))
            ys.append(torch.stack(head_ys, dim=1))
        y = torch.cat(ys, dim=-1)
        y = y + x_in * self.D.unsqueeze(0).unsqueeze(0)
        out = y * F.silu(z)
        return self.norm(self.drop_layer(self.out_proj(out)) + res)


class Mamba2Model(nn.Module):
    def __init__(self, n_feat, d_model=D_MODEL, d_state=D_STATE,
                 n_layers=N_LAYERS, n_heads=4, expand=2, drop=0.1):
        super().__init__()
        self.proj = nn.Linear(n_feat, d_model)
        self.blocks = nn.ModuleList([SSDBlock(d_model, d_state, n_heads, expand, drop)
                                      for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                   nn.Dropout(drop), nn.Linear(d_model, N_OUT))

    def forward(self, x):
        x = self.proj(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x[:, -1]))


# ── BiMamba (bidirectional Mamba) ─────────────────────────────────────
class BiMambaModel(nn.Module):
    def __init__(self, n_feat, d_model=D_MODEL, d_state=D_STATE,
                 n_layers=N_LAYERS, expand=2, d_conv=4, drop=0.1):
        super().__init__()
        self.proj = nn.Linear(n_feat, d_model)
        self.fwd_blocks = nn.ModuleList([MambaBlock(d_model, d_state, expand, d_conv, drop)
                                          for _ in range(n_layers)])
        self.bwd_blocks = nn.ModuleList([MambaBlock(d_model, d_state, expand, d_conv, drop)
                                          for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model * 2)
        self.head = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(),
                                   nn.Dropout(drop), nn.Linear(d_model, N_OUT))

    def forward(self, x):
        x = self.proj(x)
        fwd = x
        for blk in self.fwd_blocks:
            fwd = blk(fwd)
        bwd = x.flip(1)
        for blk in self.bwd_blocks:
            bwd = blk(bwd)
        bwd = bwd.flip(1)
        combined = torch.cat([fwd[:, -1], bwd[:, -1]], dim=-1)
        return self.head(self.norm(combined))


# ── S4 (Structured State Space, diagonal / S4D) — pure PyTorch ───────
class S4DKernel(nn.Module):
    def __init__(self, d_model, d_state=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        A = 0.5 * torch.ones(d_model, d_state)
        A[:, 1:] = torch.arange(1, d_state).unsqueeze(0)
        A = -A + 1j * math.pi * torch.arange(d_state).unsqueeze(0)
        self.A_real = nn.Parameter(A.real)
        self.A_imag = nn.Parameter(A.imag)
        self.C = nn.Parameter(torch.randn(d_model, d_state, dtype=torch.cfloat))
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

    def forward(self, L):
        dt = torch.exp(self.log_dt)
        A = torch.complex(self.A_real, self.A_imag)
        A_dt = A * dt.unsqueeze(-1)
        roots = torch.exp(A_dt)
        power = torch.arange(L, device=A_dt.device)
        V = roots.unsqueeze(-1) ** power.unsqueeze(0).unsqueeze(0)
        B_bar = (roots - 1.0) / A
        K = (self.C.unsqueeze(-1) * B_bar.unsqueeze(-1) * V).sum(dim=1)
        return K.real * 2


class S4Block(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.kernel = S4DKernel(d_model, d_state)
        self.D = nn.Parameter(torch.randn(d_model))
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_model * 2)
        self.linear2 = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        B, L, D = x.shape
        x_in = x.transpose(1, 2)
        k = self.kernel(L)
        k_f = torch.fft.rfft(k, n=2 * L)
        x_f = torch.fft.rfft(x_in, n=2 * L)
        y_in = torch.fft.irfft(x_f * k_f, n=2 * L)[..., :L]
        y_in = y_in + x_in * self.D.unsqueeze(0).unsqueeze(-1)
        y = y_in.transpose(1, 2)
        y = self.activation(self.norm(y))
        y = self.dropout(y)
        res = y
        y = self.linear2(self.activation(self.linear1(y)))
        y = self.dropout(y)
        return x + res + y


class S4Model(nn.Module):
    def __init__(self, n_feat, d_model=D_MODEL, d_state=D_STATE, n_layers=N_LAYERS, drop=0.15):
        super().__init__()
        self.proj = nn.Linear(n_feat, d_model)
        self.blocks = nn.ModuleList([
            S4Block(d_model, d_state=d_state, dropout=drop) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(d_model, N_OUT)
        )

    def forward(self, x):
        x = self.proj(x)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x[:, -1]))


# ── BERT-style bidirectional transformer ──────────────────────────────
class BERTModel(nn.Module):
    def __init__(self, n_feat, d_model=D_MODEL, n_heads=2, n_layers=N_LAYERS, dim_ff=128, drop=0.15):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(n_feat, self.d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.pos_embedding = nn.Parameter(torch.zeros(1, WINDOW + 1, self.d_model))
        self.dropout = nn.Dropout(drop)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=n_heads, dim_feedforward=self.d_model * 4,
            dropout=drop, activation='gelu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, self.d_model), nn.GELU(),
            nn.Dropout(drop), nn.Linear(self.d_model, N_OUT))
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x_embed = self.embedding(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_seq = torch.cat((cls_tokens, x_embed), dim=1)
        x_seq = x_seq + self.pos_embedding[:, :seq_len + 1, :]
        x_seq = self.dropout(x_seq)
        encoder_out = self.transformer_encoder(x_seq)
        cls_out = encoder_out[:, 0, :]
        return self.fc(cls_out)


MODEL_NAMES = ['RNN', 'LSTM', 'GRU', 'BiRNN', 'BiLSTM',
               'Transformer', 'BERT', 'TFT', 'S4', 'Mamba', 'Mamba2', 'BiMamba']

MODEL_CLASSES = {
    'RNN': RNNModel, 'LSTM': LSTMModel, 'GRU': GRUModel,
    'BiRNN': BiRNNModel, 'BiLSTM': BiLSTMModel,
    'Transformer': TransformerModel, 'BERT': BERTModel, 'TFT': TFTModel,
    'S4': S4Model, 'Mamba': MambaModel, 'Mamba2': Mamba2Model, 'BiMamba': BiMambaModel,
}

MODEL_PALETTE = {
    'RNN': '#2196F3', 'LSTM': '#4CAF50', 'GRU': '#FF9800',
    'BiRNN': '#9C27B0', 'BiLSTM': '#E91E63', 'Transformer': '#00BCD4',
    'BERT': '#FF5722', 'TFT': '#795548', 'S4': '#607D8B',
    'Mamba': '#F44336', 'Mamba2': '#3F51B5', 'BiMamba': '#009688',
}


def make_model(name: str, n_feat: int = N_FEAT, **kwargs) -> nn.Module:
    """Create a model instance, forwarding any extra kwargs (hidden, d_model,
    d_state, n_layers, n_heads, dim_ff, etc.) to override the constructor defaults."""
    if name not in MODEL_CLASSES:
        raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_NAMES}")
    return MODEL_CLASSES[name](n_feat, **kwargs)
