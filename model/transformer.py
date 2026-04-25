from typing import Optional
import torch
from torch.nn.functional import gelu
from model.attention import MultiHeadAttention


class LayerNorm(torch.nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.g = torch.nn.Parameter(torch.ones(d_model))
        self.b = torch.nn.Parameter(torch.zeros(d_model))
        self.d_model = d_model
        self.eps = eps

    def forward(self, X):
        """
        Applies layer normalization (https://arxiv.org/pdf/1607.06450) on inputs.

        Args:
            X: torch.Tensor of shape (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor of shape (batch_size, seq_len, d_model)
        """
        assert (
            X.shape[-1] == self.d_model
        ), f"expect X.shape[-1] to equal self.d_model, got {X.shape[-1]}, {self.d_model} instead"
        mu = torch.mean(X, dim=-1, keepdim=True)
        var = torch.var(X, dim=-1, keepdim=True, correction=0)
        # use var + rsqrt form for numerical stability and efficiency
        normalized_X = self.g * (X - mu) * torch.rsqrt(var + self.eps) + self.b
        return normalized_X


class FeedForward(torch.nn.Module):
    def __init__(self, d_model, d_hidden, dropout=0.0):
        super().__init__()
        self.d_model, self.d_hidden = d_model, d_hidden
        self.linear1 = torch.nn.Linear(d_model, d_hidden)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(d_hidden, d_model)

    def forward(self, X):
        """
        FFN(X) = Linear2(Gelu(Linear1(X)))

        Args:
            X: torch.Tensor of shape (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor of the same shape as X
        """
        assert (
            X.shape[-1] == self.d_model
        ), f"expect X.shape[-1] == d_model, got {X.shape[-1]}, {self.d_model} instead"

        out1 = gelu(self.linear1(X))
        out1 = self.dropout(out1)
        out2 = self.linear2(out1)
        out2 = self.dropout(out2)
        return out2


class DecoderBlock(torch.nn.Module):
    def __init__(self, n_head, d_head, d_hidden, attn_dropout=0.0, ffn_dropout=0.0):
        super().__init__()
        self.n_head, self.d_head, self.d_model = n_head, d_head, n_head * d_head
        self.d_hidden = d_hidden
        self.masked_attn = MultiHeadAttention(n_head, d_head, dropout=attn_dropout)
        self.attn_layer_norm = LayerNorm(self.d_model)
        self.ffn = FeedForward(self.d_model, d_hidden, dropout=ffn_dropout)
        self.ffn_layer_norm = LayerNorm(self.d_model)

    def forward(
        self,
        X,
        masking=True,
        key_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        Runs pre-norm decoder block: Norm -> MHA -> Residual -> Norm -> FFN -> Residual (X = X + sublayer(norm(X)))

        Args:
            X: torch.tensor of shape (batch_size, seq_len, d_model)
            masking: whether to block positions where the key index is after the query index. Defaults to True for decoder block
            key_padding_mask: torch.Tensor of boolean with shape (batch_size, seq_len), applies padding to masked key positions if value is True

        Returns:
            torch.Tensor of shape X
        """
        X_norm = self.attn_layer_norm(X)
        attn_out = self.masked_attn(
            X_norm,
            masking=masking,
            key_padding_mask=key_padding_mask,
        )
        X = X + attn_out
        X_norm = self.ffn_layer_norm(X)
        ffn_out = self.ffn(X_norm)
        X = X + ffn_out
        return X
