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


class GPT(torch.nn.Module):
    """
    A GPT-style decoder-only Transformer class.

    Args:
        n_block: number of decoder blocks
        d_model: latent model dimension
        n_head: number of attention heads (must be a factor of d_model)
        d_hidden: hidden dimension used by FFN
        vocab_size: vocabulary size
        max_seq_len: max input sequence length
        attn_dropout: attention dropout probability; applied after attention output projection
        ffn_dropout: FFN dropout probability; applied after activation and after second linear layer
    """

    def __init__(
        self,
        n_block,
        d_model,
        n_head,
        d_hidden,
        vocab_size,
        max_seq_len=8196,
        attn_dropout=0.0,
        ffn_dropout=0.0,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        assert (
            d_model % n_head == 0
        ), f"expect d_model to be divisible by n_head, got {d_model}, {n_head} instead"
        d_head = d_model // n_head

        self.token_embedding = torch.nn.Embedding(vocab_size, d_model)
        self.position_embedding = torch.nn.Embedding(max_seq_len, d_model)

        self.decoder_layers = torch.nn.ModuleList(
            DecoderBlock(
                n_head=n_head,
                d_head=d_head,
                d_hidden=d_hidden,
                attn_dropout=attn_dropout,
                ffn_dropout=ffn_dropout,
            )
            for _ in range(n_block)
        )

        # additional layer norm after final attention
        self.norm = LayerNorm(d_model)

        self.lm_head = torch.nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        """
        Forward pass through a decoder-only Transformer
        input_ids -> token embeddings + positional embedding -> decoder layers

        Args:
            input_ids: torch.Tensor of shape (batch_size, seq_len)

        Returns:
            torch.Tensor of shape (batch_size, seq_len, vocab_size)
        """
        assert (
            input_ids.ndim == 2
        ), f"expect input_ids to have shape (batch_size, seq_len), got rank={input_ids.ndim} instead"
        assert (
            input_ids.shape[-1] <= self.max_seq_len
        ), f"input length cannot exceed self.max_seq_len, got {input_ids.shape[-1]}, {self.max_seq_len} instead"

        # get embeddings from token_ids
        bs, seq_len = input_ids.shape
        token_emb = self.token_embedding(input_ids)
        pos_inputs = torch.arange(seq_len, device=input_ids.device)
        pos_emb = self.position_embedding(pos_inputs)
        X = token_emb + pos_emb

        # decoder blocks forward pass
        for i, decoder_layer in enumerate(self.decoder_layers):
            X = decoder_layer(X)

        # layer normalization + projection head
        out_norm = self.norm(X)
        logits = self.lm_head(out_norm)

        return logits
