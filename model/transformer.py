import torch 

class LayerNorm(torch.nn.Module):
    def __init__(self, d_model, eps = 1e-6):
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
        assert X.shape[-1] == self.d_model, f"expect X.shape[-1] to equal self.d_model, got {X.shape[-1]}, {self.d_model} instead"
        mu = torch.mean(X, dim = -1, keepdim = True)
        var = torch.var(X, dim = -1, keepdim = True, correction = 0)
        # use var + rsqrt form for numerical stability and efficiency
        normalized_X = self.g * (X - mu) * torch.rsqrt(var + self.eps) + self.b
        return normalized_X