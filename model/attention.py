import torch

class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V) -> torch.Tensor:
        """
        Calculate attention score based on Q, K, V matrices 

        Args:
            Q: torch.Tensor of shape (batch_size, l_q, d_k)
            K: torch.Tensor of shape (batch_size, l_kv, d_k)
            V: torch.Tensor of shape (batch_size, l_kv, d_v)

        Returns: 
            attn_score: shape (batch_size, l_q, d_v)
        """
        assert Q.ndim == K.ndim == V.ndim, f"expected Q, K, V to have the same rank, got {Q.ndim}, {K.ndim}, {V.ndim} instead"
        assert Q.shape[0] == K.shape[0] == V.shape[0], f"expected Q, K, V to have the same batch size, got {Q.shape[0]}, {K.shape[0]}, {V.shape[0]} instead"
        assert K.shape[-2] == V.shape[-2], f"expected K, V to have the same seq_len, got {K.shape[-2]}, {V.shape[-2]} instead"
        assert Q.shape[-1] == K.shape[-1], f"expected Q, K to have the same latent dimension, got {Q.shape[-1]}, {K.shape[-1]} instead"

        similarity_score = torch.matmul(Q, K.transpose(-2, -1)) / (K.shape[-1] ** 0.5)
        probs = torch.softmax(similarity_score, dim = -1)
        attn_score = torch.matmul(probs, V)
        return attn_score 

    def manual_softmax(self, x):
        # keeping the manual implementation of softmax for reference
        max_score = torch.max(x, dim = -1, keepdim = True).values
        adj_score = x - max_score
        exp_score = adj_score.exp()
        probs = exp_score / torch.sum(exp_score, dim = -1, keepdim = True)
        return probs


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, n_head, d_head):
        """
        Params:
            n_head: number of attention heads 
            d_head: inner dimension of each head 
        """
        super().__init__()
        self.n_head = n_head
        self.d_head = d_head
        self.d_model = n_head * d_head

        # initialize W_q, W_k, W_v, W_o
        self.W_q = torch.nn.Linear(self.d_model, self.d_model)
        self.W_k = torch.nn.Linear(self.d_model, self.d_model)
        self.W_v = torch.nn.Linear(self.d_model, self.d_model)
        self.W_o = torch.nn.Linear(self.d_model, self.d_model)

        self.attention = Attention()

    def forward(self, X, encoder_output = None) -> torch.Tensor:
        """
        Calculates multi-head attention score.

        Args:
            X: input, torch.Tensor of shape (batch_size, l_q, d_model)
            encoder_output (optional): torch.Tensor of shape (batch_size, l_kv, d_model)
                if encoder_output is None, Q, K, V are all projected from X (self attention); 
                otherwise, Q comes from X and K, V come from encoder_output (cross-attention)

        Returns:
            attn_score: torch.Tensor of shape (batch_size, l_q, d_model)
        """
        assert X.shape[-1] == self.d_model, f"expected X.shape[-1] == d_model, got {X.shape[-1]}, {self.d_model} instead"
        if encoder_output is not None: 
            assert encoder_output.shape[-1] == self.d_model, f"expected encoder_output.shape[-1] == d_model, got {encoder_output.shape[-1]}, {self.d_model} instead"
        
        bs, l_q, _ = X.shape
        if encoder_output is not None:
            l_kv = encoder_output.shape[1]
        else: 
            l_kv = l_q 

        # get projected Q, K, V
        if encoder_output is None: 
            Q, K, V = self.projectQKV(X, X, X)
        else: 
            Q, K, V = self.projectQKV(X, encoder_output, encoder_output)
        
        # reshape Q, K, V to prepare for multi-head
        Q_reshaped = Q.reshape(bs, l_q, self.n_head, self.d_head).permute(2, 0, 1, 3)
        K_reshaped = K.reshape(bs, l_kv, self.n_head, self.d_head).permute(2, 0, 1, 3)
        V_reshaped = V.reshape(bs, l_kv, self.n_head, self.d_head).permute(2, 0, 1, 3)

        attn_score = []
        for i in range(self.n_head):
            attn_score.append(self.attention(Q_reshaped[i], K_reshaped[i], V_reshaped[i]))
        attn_score = torch.cat(attn_score, dim = -1)

        projected_score = self.W_o(attn_score)
        return projected_score

    def projectQKV(self, Q_in, K_in, V_in):
        Q = self.W_q(Q_in)
        K = self.W_k(K_in)
        V = self.W_v(V_in)
        return Q, K, V