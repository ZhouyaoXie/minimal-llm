import unittest
from model.attention import Attention
import torch


class TestAttention(unittest.TestCase):
    def test_attention_masking_snapshot(self) -> None:
        Q = torch.Tensor(
            [
                [[1.0, 1.2], [1.5, 1.7], [0.8, 0.4], [0.1, 0.1]],
                [[0.3, 0.9], [0.3, 0.6], [0.2, 0.4], [0.2, 0.2]],
            ]
        )
        K = torch.Tensor(
            [
                [[0.1, 0.2], [1.5, 1.6], [1.6, 1.8], [1.0, 1.0]],
                [[0.6, 1.8], [1.6, 1.8], [0.3, 0.1], [0.2, 0.2]],
            ]
        )
        V = torch.Tensor(
            [
                [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
                [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
            ]
        )

        attn_score = Attention()(Q, K, V, masking=True)
        self.assertEqual(attn_score.shape, (2, 4, 2))

        expected = torch.tensor(
            [
                [
                    [1.0000, 1.0000],
                    [1.9596, 1.9596],
                    [2.3361, 2.3361],
                    [2.5446, 2.5446],
                ],
                [
                    [1.0000, 1.0000],
                    [1.5528, 1.5528],
                    [1.8516, 1.8516],
                    [2.3446, 2.3446],
                ],
            ],
            dtype=attn_score.dtype,
            device=attn_score.device,
        )

        torch.testing.assert_close(
            attn_score,
            expected,
            rtol=1e-4,
            atol=1e-4,
        )

    def test_causal_mask_blocks_future_information_flow(self) -> None:
        Q = torch.tensor(
            [
                [[1.0, 1.2], [1.5, 1.7], [0.8, 0.4], [0.1, 0.1]],
                [[0.3, 0.9], [0.3, 0.6], [0.2, 0.4], [0.2, 0.2]],
            ]
        )
        K = torch.tensor(
            [
                [[0.1, 0.2], [1.5, 1.6], [1.6, 1.8], [1.0, 1.0]],
                [[0.6, 1.8], [1.6, 1.8], [0.3, 0.1], [0.2, 0.2]],
            ]
        )
        V = torch.tensor(
            [
                [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
                [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]],
            ]
        )

        attention = Attention()
        baseline = attention(Q, K, V, masking=True)

        split_idx = 1
        K_changed = K.clone()
        V_changed = V.clone()
        K_changed[:, split_idx + 1 :, :] = K_changed[:, split_idx + 1 :, :] * -3.0 + 5.0
        V_changed[:, split_idx + 1 :, :] = V_changed[:, split_idx + 1 :, :] * 7.0 - 11.0

        changed = attention(Q, K_changed, V_changed, masking=True)

        # Past/current positions should be unchanged when only future tokens change.
        torch.testing.assert_close(
            changed[:, : split_idx + 1, :],
            baseline[:, : split_idx + 1, :],
            rtol=0,
            atol=1e-6,
        )

        # Future positions are allowed to change; this guards against a vacuous test.
        self.assertFalse(torch.allclose(changed[:, split_idx + 1 :, :], baseline[:, split_idx + 1 :, :]))
