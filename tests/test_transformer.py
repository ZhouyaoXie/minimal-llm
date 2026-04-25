import unittest

import torch

from model.transformer import DecoderBlock, FeedForward


class TestDecoderBlock(unittest.TestCase):
    def test_decoder_block_preserves_shape_and_is_finite(self) -> None:
        torch.manual_seed(0)
        block = DecoderBlock(
            n_head=2,
            d_head=4,
            d_hidden=32,
            attn_dropout=0.0,
            ffn_dropout=0.0,
        )
        X = torch.randn(2, 5, 8)

        out = block(X)

        self.assertEqual(out.shape, X.shape)
        self.assertTrue(torch.isfinite(out).all())

    def test_decoder_block_backward_flows_through_parameters(self) -> None:
        torch.manual_seed(0)
        block = DecoderBlock(
            n_head=2,
            d_head=4,
            d_hidden=32,
            attn_dropout=0.0,
            ffn_dropout=0.0,
        )
        X = torch.randn(2, 5, 8, requires_grad=True)

        out = block(X)
        out.sum().backward()

        self.assertIsNotNone(X.grad)
        self.assertTrue(torch.isfinite(X.grad).all())

        params_with_grad = {
            name
            for name, param in block.named_parameters()
            if param.requires_grad and param.grad is not None
        }
        expected_params = {
            "masked_attn.W_q.weight",
            "masked_attn.W_o.weight",
            "ffn.linear1.weight",
            "ffn.linear2.weight",
            "attn_layer_norm.g",
            "ffn_layer_norm.g",
        }

        self.assertTrue(expected_params.issubset(params_with_grad))


class TestFeedForward(unittest.TestCase):
    def test_feedforward_preserves_model_dimension_and_is_finite(self) -> None:
        torch.manual_seed(0)
        ffn = FeedForward(d_model=8, d_hidden=32, dropout=0.0)
        X = torch.randn(2, 5, 8)

        out = ffn(X)

        self.assertEqual(out.shape, X.shape)
        self.assertTrue(torch.isfinite(out).all())

    def test_feedforward_backward_flows_through_linear_layers(self) -> None:
        torch.manual_seed(0)
        ffn = FeedForward(d_model=8, d_hidden=32, dropout=0.0)
        X = torch.randn(2, 5, 8, requires_grad=True)

        out = ffn(X)
        out.sum().backward()

        self.assertIsNotNone(X.grad)
        self.assertTrue(torch.isfinite(X.grad).all())
        self.assertIsNotNone(ffn.linear1.weight.grad)
        self.assertIsNotNone(ffn.linear2.weight.grad)
        self.assertTrue(torch.isfinite(ffn.linear1.weight.grad).all())
        self.assertTrue(torch.isfinite(ffn.linear2.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
