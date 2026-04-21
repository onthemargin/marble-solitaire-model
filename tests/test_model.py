import pytest
import torch
from marble_solitaire.model import SolitaireNet


class TestModelArchitecture:
    @pytest.fixture
    def model(self):
        return SolitaireNet()

    def test_output_shapes(self, model):
        x = torch.randn(4, 2, 7, 7)
        policy, value = model(x)
        assert policy.shape == (4, 196)
        assert value.shape == (4, 1)

    def test_single_input(self, model):
        x = torch.randn(1, 2, 7, 7)
        policy, value = model(x)
        assert policy.shape == (1, 196)
        assert value.shape == (1, 1)

    def test_value_range(self, model):
        model.eval()
        x = torch.randn(10, 2, 7, 7)
        with torch.no_grad():
            _, value = model(x)
        assert (value >= -1.0).all()
        assert (value <= 1.0).all()

    def test_deterministic_eval(self, model):
        model.eval()
        x = torch.randn(1, 2, 7, 7)
        with torch.no_grad():
            p1, v1 = model(x)
            p2, v2 = model(x)
        torch.testing.assert_close(p1, p2)
        torch.testing.assert_close(v1, v2)

    def test_parameter_count(self, model):
        total = sum(p.numel() for p in model.parameters())
        assert total < 200_000
        assert total > 10_000

    def test_gradient_flow(self, model):
        x = torch.randn(1, 2, 7, 7)
        policy, value = model(x)
        loss = policy.sum() + value.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
