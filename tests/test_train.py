import pytest
import torch
import numpy as np
import os
from marble_solitaire.model import SolitaireNet
from marble_solitaire.train import compute_loss, train_step, save_checkpoint, load_checkpoint


class TestComputeLoss:
    def test_loss_is_scalar(self):
        model = SolitaireNet()
        policy_logits = torch.randn(4, 196)
        policy_targets = torch.softmax(torch.randn(4, 196), dim=1)
        value_pred = torch.randn(4, 1)
        value_targets = torch.randn(4, 1)
        loss = compute_loss(policy_logits, policy_targets, value_pred, value_targets)
        assert loss.dim() == 0

    def test_loss_positive(self):
        model = SolitaireNet()
        policy_logits = torch.randn(4, 196)
        policy_targets = torch.softmax(torch.randn(4, 196), dim=1)
        value_pred = torch.randn(4, 1)
        value_targets = torch.randn(4, 1)
        loss = compute_loss(policy_logits, policy_targets, value_pred, value_targets)
        assert loss.item() > 0


class TestTrainStep:
    def test_reduces_loss(self):
        model = SolitaireNet()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # Create a small batch
        states = torch.randn(8, 2, 7, 7)
        policy_targets = torch.softmax(torch.randn(8, 196), dim=1)
        value_targets = torch.randn(8, 1).clamp(-1, 1)

        loss1 = train_step(model, optimizer, states, policy_targets, value_targets)
        # Run several more steps
        for _ in range(20):
            loss2 = train_step(model, optimizer, states, policy_targets, value_targets)
        assert loss2 < loss1


class TestCheckpoint:
    def test_save_and_load(self, tmp_path):
        model = SolitaireNet()
        model.eval()
        x = torch.randn(1, 2, 7, 7)
        with torch.no_grad():
            p1, v1 = model(x)

        path = str(tmp_path / "test.pt")
        save_checkpoint(model, path)

        loaded = load_checkpoint(path)
        loaded.eval()
        with torch.no_grad():
            p2, v2 = loaded(x)

        torch.testing.assert_close(p1, p2)
        torch.testing.assert_close(v1, v2)
