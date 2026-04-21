import pytest
import torch
import numpy as np
import os
from marble_solitaire.model import SolitaireNet
from marble_solitaire.export import export_to_onnx


class TestOnnxExport:
    def test_creates_file(self, tmp_path):
        model = SolitaireNet()
        path = str(tmp_path / "test.onnx")
        export_to_onnx(model, path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_output_matches_pytorch(self, tmp_path):
        model = SolitaireNet()
        model.eval()
        path = str(tmp_path / "test.onnx")
        export_to_onnx(model, path)

        # Compare PyTorch vs ONNX
        import onnxruntime as ort
        x = np.random.randn(1, 2, 7, 7).astype(np.float32)

        with torch.no_grad():
            pt_policy, pt_value = model(torch.from_numpy(x))

        sess = ort.InferenceSession(path)
        onnx_out = sess.run(None, {"board": x})

        np.testing.assert_allclose(pt_policy.numpy(), onnx_out[0], atol=1e-5)
        np.testing.assert_allclose(pt_value.numpy(), onnx_out[1], atol=1e-5)
