import torch
from marble_solitaire.model import SolitaireNet


def export_to_onnx(model, output_path, opset_version=17):
    """Export model to ONNX format for browser inference."""
    model.eval()
    dummy_input = torch.randn(1, 2, 7, 7)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["board"],
        output_names=["policy", "value"],
        dynamic_axes={
            "board": {0: "batch"},
            "policy": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=opset_version,
    )


GEN_LABELS = {
    1: "gen1_random",
    5: "gen2_novice",
    15: "gen3_apprentice",
    30: "gen4_skilled",
    50: "gen5_expert",
}


def export_all_checkpoints(models_dir="models", output_dir="web/public/models"):
    """Export all 5 training checkpoints to ONNX for the web UI."""
    import os
    from marble_solitaire.train import load_checkpoint

    os.makedirs(output_dir, exist_ok=True)
    exported = []

    for iteration, label in GEN_LABELS.items():
        pt_path = os.path.join(models_dir, f"iter_{iteration:03d}.pt")
        if not os.path.exists(pt_path):
            print(f"  Skipping {label}: {pt_path} not found")
            continue

        onnx_path = os.path.join(output_dir, f"{label}.onnx")
        model = load_checkpoint(pt_path)
        export_to_onnx(model, onnx_path)
        size_kb = os.path.getsize(onnx_path) / 1024
        print(f"  Exported {label}: {onnx_path} ({size_kb:.0f} KB)")
        exported.append(onnx_path)

    return exported


if __name__ == "__main__":
    export_all_checkpoints()
