"""Vertex AI Custom Training entrypoint for marble solitaire.

Usage (local test):
    python run.py --iterations 5 --episodes 5 --simulations 10

On Vertex AI, checkpoints are saved to GCS via AIP_MODEL_DIR env var.
"""
import os
import sys
import argparse
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import numpy as np
from marble_solitaire.model import SolitaireNet
from marble_solitaire.train import compute_loss, train_step, save_checkpoint
from marble_solitaire.self_play import run_episode, ReplayBuffer
from marble_solitaire.inference import solve_greedy
from marble_solitaire.export import export_to_onnx, GEN_LABELS
from marble_solitaire.board import initial_board


def upload_to_gcs(local_path, gcs_dir):
    """Upload a file to GCS if gcs_dir is a gs:// path."""
    if not gcs_dir or not gcs_dir.startswith("gs://"):
        return
    try:
        from google.cloud import storage
        # Parse gs://bucket/path
        parts = gcs_dir.replace("gs://", "").split("/", 1)
        bucket_name = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        blob_name = os.path.join(prefix, os.path.basename(local_path))

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_path)
        print(f"  -> Uploaded to gs://{bucket_name}/{blob_name}")
    except Exception as e:
        print(f"  -> GCS upload failed: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--simulations", type=int, default=400)
    parser.add_argument("--epochs-per-iter", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint-iters", type=str, default="1,15,50,150,250,350,500")
    args = parser.parse_args()

    checkpoint_iters = set(int(x) for x in args.checkpoint_iters.split(","))

    # Vertex AI sets AIP_MODEL_DIR for output
    output_dir = os.environ.get("AIP_MODEL_DIR", "/app/output")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = SolitaireNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    replay_buffer = ReplayBuffer(max_size=100000)

    total_start = time.time()

    for iteration in range(1, args.iterations + 1):
        iter_start = time.time()

        # Self-play (CPU — MCTS is sequential, GPU doesn't help much here
        # but the forward passes inside MCTS use GPU)
        model.eval()
        episode_lengths = []
        for ep in range(args.episodes):
            examples = run_episode(model, n_simulations=args.simulations)
            replay_buffer.add(examples)
            episode_lengths.append(len(examples))

        # Training on GPU
        if len(replay_buffer) >= args.batch_size:
            total_loss = 0.0
            n_steps = 0
            for _ in range(args.epochs_per_iter):
                actual_batch = min(args.batch_size, len(replay_buffer))
                states, policies, outcomes = replay_buffer.sample(actual_batch)
                states_t = torch.FloatTensor(states).to(device)
                policies_t = torch.FloatTensor(policies).to(device)
                outcomes_t = torch.FloatTensor(outcomes).unsqueeze(1).to(device)
                loss = train_step(model, optimizer, states_t, policies_t, outcomes_t)
                total_loss += loss
                n_steps += 1
            avg_loss = total_loss / n_steps
        else:
            avg_loss = float("nan")

        # Evaluate
        model.eval()
        solve_results = []
        for _ in range(10):
            board = initial_board()
            moves = solve_greedy(model, board)
            result = board
            for m in moves:
                result = result.apply_move(m)
            solve_results.append(result.count_marbles())

        avg_remaining = sum(solve_results) / len(solve_results)
        solved = sum(1 for r in solve_results if r == 1)
        elapsed = time.time() - iter_start

        print(
            f"Iter {iteration:3d}/{args.iterations} | "
            f"Loss: {avg_loss:.4f} | "
            f"Avg remaining: {avg_remaining:.1f} | "
            f"Solved: {solved}/10 | "
            f"Buffer: {len(replay_buffer)} | "
            f"Ep len: {sum(episode_lengths)/len(episode_lengths):.0f} | "
            f"Time: {elapsed:.0f}s"
        )

        # Save checkpoint
        if iteration in checkpoint_iters:
            local_path = os.path.join(output_dir, f"iter_{iteration:03d}.pt")
            # Move model to CPU for saving, then back
            model.cpu()
            save_checkpoint(model, local_path)
            model.to(device)
            print(f"  -> Saved checkpoint: {local_path}")

            # Upload to GCS if AIP_MODEL_DIR is gs://
            upload_to_gcs(local_path, os.environ.get("AIP_MODEL_DIR"))

    # Export ONNX models
    print("\nExporting ONNX models...")
    onnx_dir = os.path.join(output_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    model.cpu()
    for iteration, label in GEN_LABELS.items():
        pt_path = os.path.join(output_dir, f"iter_{iteration:03d}.pt")
        if not os.path.exists(pt_path):
            print(f"  Skipping {label}: {pt_path} not found")
            continue
        onnx_path = os.path.join(onnx_dir, f"{label}.onnx")
        checkpoint_model = SolitaireNet()
        checkpoint_model.load_state_dict(torch.load(pt_path, weights_only=True))
        export_to_onnx(checkpoint_model, onnx_path)
        size_kb = os.path.getsize(onnx_path) / 1024
        print(f"  Exported {label}: {onnx_path} ({size_kb:.0f} KB)")
        upload_to_gcs(onnx_path, os.environ.get("AIP_MODEL_DIR"))

    total_time = time.time() - total_start
    print(f"\nDone! Total time: {total_time/60:.1f} minutes")


if __name__ == "__main__":
    main()
