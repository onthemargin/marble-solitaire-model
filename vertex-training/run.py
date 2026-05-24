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


def upload_to_gcs(local_path, local_root, gcs_dir):
    """Upload local_path to GCS, preserving the path relative to local_root."""
    if not gcs_dir or not gcs_dir.startswith("gs://"):
        return
    try:
        from google.cloud import storage
        parts = gcs_dir.replace("gs://", "").split("/", 1)
        bucket_name = parts[0]
        prefix = parts[1].rstrip("/") if len(parts) > 1 else ""
        rel_path = os.path.relpath(local_path, local_root)
        blob_name = f"{prefix}/{rel_path}" if prefix else rel_path

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_path)
        print(f"  -> Uploaded to gs://{bucket_name}/{blob_name}")
    except Exception as e:
        print(f"  -> GCS upload failed: {e}")


def run_export_mapping(args):
    """Export a custom label→.pt mapping. Each label becomes <label>.onnx."""
    from google.cloud import storage

    gcs_output_dir = os.environ.get("AIP_MODEL_DIR", "")
    output_dir = "/tmp/output" if gcs_output_dir.startswith("gs://") else (gcs_output_dir or "/tmp/output")
    onnx_dir = os.path.join(output_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    client = storage.Client()
    print(f"Network: {args.channels}ch × {args.n_blocks} blocks")
    print("Custom export mapping:")

    for pair in args.export_mapping.split(","):
        label, gcs_path = pair.split("=", 1)
        label = label.strip()
        gcs_path = gcs_path.strip()
        parts = gcs_path.replace("gs://", "").split("/", 1)
        bucket = client.bucket(parts[0])
        blob = bucket.blob(parts[1])
        if not blob.exists():
            print(f"  Skipping {label}: {gcs_path} not found")
            continue
        local_pt = os.path.join(output_dir, f"{label}.pt")
        blob.download_to_filename(local_pt)

        model = SolitaireNet(channels=args.channels, n_blocks=args.n_blocks)
        state = torch.load(local_pt, weights_only=True, map_location="cpu")
        model.load_state_dict(state)
        onnx_path = os.path.join(onnx_dir, f"{label}.onnx")
        export_to_onnx(model, onnx_path)
        size_kb = os.path.getsize(onnx_path) / 1024
        print(f"  {label}: {gcs_path} → {onnx_path} ({size_kb:.0f} KB)")
        upload_to_gcs(onnx_path, output_dir, gcs_output_dir)

    print("\nDone.")


def run_export_only(args):
    """Download .pt checkpoints from GCS, export each GEN_LABELS iter to ONNX, upload."""
    from google.cloud import storage

    gcs_output_dir = os.environ.get("AIP_MODEL_DIR", "")
    output_dir = "/tmp/output" if gcs_output_dir.startswith("gs://") else (gcs_output_dir or "/tmp/output")
    onnx_dir = os.path.join(output_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)

    src_prefix = args.export_only_from.rstrip("/") + "/"
    parts = src_prefix.replace("gs://", "").split("/", 1)
    bucket = storage.Client().bucket(parts[0])
    src_inner = parts[1] if len(parts) > 1 else ""

    print(f"Exporting checkpoints from {src_prefix} → ONNX")
    print(f"Network: {args.channels}ch × {args.n_blocks} blocks")

    for iteration, label in GEN_LABELS.items():
        pt_name = f"iter_{iteration:03d}.pt"
        blob = bucket.blob(f"{src_inner}{pt_name}")
        if not blob.exists():
            print(f"  Skipping {label}: gs://{parts[0]}/{src_inner}{pt_name} not found")
            continue
        local_pt = os.path.join(output_dir, pt_name)
        blob.download_to_filename(local_pt)

        model = SolitaireNet(channels=args.channels, n_blocks=args.n_blocks)
        state = torch.load(local_pt, weights_only=True, map_location="cpu")
        model.load_state_dict(state)
        onnx_path = os.path.join(onnx_dir, f"{label}.onnx")
        export_to_onnx(model, onnx_path)
        size_kb = os.path.getsize(onnx_path) / 1024
        print(f"  Exported {label}: {onnx_path} ({size_kb:.0f} KB)")
        upload_to_gcs(onnx_path, output_dir, gcs_output_dir)

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--simulations", type=int, default=400)
    parser.add_argument("--epochs-per-iter", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint-iters", type=str, default="1,15,50,150,250,350,500")
    parser.add_argument("--channels", type=int, default=64, help="Conv channels in network trunk")
    parser.add_argument("--n-blocks", type=int, default=3, help="Number of conv blocks")
    parser.add_argument("--seed-attempts", type=int, default=0,
                        help="If >0, try to find seed solutions before training (uses high-MCTS).")
    parser.add_argument("--seed-sims", type=int, default=3000,
                        help="MCTS sims when searching for seeds.")
    parser.add_argument("--seed-copies", type=int, default=10,
                        help="Replicate each seed N times in buffer to weight it.")
    parser.add_argument("--bootstrap-checkpoint", type=str, default="",
                        help="GCS path to a .pt to load as bootstrap network for seed search.")
    parser.add_argument("--curriculum-fraction", type=float, default=0.0,
                        help="Fraction of episodes that start from a mid/late-game position (0.0-1.0).")
    parser.add_argument("--curriculum-min-marbles", type=int, default=5)
    parser.add_argument("--curriculum-max-marbles", type=int, default=12)
    parser.add_argument("--dirichlet-epsilon", type=float, default=0.25,
                        help="Mixing weight for Dirichlet noise on MCTS root priors.")
    parser.add_argument("--export-only-from", type=str, default="",
                        help="If set, download .pt files from this GCS prefix, "
                             "export each GEN_LABELS iter to ONNX, upload, and exit.")
    parser.add_argument("--export-mapping", type=str, default="",
                        help="Comma-separated label=gcs_path pairs. Each .pt is "
                             "downloaded, exported to ONNX as <label>.onnx, uploaded.")
    args = parser.parse_args()

    if args.export_only_from:
        run_export_only(args)
        return
    if args.export_mapping:
        run_export_mapping(args)
        return

    checkpoint_iters = set(int(x) for x in args.checkpoint_iters.split(","))

    # Vertex AI sets AIP_MODEL_DIR to a gs:// URI. Write locally first, upload after.
    gcs_output_dir = os.environ.get("AIP_MODEL_DIR", "")
    output_dir = "/tmp/output" if gcs_output_dir.startswith("gs://") else (gcs_output_dir or "/tmp/output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Local output dir: {output_dir}")
    if gcs_output_dir.startswith("gs://"):
        print(f"GCS upload dir:   {gcs_output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = SolitaireNet(channels=args.channels, n_blocks=args.n_blocks).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Network: {args.channels}ch × {args.n_blocks} blocks → {n_params/1e3:.0f}K params")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    replay_buffer = ReplayBuffer(max_size=100000)

    total_start = time.time()

    # Optional: bootstrap network from a previously-trained checkpoint.
    # Only works if shape matches (channels/n_blocks must match the checkpoint).
    if args.bootstrap_checkpoint:
        local_bs = args.bootstrap_checkpoint
        if local_bs.startswith("gs://"):
            from google.cloud import storage
            parts = local_bs.replace("gs://", "").split("/", 1)
            bucket = storage.Client().bucket(parts[0])
            local_bs = "/tmp/bootstrap.pt"
            bucket.blob(parts[1]).download_to_filename(local_bs)
            print(f"Downloaded bootstrap checkpoint to {local_bs}")
        state = torch.load(local_bs, weights_only=True, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded bootstrap weights from {args.bootstrap_checkpoint}")

    # Optional: seed the replay buffer with high-MCTS solutions
    if args.seed_attempts > 0:
        from marble_solitaire.seeds import find_solution, moves_to_examples
        print(f"\n=== Searching for seed solutions ({args.seed_attempts} attempts, "
              f"{args.seed_sims} MCTS sims each) ===")
        model.eval()
        seed_search_start = time.time()
        moves = find_solution(
            model,
            n_simulations=args.seed_sims,
            max_attempts=args.seed_attempts,
            target_marbles=2,
            prefer_center=True,
        )
        if moves is not None:
            examples = moves_to_examples(moves)
            # Replicate to weight the seed strongly in the buffer
            seeded_examples = examples * args.seed_copies
            replay_buffer.add(seeded_examples)
            print(f"Added {len(seeded_examples)} seeded examples to replay buffer "
                  f"({args.seed_copies} copies of {len(examples)}-move trajectory)")
            print(f"Seed search took {(time.time()-seed_search_start)/60:.1f} min")
        else:
            print("No seed solutions found — proceeding with cold-start training")

    for iteration in range(1, args.iterations + 1):
        iter_start = time.time()

        # Self-play (CPU — MCTS is sequential, GPU doesn't help much here
        # but the forward passes inside MCTS use GPU)
        model.eval()
        episode_lengths = []
        n_curriculum = 0
        for ep in range(args.episodes):
            is_curriculum = (args.curriculum_fraction > 0 and
                             np.random.random() < args.curriculum_fraction)
            examples = run_episode(
                model,
                n_simulations=args.simulations,
                curriculum_endgame=is_curriculum,
                curriculum_min_marbles=args.curriculum_min_marbles,
                curriculum_max_marbles=args.curriculum_max_marbles,
                dirichlet_epsilon=args.dirichlet_epsilon,
            )
            replay_buffer.add(examples)
            episode_lengths.append(len(examples))
            if is_curriculum:
                n_curriculum += 1

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
            f"Curr: {n_curriculum}/{args.episodes} | "
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

            upload_to_gcs(local_path, output_dir, gcs_output_dir)

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
        checkpoint_model = SolitaireNet(channels=args.channels, n_blocks=args.n_blocks)
        checkpoint_model.load_state_dict(torch.load(pt_path, weights_only=True))
        export_to_onnx(checkpoint_model, onnx_path)
        size_kb = os.path.getsize(onnx_path) / 1024
        print(f"  Exported {label}: {onnx_path} ({size_kb:.0f} KB)")
        upload_to_gcs(onnx_path, output_dir, gcs_output_dir)

    total_time = time.time() - total_start
    print(f"\nDone! Total time: {total_time/60:.1f} minutes")


if __name__ == "__main__":
    main()
