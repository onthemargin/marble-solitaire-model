import torch
import torch.nn as nn
import torch.nn.functional as F
from marble_solitaire.model import SolitaireNet


def compute_loss(policy_logits, policy_targets, value_pred, value_targets):
    """Combined policy + value loss."""
    # Policy: cross-entropy (targets are probability distributions, not class indices)
    policy_loss = -(policy_targets * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
    # Value: MSE
    value_loss = F.mse_loss(value_pred, value_targets)
    return policy_loss + value_loss


def train_step(model, optimizer, states, policy_targets, value_targets):
    """Single training step. Returns loss value."""
    model.train()
    optimizer.zero_grad()
    policy_logits, value_pred = model(states)
    loss = compute_loss(policy_logits, policy_targets, value_pred, value_targets)
    loss.backward()
    optimizer.step()
    return loss.item()


def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)


def load_checkpoint(path):
    model = SolitaireNet()
    model.load_state_dict(torch.load(path, weights_only=True))
    return model


def run_training(
    n_iterations=50,
    episodes_per_iter=50,
    n_simulations=50,
    epochs_per_iter=10,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-4,
    output_dir="models",
    checkpoint_iters=None,
    resume_from=None,
    start_iter=1,
):
    """Full AlphaZero training loop."""
    import os
    import time
    from marble_solitaire.self_play import run_episode, ReplayBuffer
    from marble_solitaire.inference import solve_greedy
    from marble_solitaire.board import initial_board

    if checkpoint_iters is None:
        checkpoint_iters = {1, 5, 15, 30, 50}

    os.makedirs(output_dir, exist_ok=True)

    if resume_from:
        model = load_checkpoint(resume_from)
        print(f"Resumed from {resume_from}")
    else:
        model = SolitaireNet()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    replay_buffer = ReplayBuffer(max_size=50000)

    for iteration in range(start_iter, start_iter + n_iterations):
        iter_start = time.time()

        # Self-play
        model.eval()
        episode_lengths = []
        for ep in range(episodes_per_iter):
            examples = run_episode(model, n_simulations=n_simulations)
            replay_buffer.add(examples)
            episode_lengths.append(len(examples))

        # Training
        if len(replay_buffer) >= batch_size:
            total_loss = 0.0
            n_steps = 0
            for _ in range(epochs_per_iter):
                actual_batch = min(batch_size, len(replay_buffer))
                states, policies, outcomes = replay_buffer.sample(actual_batch)
                states_t = torch.FloatTensor(states)
                policies_t = torch.FloatTensor(policies)
                outcomes_t = torch.FloatTensor(outcomes).unsqueeze(1)
                loss = train_step(model, optimizer, states_t, policies_t, outcomes_t)
                total_loss += loss
                n_steps += 1
            avg_loss = total_loss / n_steps
        else:
            avg_loss = float("nan")

        # Evaluate: solve 10 boards with greedy policy
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
            f"Iter {iteration:3d}/{start_iter + n_iterations - 1} | "
            f"Loss: {avg_loss:.4f} | "
            f"Avg remaining: {avg_remaining:.1f} | "
            f"Solved: {solved}/10 | "
            f"Buffer: {len(replay_buffer)} | "
            f"Ep len: {sum(episode_lengths)/len(episode_lengths):.0f} | "
            f"Time: {elapsed:.0f}s"
        )

        # Save checkpoint
        if iteration in checkpoint_iters:
            path = os.path.join(output_dir, f"iter_{iteration:03d}.pt")
            save_checkpoint(model, path)
            print(f"  -> Saved checkpoint: {path}")

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train marble solitaire AlphaZero model")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--simulations", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--start-iter", type=int, default=1)
    parser.add_argument("--checkpoint-iters", type=str, default=None,
                        help="Comma-separated iteration numbers to save checkpoints")
    args = parser.parse_args()

    ckpt_iters = None
    if args.checkpoint_iters:
        ckpt_iters = set(int(x) for x in args.checkpoint_iters.split(","))

    run_training(
        n_iterations=args.iterations,
        episodes_per_iter=args.episodes,
        n_simulations=args.simulations,
        output_dir=args.output_dir,
        resume_from=args.resume,
        start_iter=args.start_iter,
        checkpoint_iters=ckpt_iters,
    )
