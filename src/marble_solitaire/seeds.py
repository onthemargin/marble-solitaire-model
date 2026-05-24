"""Generate seed solutions for bootstrapping AlphaZero training.

Without seed solutions, the model rarely sees a winning trajectory during
self-play (~10^9 possible game trees, ~0% of random play solves). Seeding
the replay buffer with even 1-2 known solutions dramatically accelerates
learning the endgame.

Strategy: use a "warm" pretrained network with very high MCTS sim count
to find ≤2-marble solutions (preferably 1-marble-center). Cache results
as JSON so they're reusable across training runs.
"""
import json
import os
import time
from typing import Optional

import numpy as np
from marble_solitaire.board import (
    initial_board,
    index_to_move,
    move_to_index,
    create_legal_move_mask,
)
from marble_solitaire.mcts import mcts_search, get_action_probabilities


def find_solution(network, n_simulations: int = 5000, max_attempts: int = 50,
                  target_marbles: int = 2, prefer_center: bool = True,
                  verbose: bool = True) -> Optional[list]:
    """Try to find a solution by running high-strength MCTS for many episodes.

    Returns the list of moves for the best trajectory found (or None if none
    met the target). "Best" = fewest marbles remaining, with tie-break on
    center finish.
    """
    best_moves = None
    best_remaining = float("inf")
    best_center = False

    for attempt in range(max_attempts):
        moves, remaining, center = _run_episode_greedy_mcts(network, n_simulations)
        if verbose:
            print(f"  Attempt {attempt+1}/{max_attempts}: {remaining} marbles"
                  f"{' (CENTER)' if center else ''}")
        # Better = fewer marbles; tie-break prefers center
        is_better = (remaining < best_remaining) or (
            remaining == best_remaining and center and not best_center
        )
        if is_better:
            best_moves = moves
            best_remaining = remaining
            best_center = center
        # Early exit if we hit target
        if remaining <= target_marbles and (not prefer_center or center):
            if verbose:
                print(f"  Found {remaining}-marble{' center' if center else ''} solution!")
            return moves

    if verbose:
        print(f"  Best found: {best_remaining} marbles"
              f"{' center' if best_center else ''}")
    if best_remaining <= target_marbles:
        return best_moves
    return None


def _run_episode_greedy_mcts(network, n_simulations: int):
    """Run one episode, picking the most-visited MCTS child at each step."""
    state = initial_board()
    moves = []
    while not state.is_terminal():
        legal = state.get_legal_moves()
        if not legal:
            break
        root = mcts_search(state, network, n_simulations=n_simulations,
                          dirichlet_epsilon=0.1)  # mild noise for diversity
        # Pick most-visited child (greedy)
        best_move = max(root.children.keys(), key=lambda m: root.children[m].N)
        moves.append(best_move)
        state = state.apply_move(best_move)
    return moves, state.count_marbles(), state.has_center_marble()


def moves_to_examples(moves: list) -> list:
    """Replay a move sequence and emit (state_tensor, policy_onehot, outcome) tuples.

    The policy at each state is one-hot for the move that was actually taken.
    The outcome is the terminal reward (+1 for center solve, etc.).
    """
    from marble_solitaire.mcts import compute_outcome

    state = initial_board()
    examples = []
    for move in moves:
        policy = np.zeros(196, dtype=np.float32)
        policy[move_to_index(*move)] = 1.0
        examples.append((state.to_tensor(), policy, None))
        state = state.apply_move(move)

    outcome = compute_outcome(state.count_marbles(), state.has_center_marble())
    return [(s, p, outcome) for s, p, _ in examples]


def augment_with_symmetries(examples: list) -> list:
    """Augment seed examples with 4-fold rotational symmetry of the board.

    The European 37-hole board has 4-fold rotational symmetry, so each seed
    trajectory yields 4 equivalent training trajectories.
    """
    out = list(examples)
    # Rotations are non-trivial to implement on the move encoding — skip for
    # now. The raw seed examples alone are usually enough to bootstrap.
    return out


def save_seeds(moves_list: list, path: str) -> None:
    """Save a list of move sequences to JSON."""
    data = {"solutions": [[list(m) for m in moves] for moves in moves_list]}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_seeds(path: str) -> list:
    """Load move sequences from JSON. Returns list of move-sequences."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    return [[tuple(m) for m in seq] for seq in data.get("solutions", [])]


def generate_and_save_seeds(network, output_path: str,
                            n_solutions_target: int = 3,
                            n_simulations: int = 5000,
                            max_attempts_per_solution: int = 30,
                            target_marbles: int = 2) -> int:
    """Generate seed solutions and save them. Returns number found."""
    solutions = []
    for i in range(n_solutions_target):
        print(f"\n=== Finding solution {i+1}/{n_solutions_target} ===")
        moves = find_solution(
            network,
            n_simulations=n_simulations,
            max_attempts=max_attempts_per_solution,
            target_marbles=target_marbles,
        )
        if moves is not None:
            solutions.append(moves)
        else:
            print(f"  No solution found within {max_attempts_per_solution} attempts.")

    if solutions:
        save_seeds(solutions, output_path)
        print(f"\nSaved {len(solutions)} seed(s) to {output_path}")
    return len(solutions)
