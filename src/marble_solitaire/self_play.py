"""Self-play module for AlphaZero-style marble solitaire training."""

import numpy as np
import torch
from marble_solitaire.board import initial_board, index_to_move
from marble_solitaire.mcts import mcts_search, get_action_probabilities, compute_outcome


def sample_curriculum_start(min_marbles=5, max_marbles=12):
    """Reach a curriculum starting position by random play from the initial board.

    All resulting states are legally reachable (no cheating — we don't inject
    impossible positions). The model gets concentrated endgame practice while
    still seeing only valid trajectories.
    """
    state = initial_board()
    target = np.random.randint(min_marbles, max_marbles + 1)
    while state.count_marbles() > target:
        legal = state.get_legal_moves()
        if not legal:
            break
        move_idx = np.random.randint(len(legal))
        state = state.apply_move(legal[move_idx])
    return state


def run_episode(network, n_simulations=50, temp_threshold=15,
                endgame_temp=0.05, endgame_threshold=25,
                curriculum_endgame=False,
                curriculum_min_marbles=5, curriculum_max_marbles=12,
                dirichlet_epsilon=0.25):
    """Run one self-play episode. Returns list of (state_tensor, policy, outcome).

    If curriculum_endgame is True, starts from a random mid/late-game position
    (reachable by random play) instead of the initial board. This gives the
    model concentrated endgame practice without injecting solutions.
    """
    if curriculum_endgame:
        state = sample_curriculum_start(curriculum_min_marbles, curriculum_max_marbles)
    else:
        state = initial_board()
    trajectory = []
    move_count = 0

    while not state.is_terminal():
        legal_moves = state.get_legal_moves()
        if not legal_moves:
            break

        root = mcts_search(state, network, n_simulations,
                           dirichlet_epsilon=dirichlet_epsilon)

        # Curriculum episodes are short — keep temperature low throughout for
        # high-quality endgame play. Full episodes use the normal schedule.
        if curriculum_endgame:
            temperature = endgame_temp
        elif move_count < temp_threshold:
            temperature = 1.0
        elif move_count < endgame_threshold:
            temperature = 0.1
        else:
            temperature = endgame_temp
        policy = get_action_probabilities(root, temperature)

        trajectory.append((state.to_tensor(), policy, None))

        # Sample move from policy
        move_idx = np.random.choice(196, p=policy)
        move = index_to_move(move_idx)
        state = state.apply_move(move)
        move_count += 1

    outcome = compute_outcome(state.count_marbles())
    examples = [(s, p, outcome) for s, p, _ in trajectory]
    return examples


class ReplayBuffer:
    """Fixed-size buffer storing (state, policy, outcome) training examples."""

    def __init__(self, max_size=50000):
        self.max_size = max_size
        self.buffer = []

    def add(self, examples):
        """Add a list of examples, dropping oldest if over capacity."""
        self.buffer.extend(examples)
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]

    def sample(self, batch_size):
        """Sample a random batch. Returns (states, policies, outcomes) as numpy arrays."""
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states = np.array([b[0] for b in batch])
        policies = np.array([b[1] for b in batch])
        outcomes = np.array([b[2] for b in batch], dtype=np.float32)
        return states, policies, outcomes

    def __len__(self):
        return len(self.buffer)
