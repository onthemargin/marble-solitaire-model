import torch
import numpy as np
from marble_solitaire.board import BoardState, create_legal_move_mask, index_to_move


def solve_greedy(network, board):
    """Solve using greedy policy (highest-probability valid move each step)."""
    moves = []
    current = board
    network.eval()

    while not current.is_terminal():
        legal = current.get_legal_moves()
        if not legal:
            break

        state_tensor = torch.FloatTensor(current.to_tensor()).unsqueeze(0)
        with torch.no_grad():
            policy_logits, _ = network(state_tensor)

        logits = policy_logits.squeeze(0).numpy()
        mask = create_legal_move_mask(legal)

        # Mask illegal moves to -inf
        masked_logits = np.where(mask > 0, logits, -1e9)
        best_idx = int(np.argmax(masked_logits))
        move = index_to_move(best_idx)

        moves.append(move)
        current = current.apply_move(move)

    return moves
