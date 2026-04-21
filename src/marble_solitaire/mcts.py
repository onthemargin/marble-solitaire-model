import math
import numpy as np
import torch
from marble_solitaire.board import BoardState, create_legal_move_mask, move_to_index


def compute_outcome(remaining_marbles: int) -> float:
    if remaining_marbles == 1:
        return 1.0
    return -1.0 + 2.0 / remaining_marbles


class MCTSNode:
    def __init__(self, state: BoardState, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.N = 0
        self.W = 0.0
        self.P = 0.0

    @property
    def Q(self) -> float:
        return self.W / self.N if self.N > 0 else 0.0


def mcts_search(root_state, network, n_simulations, c_puct=1.5,
                dirichlet_alpha=0.3, dirichlet_epsilon=0.25):
    root = MCTSNode(state=root_state)
    if root_state.is_terminal():
        return root
    _expand(root, network)

    # Add Dirichlet noise to root priors for exploration
    if dirichlet_epsilon > 0 and root.children:
        noise = np.random.dirichlet([dirichlet_alpha] * len(root.children))
        for i, child in enumerate(root.children.values()):
            child.P = (1 - dirichlet_epsilon) * child.P + dirichlet_epsilon * noise[i]

    for _ in range(n_simulations):
        node = root
        path = [node]

        while node.children and not node.state.is_terminal():
            best_score = -float('inf')
            best_move = None
            for move, child in node.children.items():
                puct = child.Q + c_puct * child.P * math.sqrt(node.N) / (1 + child.N)
                if puct > best_score:
                    best_score = puct
                    best_move = move
            node = node.children[best_move]
            path.append(node)

        if node.state.is_terminal():
            value = compute_outcome(node.state.count_marbles())
        elif not node.children:
            legal = node.state.get_legal_moves()
            if not legal:
                value = compute_outcome(node.state.count_marbles())
            else:
                value = _expand(node, network)
        else:
            value = 0.0

        for ancestor in reversed(path):
            ancestor.N += 1
            ancestor.W += value

    return root


def _expand(node, network):
    legal_moves = node.state.get_legal_moves()
    if not legal_moves:
        return compute_outcome(node.state.count_marbles())

    state_tensor = torch.FloatTensor(node.state.to_tensor()).unsqueeze(0)
    # Move to same device as model
    device = next(network.parameters()).device
    state_tensor = state_tensor.to(device)
    with torch.no_grad():
        policy_logits, value = network(state_tensor)

    policy_logits = policy_logits.squeeze(0).cpu().numpy()
    value = value.item()

    mask = create_legal_move_mask(legal_moves)
    masked = np.where(mask > 0, policy_logits, -1e9)
    exp_l = np.exp(masked - masked.max())
    exp_l = exp_l * mask
    priors = exp_l / exp_l.sum()

    for move in legal_moves:
        child_state = node.state.apply_move(move)
        child = MCTSNode(state=child_state, parent=node)
        child.P = priors[move_to_index(*move)]
        node.children[move] = child

    return value


def get_action_probabilities(root, temperature=1.0):
    probs = np.zeros(196, dtype=np.float32)
    if not root.children:
        return probs
    if temperature < 0.01:
        best = max(root.children.keys(), key=lambda m: root.children[m].N)
        probs[move_to_index(*best)] = 1.0
    else:
        log_counts = np.full(196, -np.inf, dtype=np.float64)
        for move, child in root.children.items():
            if child.N > 0:
                log_counts[move_to_index(*move)] = np.log(child.N) / temperature
        log_counts -= log_counts.max()
        exp_counts = np.exp(log_counts)
        exp_counts[log_counts == -np.inf] = 0.0
        total = exp_counts.sum()
        if total > 0:
            probs = (exp_counts / total).astype(np.float32)
        else:
            # fallback: uniform over children
            for move in root.children:
                probs[move_to_index(*move)] = 1.0 / len(root.children)
    return probs
