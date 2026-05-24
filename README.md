# Marble Solitaire AI

An AlphaZero-style neural network that learns to solve European 37-hole marble solitaire through self-play. The web UI shows 5 generations of the model improving from random play to near-perfect endgame.

## How It Works

A small CNN with two heads:
- **Policy head**: predicts which move to make (196 possible moves)
- **Value head**: estimates position quality (-1 to +1, tanh)

Training uses Monte Carlo Tree Search (MCTS) guided by the network to play games against itself. Each iteration trains on the outcomes of previous games. Self-play, MCTS, and gradient updates compound to push the network toward stronger play.

## Project Structure

```
src/marble_solitaire/
  board.py        # Board state, move generation, game logic
  model.py        # Dual-headed CNN (configurable channels + blocks)
  mcts.py         # Single-player MCTS with PUCT selection + Dirichlet noise
  self_play.py    # Episode generation + replay buffer (+ curriculum starts)
  train.py        # Training loop with checkpoint saving
  inference.py    # Greedy policy solver (device-aware)
  export.py       # ONNX export for browser inference
  seeds.py        # High-MCTS seed search

tests/             # Unit tests
vertex-training/   # Cloud training pipeline (Docker + training entrypoint)
web/               # Vite + TypeScript + ONNX Runtime Web UI
```

## Training approach

Training runs in the cloud on a single GPU. Three phases, each building on the last:

| Phase | What it tried | Outcome |
|-------|---------------|---------|
| Phase 1 | Vanilla AlphaZero self-play from scratch | Plateaued at ~5 marbles remaining |
| Phase 2 | Bootstrapped from Phase 1 + larger MCTS + sharper reward + one-time seed search | **2 marbles remaining (best)** |
| Phase 3 | Pure-discovery improvements (curriculum-start episodes + higher exploration noise) | Regressed — over-explored, unlearned Phase 2 gains |

**Why gen5 doesn't fully solve**: marble solitaire has ~10⁹ game paths. AlphaZero learns from terminal rewards — if the model never finishes a game during self-play, it never sees the `+1.0` signal and has nothing to learn from for the final move. Pure self-play within a capped budget consistently reaches 2 marbles but stalls there. We chose not to inject brute-force-solver solutions, because that would defeat the "watch the AI learn" story.

## Generations

The 5 ONNX models served by the UI are mapped across phases to show the learning arc:

| Gen | Label | Source | Expected avg marbles |
|-----|-------|--------|---------------------|
| 1 | Clueless | Phase 1 early checkpoint | ~25 (random) |
| 2 | Beginner | Phase 1 mid checkpoint | ~12–18 |
| 3 | Intermediate | Phase 1 later checkpoint | ~6–10 |
| 4 | Advanced | Phase 1 final checkpoint | ~4–6 |
| 5 | Master | Phase 2 final checkpoint | ~2 |

## Web UI

```bash
cd web
npm install
npm run dev    # Development server
npm run build  # Production build → web/dist/
```

The UI lets users:
- Select between 5 model generations
- Watch each play (Solve button) or play themselves (Play button)
- See live stats (marbles remaining, confidence, move count)
- Read inline explanations of AlphaZero, MCTS, neural networks, and why Gen 5 stalls at 2 marbles

## Architecture Details

- **Board**: European 37-hole, represented as 7×7 grid with validity mask
- **Move encoding**: `(row * 7 + col) * 4 + direction` (UP=0, DOWN=1, LEFT=2, RIGHT=3). Must match between Python (`board.py`) and TypeScript (`solver.ts`).
- **Network input**: 2-channel 7×7 tensor (marbles + valid positions)
- **Network output**: 196 move logits + scalar value (tanh)
- **MCTS**: Single-player variant (no sign flip during backup), PUCT selection, Dirichlet noise on root priors
- **Reward** (see `mcts.py:compute_outcome`):
  - +1.0: 1 marble at center
  - +0.6: 1 marble off-center
  - -0.3: 2 marbles remaining
  - Smooth gradient `-1 + 2/n` for n ≥ 3 marbles
- **Training loss**: Cross-entropy (policy) + MSE (value) + L2 regularization
