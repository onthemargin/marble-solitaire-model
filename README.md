# Marble Solitaire AI

An AlphaZero-style neural network that learns to solve European 37-hole marble solitaire through self-play. The web UI shows 5 generations of the model improving from random moves to skilled solving.

## How It Works

A small CNN (~75K parameters) with two heads:
- **Policy head**: predicts which move to make (196 possible moves)
- **Value head**: estimates position quality (-1 to +1)

Training uses Monte Carlo Tree Search (MCTS) guided by the network to play games against itself. Each generation trains on the outcomes of previous games, gradually learning to solve the puzzle.

## Project Structure

```
src/marble_solitaire/
  board.py        # Board state, move generation, game logic
  model.py        # Dual-headed CNN (policy + value)
  mcts.py         # Single-player MCTS with PUCT selection
  self_play.py    # Episode generation + replay buffer
  train.py        # Training loop with checkpoint saving
  inference.py    # Greedy policy solver
  export.py       # ONNX export for browser inference

tests/             # 60 tests covering all modules
web/               # Vite + TypeScript + ONNX Runtime Web UI
```

## Setup

Requires Python 3.11+ and Node.js 20+.

```bash
# Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pytest
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install onnx onnxruntime

# Run tests
pytest

# Web UI
cd web
npm install
```

## Training

Training generates 5 checkpoints at iterations 1, 5, 15, 30, and 50.

```bash
# Full training (~3-5 hours on CPU)
PYTHONPATH=src python -m marble_solitaire.train \
  --iterations 50 \
  --episodes 50 \
  --simulations 50 \
  --output-dir models

# Quick test run (5 minutes)
PYTHONPATH=src python -m marble_solitaire.train \
  --iterations 2 \
  --episodes 5 \
  --simulations 10 \
  --output-dir models
```

**Output**: `models/iter_001.pt`, `iter_005.pt`, `iter_015.pt`, `iter_030.pt`, `iter_050.pt`

Model checkpoints are ~320KB each and are not included in the repository.

## ONNX Export

After training, export checkpoints to ONNX for the web UI:

```bash
PYTHONPATH=src python -m marble_solitaire.export
```

This creates 5 ONNX files in `web/public/models/`:
- `gen1_random.onnx` (iteration 1)
- `gen2_novice.onnx` (iteration 5)
- `gen3_apprentice.onnx` (iteration 15)
- `gen4_skilled.onnx` (iteration 30)
- `gen5_expert.onnx` (iteration 50)

## Web UI

```bash
cd web
npm run dev    # Development server
npm run build  # Production build → web/dist/
```

The UI lets you:
- Select between 5 model generations
- Watch each generation attempt to solve the puzzle
- See live stats (marbles remaining, confidence, move count)
- Learn about AlphaZero, MCTS, and neural networks

## CLI Inference

```bash
PYTHONPATH=src python -m marble_solitaire.inference --model models/iter_050.pt
```

## Architecture Details

- **Board**: European 37-hole, represented as 7x7 grid with validity mask
- **Network input**: 2-channel 7x7 tensor (marbles + valid positions)
- **Network output**: 196 move logits (4 directions x 7 x 7) + scalar value
- **MCTS**: Single-player variant (no sign flip during backup)
- **Reward**: +1.0 for 1 marble remaining, smooth gradient for partial solutions
- **Training**: Cross-entropy (policy) + MSE (value) + L2 regularization
