# Marble Solitaire AI

An AlphaZero-style neural network that learns to solve European 37-hole marble solitaire through self-play. The web UI at https://app.gyatso.me/marble-solitaire/ shows 5 generations of the model improving from random moves toward skilled play.

## How It Works

A small CNN (75K–450K parameters depending on phase) with two heads:
- **Policy head**: predicts which move to make (196 possible moves)
- **Value head**: estimates position quality (-1 to +1)

Training uses Monte Carlo Tree Search (MCTS) guided by the network to play games against itself. Each iteration trains on the outcomes of previous games. Self-play, MCTS, and gradient updates compound to push the network toward strong play.

## Project Structure

```
src/marble_solitaire/
  board.py        # Board state, move generation, game logic
  model.py        # Dual-headed CNN (configurable channels + blocks)
  mcts.py         # Single-player MCTS with PUCT selection + Dirichlet noise
  self_play.py    # Episode generation + replay buffer
  train.py        # Training loop with checkpoint saving
  inference.py    # Greedy policy solver (device-aware)
  export.py       # ONNX export for browser inference
  seeds.py        # Generate seed solutions via high-MCTS search

tests/             # Unit tests
vertex-training/   # Cloud training pipeline (Docker + Vertex AI custom job)
web/               # Vite + TypeScript + ONNX Runtime Web UI
```

## Training

**Training runs in the cloud on Vertex AI** with NVIDIA T4 GPUs. The dev VM is locked down (no torch installed locally) so all training is cloud-only.

### Pipeline

1. Edit code locally
2. Build container: `./vertex-training/build.sh` — produces a timestamped image in GCR
3. Edit `vertex-training/phase2-spec.yaml` (or write a new spec) with the image URI
4. Submit job: `gcloud ai custom-jobs create --region=us-central1 --display-name=<name> --config=<spec.yaml> --project=ai-dev-463705`
5. Monitor in [Vertex AI console](https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=ai-dev-463705)
6. Outputs land in `gs://ai-dev-463705-ml-artifacts/marble-solitaire/<run>/`

### Training history

| Run | Network | Iters | Episodes | MCTS sims | Time | Best result |
|-----|---------|-------|----------|-----------|------|------------|
| Phase 1 | 64ch × 3 blocks (75K) | 500 | 25 | 100 | 32h ($20) | 5 marbles avg, 0/10 solves |
| Phase 2 (planned) | 64ch × 3 blocks (75K) | 750 | 50 | 200 | ~24h (~$15) | Goal: 1 marble center |

**Why Phase 1 didn't solve**: Model never saw a winning trajectory in self-play (~10⁹ game paths) so no `+1.0` reward signal to learn from. Plateaued at ~5 marbles.

**Phase 2 changes**:
- Bootstrap from Phase 1's gen5_master (preserves what was already learned)
- **Seed search** at startup: runs 30 episodes of high-MCTS (3000 sims/move) to find ≤2-marble solutions, seeds replay buffer with those (10× weight)
- More episodes per iter (25 → 50) for more endgame diversity
- More MCTS sims per move (100 → 200) for stronger teacher signal
- Sharper reward (2 marbles → -0.3, was 0.0) to widen the "solved vs not" gap
- Endgame temperature drops to 0.05 after move 25 (forces best-move selection)

## Generations

After training, 5 checkpoints become the 5 generations shown in the UI:

| Generation | Iteration | Label |
|------------|-----------|-------|
| Gen 1 | 1 | Clueless |
| Gen 2 | 15 | Beginner |
| Gen 3 | 50 | Intermediate |
| Gen 4 | 150 | Advanced |
| Gen 5 | 250 | Master |

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
- Learn about AlphaZero, MCTS, and neural networks (inline explanations)

`web/dist/` is checked into git (needed for Docker deploy). Rebuild after model updates.

## Deployment

Marble Solitaire deploys to https://app.gyatso.me/marble-solitaire/ via the parent monorepo's `/go` command. The Cloud Build pipeline copies `web/dist/` into the nginx container.

After a new model trains:
```bash
# Download ONNX models from GCS to web/public/models/
gsutil -m cp gs://ai-dev-463705-ml-artifacts/marble-solitaire/<run>/onnx/*.onnx web/public/models/
cd web && npx vite build       # rebuilds dist with new models
# commit web/dist/ in monorepo and run /go
```

## GCP Infrastructure

| Resource | Purpose |
|----------|---------|
| GCS bucket `ai-dev-463705-ml-artifacts` | Training checkpoints + ONNX exports (uniform access, public access prevented, audit logs enabled, 90-day lifecycle) |
| GCR image `marble-solitaire-training` | Training container (PyTorch 2.6, CUDA 12.4, non-root, pinned deps) |
| Vertex AI Custom Jobs | T4 GPU training |
| Cloud Build | Container builds via `cloudbuild-deploy@` SA |
| Compute SA `881150409277-compute@` | Scoped permissions: Cloud Build Editor, Service Usage Consumer, Service Account User, Logging Viewer, Cloud Run Developer, Storage Object Admin (on artifacts + cloudbuild buckets only) |

T4 GPU quota in us-central1 is set to 1.

## CLI Inference (CPU)

If you set up a local Python env with torch:
```bash
PYTHONPATH=src python -m marble_solitaire.inference --model models/iter_250.pt
```

## Architecture Details

- **Board**: European 37-hole, represented as 7x7 grid with validity mask
- **Move encoding**: `(row * 7 + col) * 4 + direction` (UP=0, DOWN=1, LEFT=2, RIGHT=3). Must match between Python (`board.py`) and TypeScript (`solver.ts`).
- **Network input**: 2-channel 7x7 tensor (marbles + valid positions)
- **Network output**: 196 move logits + scalar value (tanh)
- **MCTS**: Single-player variant (no sign flip during backup), PUCT selection, Dirichlet noise on root priors
- **Reward**:
  - +1.0: 1 marble at center
  - +0.6: 1 marble off-center
  - -0.3: 2 marbles remaining
  - Smooth gradient `-1 + 2/n` for n ≥ 3 marbles
- **Training loss**: Cross-entropy (policy) + MSE (value) + L2 regularization
