# Marble Solitaire AI

An AlphaZero-style neural network that learns European 37-hole marble solitaire through self-play. The web UI shows five snapshots of the same network at successive points in training, from random play to the proven mathematical optimum.

**Live demo**: https://app.gyatso.me/marble-solitaire/

## How it works

A small CNN with two heads:
- **Policy head** — 196 logits, one per (`row`, `col`, `direction`) move
- **Value head** — scalar in `[-1, +1]` (tanh) estimating position quality

Training is a single-player AlphaZero loop: MCTS guided by the network generates self-play games; the recorded games train the network; the better network generates better games; repeat. No opponent, no sign flip during backup, value head learns "how close to a perfect solve."

## The "proven optimum" wrinkle

The 37-hole European board with the **centre hole as the starting empty position** has *no* single-marble solution anywhere on the board. Durango Bill's exhaustive enumeration of all 16,662,591,542,028,595,169,821,912 possible game sequences from this start found zero that end at 1 marble — see https://www.durangobill.com/Peg37.html. Jaap Scherphuis gives the parity proof: every move preserves the equal-parity property created by the centre-empty start, but a single-marble endgame requires unequal parities.

So the achievable optimum from this start is **2 marbles remaining**. The trained "Master" generation hits this consistently — it's playing optimally, not stuck.

This is encoded in the reward function (`src/marble_solitaire/mcts.py:compute_outcome`):

| Marbles remaining | Reward | Notes |
|---:|---:|---|
| 1 | +1.0 | Unreachable from a centre-start, but kept so the function is start-config-agnostic |
| 2 | -0.3 | Achievable optimum |
| n ≥ 3 | -1 + 2/n | Smooth gradient |

## Project layout

```
src/marble_solitaire/
  board.py        Board state, move generation, game rules
  model.py        Dual-headed CNN (configurable channels + blocks)
  mcts.py         Single-player MCTS + Dirichlet noise + reward
  self_play.py    Episode generation + replay buffer
  train.py        Training loop with checkpoint saving
  inference.py    Greedy policy solver
  export.py       ONNX export for browser inference
  seeds.py        High-MCTS seed search (mostly historical at this point)
tests/            Unit tests (pytest)
vertex-training/  Cloud-training container + job specs
web/              Vite + TypeScript + ONNX Runtime browser UI
docs/history/     Original v1 design notes (kept for context, not current)
```

## Generations

The web UI loads five ONNX snapshots from one training run. Marble counts below are deterministic greedy-eval from the deployed checkpoints:

| Gen | Label | Iter | Greedy marbles left |
|---:|---|---:|---:|
| 1 | Clueless | 1 | 12 |
| 2 | Beginner | 15 | 8 |
| 3 | Intermediate | 50 | 8 |
| 4 | Advanced | 150 | 5 |
| 5 | Master | 250 | 2 *(proven optimum)* |

The Beginner→Intermediate plateau at 8 is real: both checkpoints play the same opening and get stuck in the same shape.

## Running locally

### Python (training, inference, tests)

```bash
pip install -e ".[dev]"
pytest                                     # unit tests
python -m marble_solitaire.train --help    # local training (CPU is slow; cloud training below)
```

### Web UI

```bash
cd web
npm install
npm run dev
```

This needs the five ONNX models at `web/public/models/gen{1..5}_*.onnx`. Two ways to get them:

1. **Train + export your own** (see below). Each run produces snapshot `.pt` files plus ONNX files in the output dir; copy the five ONNX files into `web/public/models/`.
2. **Download a release artifact** — if you don't want to spend GPU time, check the GitHub Releases for a `models.zip` containing the same five `.onnx` files used by the live demo.

`npm run build` writes a production bundle to `web/dist/`.

## Cloud training (optional)

Training runs on Vertex AI Custom Jobs against a single T4 GPU. The dev workflow is intentionally minimal — there are no Terraform modules or magic scripts; you build a container image and submit a job from the Vertex Console.

Required setup (do this once, in your own GCP project):
- A GCP project with billing, Vertex AI enabled, and T4 GPU quota in your chosen region
- A GCS bucket for build sources and a separate path for training artifacts
- A Cloud Build service account with permission to push to Artifact Registry / GCR (or use the default Cloud Build SA)

Build the training container:

```bash
PROJECT_ID=your-project ./vertex-training/build.sh
```

This pushes `gcr.io/your-project/marble-solitaire-training:<timestamp>`. The script also prints the image URI.

Submit a job — replace placeholders in any of the `*-spec.yaml` files:

| File | Notes |
|---|---|
| `phase2-spec.yaml` | The "canonical" run: bootstraps from a previous checkpoint, sharper reward, one-time seed search. Produces the deployed Master. |
| `phase3-spec.yaml` | Pure-discovery experiment (curriculum + higher exploration noise). Regressed in practice — kept as a cautionary record. |
| `phase4-spec.yaml` | Re-run under the corrected reward to confirm the 2-marble plateau is the optimum, not a training failure. |

Substitute `YOUR_PROJECT`, `YOUR_BUCKET`, and the image tag, then submit via either:

```bash
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=marble-solitaire-run \
  --config=vertex-training/phase2-spec.yaml \
  --project=your-project
```

…or the Vertex AI Console (steps in `vertex-training/submit-console-steps.md`). Outputs land in the GCS path you set as `AIP_MODEL_DIR`.

To deploy the resulting models locally:

```bash
gsutil -m cp 'gs://your-bucket/marble-solitaire/<run>/onnx/*.onnx' web/public/models/
cd web && npm run build
```

## Architecture details

- **Board** — European 37-hole, represented as a 7×7 grid with a validity mask
- **Move encoding** — `(row * 7 + col) * 4 + direction`, with `UP=0 DOWN=1 LEFT=2 RIGHT=3`. The encoding must agree between Python (`board.py`) and TypeScript (`web/src/solver.ts`) — a mismatch silently breaks the browser solver
- **Network input** — 2-channel `7×7` tensor: `[marbles, valid_mask]`
- **Network output** — 196 move logits + 1 scalar value
- **MCTS** — single-player variant (no opponent, no sign flip during backup), PUCT selection, Dirichlet noise on root priors for exploration
- **Training loss** — cross-entropy (policy) + MSE (value) + L2 regularisation

## Contributing

Issues and pull requests welcome. The `tests/` directory uses pytest; please add a failing test first when fixing a bug.

## License

MIT — see [LICENSE](./LICENSE).
