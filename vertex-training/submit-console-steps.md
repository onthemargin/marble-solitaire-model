# Submit Training Job from GCP Console

After running `./vertex-training/build.sh` from the dev VM, submit the job via Console.

## Step 1: Go to Vertex AI Training
https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=ai-dev-463705

## Step 2: Create Custom Job
1. Click **Create**
2. **Name**: `marble-solitaire-YYYYMMDD` (any name)
3. **Region**: `us-central1`

## Step 3: Configure Worker
1. **Machine type**: `n1-standard-4`
2. **Accelerator type**: `NVIDIA_TESLA_T4`
3. **Accelerator count**: `1`
4. **Container image**: use the exact image URI printed by `build.sh` (e.g. `gcr.io/ai-dev-463705/marble-solitaire-training:20260423-120000`)
5. **Arguments** (add each as separate arg):
   - `--iterations=500`
   - `--episodes=50`
   - `--simulations=400`

## Step 4: Set Environment Variable
Add environment variable:
- **Key**: `AIP_MODEL_DIR`
- **Value**: `gs://ai-dev-463705-ml-artifacts/marble-solitaire/run-YYYYMMDD`

(Replace YYYYMMDD with today's date)

## Step 5: Submit
Click **Start Training**. Job takes ~1-2 hours on T4.

## Step 6: Download Results
After job completes, from the dev VM (if storage perms added) or Cloud Shell:
```bash
gsutil -m cp 'gs://ai-dev-463705-ml-artifacts/marble-solitaire/run-YYYYMMDD/onnx/*.onnx' \
  ~/app.gyatso.me/marble-solitaire-model/web/public/models/
cd ~/app.gyatso.me/marble-solitaire-model/web && npx vite build
```

Then commit web/dist/ and deploy via /go.
