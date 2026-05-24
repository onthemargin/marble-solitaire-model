# Submit Training Job from GCP Console

After running `./vertex-training/build.sh`, submit the job via Vertex AI Console.

## Step 1: Open Vertex AI Custom Jobs
Navigate to **Vertex AI → Training → Custom Jobs** in your GCP project.

## Step 2: Create Custom Job
1. Click **Create**
2. **Name**: any (e.g. `marble-solitaire-YYYYMMDD`)
3. **Region**: us-central1 (or wherever your T4 GPU quota is)

## Step 3: Configure Worker
1. **Machine type**: `n1-standard-4`
2. **Accelerator type**: `NVIDIA_TESLA_T4`
3. **Accelerator count**: `1`
4. **Container image**: paste the image URI printed by `build.sh`
5. **Arguments**: see `phase2-spec.yaml` or `phase3-spec.yaml` in this directory for the full arg list per phase

## Step 4: Set Environment Variable
- **Key**: `AIP_MODEL_DIR`
- **Value**: a GCS path under your artifacts bucket (e.g. `gs://YOUR_BUCKET/marble-solitaire/run-YYYYMMDD/`)

## Step 5: Submit
Click **Start Training**.

## Step 6: Download Results
After job completes:
```bash
gsutil -m cp 'gs://YOUR_BUCKET/marble-solitaire/<run>/onnx/*.onnx' web/public/models/
cd web && npx vite build
```
