#!/bin/bash
# Submit marble solitaire training job to Vertex AI Custom Training.
#
# Prerequisites:
#   gcloud auth login
#   gcloud config set project ai-dev-463705
#   gcloud services enable aiplatform.googleapis.com
#
# Usage:
#   ./vertex-training/submit.sh

set -euo pipefail

PROJECT_ID="ai-dev-463705"
REGION="us-central1"
IMAGE_URI="gcr.io/${PROJECT_ID}/marble-solitaire-training:latest"
JOB_NAME="marble-solitaire-$(date +%Y%m%d-%H%M%S)"
GCS_OUTPUT="gs://${PROJECT_ID}-ml-artifacts/marble-solitaire/${JOB_NAME}"

echo "=== Marble Solitaire Vertex AI Training ==="
echo "Project:  ${PROJECT_ID}"
echo "Region:   ${REGION}"
echo "Image:    ${IMAGE_URI}"
echo "Output:   ${GCS_OUTPUT}"
echo ""

# Step 1: Build and push the training container
echo "Building training container..."
cd "$(dirname "$0")/.."
gcloud builds submit \
  --tag "${IMAGE_URI}" \
  --project "${PROJECT_ID}" \
  -f vertex-training/Dockerfile \
  .

# Step 2: Create GCS bucket for artifacts (if needed)
gsutil ls "gs://${PROJECT_ID}-ml-artifacts" 2>/dev/null || \
  gsutil mb -l "${REGION}" "gs://${PROJECT_ID}-ml-artifacts"

# Step 3: Submit training job
echo ""
echo "Submitting training job: ${JOB_NAME}"
gcloud ai custom-jobs create \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --display-name="${JOB_NAME}" \
  --worker-pool-spec="\
machine-type=n1-standard-4,\
accelerator-type=NVIDIA_TESLA_T4,\
accelerator-count=1,\
container-image-uri=${IMAGE_URI}" \
  --args="--iterations=500,--episodes=50,--simulations=400" \
  --env-vars="AIP_MODEL_DIR=${GCS_OUTPUT}"

echo ""
echo "Job submitted! Monitor at:"
echo "  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}"
echo ""
echo "When complete, download models:"
echo "  gsutil -m cp '${GCS_OUTPUT}/onnx/*.onnx' web/public/models/"
echo "  cd web && npx vite build"
