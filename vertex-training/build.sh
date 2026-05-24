#!/bin/bash
# Build and push the training container via Cloud Build.
# This works from the dev VM (Cloud Build SA handles the build).
#
# After this, submit the training job from GCP Console:
#   Vertex AI → Training → Create Custom Job
#   See submit-console-steps.md for details.

set -euo pipefail

PROJECT_ID="ai-dev-463705"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
IMAGE_URI="gcr.io/${PROJECT_ID}/marble-solitaire-training:${TIMESTAMP}"

echo "=== Building Marble Solitaire Training Container ==="
echo "Project: ${PROJECT_ID}"
echo "Image:   ${IMAGE_URI}"
echo ""

cd "$(dirname "$0")/.."
gcloud builds submit \
  --config vertex-training/cloudbuild.yaml \
  --substitutions "_IMAGE_URI=${IMAGE_URI}" \
  --gcs-source-staging-dir "gs://${PROJECT_ID}_cloudbuild/source" \
  --project "${PROJECT_ID}" \
  .

echo ""
echo "Container built and pushed: ${IMAGE_URI}"
echo ""
echo "Use this image URI when creating the job in Console:"
echo "  ${IMAGE_URI}"
echo ""
echo "Next: Submit training job from GCP Console."
echo "See vertex-training/submit-console-steps.md"
