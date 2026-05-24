#!/bin/bash
# Build and push the training container via Cloud Build.
#
# Replace YOUR_PROJECT with your GCP project id before running, or set
# PROJECT_ID env var.
#
# After this, submit the training job from the Vertex AI Console using
# one of the *-spec.yaml files in this directory.

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-YOUR_PROJECT}"
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
