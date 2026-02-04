#!/bin/sh
set -e

echo "Initializing knowledgeBase..."

mkdir -p /tmp/knowledgeBase

# Copy model into shared volume only if not already present
if [ ! -f /tmp/knowledgeBase/currentAiSolution_INFERENCE.keras ]; then
  cp /models/currentAiSolution_INFERENCE.keras /tmp/knowledgeBase/
  echo "Model copied to shared volume"
else
  echo "Model already present in shared volume"
fi

exec "$@"