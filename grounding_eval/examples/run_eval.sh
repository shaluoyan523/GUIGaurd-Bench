#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY first}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"

grounding-eval --config configs/example.json
