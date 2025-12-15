#!/usr/bin/env bash
# Start the Ollama server.

# Configure bash
set -euo pipefail

# Constants
INITIAL_PROMPT="1+1="
WAIT_MAX_SECONDS=60

# Get the script directory
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# Environment configuration
export OLLAMA_HOST="0.0.0.0:11434"
export OLLAMA_KEEP_ALIVE="168h"

# Start Ollama
echo "Starting Ollama..."
nohup ollama serve &> "${SCRIPT_DIR}/ollama.log" &

# Wait for Ollama to start
for I in $(seq 1 $((WAIT_MAX_SECONDS + 1))); do
  if [[ "${I}" -eq $((WAIT_MAX_SECONDS + 1)) ]]; then
    echo "Failed to start Ollama."
    exit 1
  fi

  if ollama ps &> /dev/null; then
    echo "Ollama started."
    break
  else
    echo "[${I}/${WAIT_MAX_SECONDS}] Ollama not started yet. Retrying in 1 second..."
  fi

  sleep 1
done

# Load the initial model
for INITIAL_MODEL in "$@"; do
  echo "Loading ${INITIAL_MODEL}..."
  ollama run "${INITIAL_MODEL}" "${INITIAL_PROMPT}"
  echo "Done loading model."
done

echo "Done loading models."
