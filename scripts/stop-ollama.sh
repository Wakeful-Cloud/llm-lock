#!/usr/bin/env bash
# Stop the Ollama server.

# Configure bash
set -euo pipefail

# Get the script directory
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# Stop Ollama
pkill ollama

# Delete the log
rm "${SCRIPT_DIR}/ollama.log"
