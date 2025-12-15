#!/usr/bin/env bash
# Reproduce all artifacts

# Constants
DATASET_NAMES=("ahsanayub/malicious-prompts" "jayavibhav/prompt-injection-safety" "synthetic-dataset")
EMBEDDING_MODEL_NAMES=("sentence-transformers/all-MiniLM-L6-v2" "thenlper/gte-large")
CLASSIFIER_MODEL_NAMES=("protectai/deberta-v3-base-prompt-injection-v2" "meta-llama/Prompt-Guard-86M" "meta-llama/Llama-Prompt-Guard-2-86M" "qualifire/prompt-injection-jailbreak-sentinel-v2")

# Configure bash
set -euo pipefail

# Get the script directory
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# Prepare to run the notebooks
pushd "${SCRIPT_DIR}/.."

# Run the dataset preparation notebook
echo "Running prepare-datasets notebook..."
uv run --with jupyter jupyter execute "${SCRIPT_DIR}/../notebooks/prepare-datasets.ipynb"

# Run the classifier notebooks
for DATASET_NAME in "${DATASET_NAMES[@]}"; do
  for EMBEDDING_MODEL_NAME in "${EMBEDDING_MODEL_NAMES[@]}"; do
    echo "Running classifier-embedding notebook for dataset: ${DATASET_NAME}, embedding model: ${EMBEDDING_MODEL_NAME}"
    DATASET_NAME="${DATASET_NAME}" \
      EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME}" \
      uv run --with jupyter jupyter execute "${SCRIPT_DIR}/../notebooks/classifier-embedding.ipynb"
  done
done

for DATASET_NAME in "${DATASET_NAMES[@]}"; do
  for EMBEDDING_MODEL_NAME in "${EMBEDDING_MODEL_NAMES[@]}"; do
    for CLASSIFIER_MODEL_NAME in "${CLASSIFIER_MODEL_NAMES[@]}"; do
      echo "Running classifier-hybrid notebook for dataset: ${DATASET_NAME}, embedding model: ${EMBEDDING_MODEL_NAME}, classifier model: ${CLASSIFIER_MODEL_NAME}"
      DATASET_NAME="${DATASET_NAME}" \
        EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME}" \
        CLASSIFIER_MODEL_NAME="${CLASSIFIER_MODEL_NAME}" \
        uv run --with jupyter jupyter execute "${SCRIPT_DIR}/../notebooks/classifier-hybrid.ipynb"
    done
  done
done

for DATASET_NAME in "${DATASET_NAMES[@]}"; do
  for CLASSIFIER_MODEL_NAME in "${CLASSIFIER_MODEL_NAMES[@]}"; do
    echo "Running classifier-transformer notebook for dataset: ${DATASET_NAME}, classifier model: ${CLASSIFIER_MODEL_NAME}"
    DATASET_NAME="${DATASET_NAME}" \
      CLASSIFIER_MODEL_NAME="${CLASSIFIER_MODEL_NAME}" \
      uv run --with jupyter jupyter execute "${SCRIPT_DIR}/../notebooks/classifier-transformer.ipynb"
  done
done

echo "Running classifier-multiclass for dataset: synthetic-dataset, embedding model: sentence-transformers/all-MiniLM-L6-v2"
DATASET_NAME="synthetic-dataset" \
  EMBEDDING_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2" \
  uv run --with jupyter jupyter execute "${SCRIPT_DIR}/../notebooks/classifier-multiclass.ipynb"

# Run the parameter search notebooks
for EMBEDDING_MODEL_NAME in "${EMBEDDING_MODEL_NAMES[@]}"; do
  echo "Running parameter-search-embedding notebook for dataset: ahsanayub/malicious-prompts, embedding model: ${EMBEDDING_MODEL_NAME}"
  DATASET_NAME="ahsanayub/malicious-prompts" \
    EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME}" \
    uv run --with jupyter jupyter execute "${SCRIPT_DIR}/../notebooks/parameter-search-embedding.ipynb"
done

echo "Running parameter-search-hybrid notebook for dataset: jayavibhav/prompt-injection-safety, embedding model: sentence-transformers/all-MiniLM-L6-v2, classifier model: qualifire/prompt-injection-jailbreak-sentinel-v2"
DATASET_NAME="jayavibhav/prompt-injection-safety" \
  EMBEDDING_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2" \
  CLASSIFIER_MODEL_NAME="qualifire/prompt-injection-jailbreak-sentinel-v2" \
  uv run --with jupyter jupyter execute "${SCRIPT_DIR}/../notebooks/parameter-search-hybrid.ipynb"

# Finish up running the notebooks
popd

# Prepare to run the AgentDojo benchmarks
"${SCRIPT_DIR}/start-ollama.sh" "hf.co/unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF:Q3_K_XL" "hf.co/unsloth/Qwen3-8B-GGUF:Q5_K_XL"

Run the AgentDojo benchmarks
echo "Running AgentDojo with no defense"
uv run "${SCRIPT_DIR}/../agentdojo/main.py" run-benchmark \
  --defense none \
  --logs "${SCRIPT_DIR}/../data/agentdojo/none/logs"
uv run "${SCRIPT_DIR}/../agentdojo/main.py" compute-results \
  --logs "${SCRIPT_DIR}/../data/agentdojo/none/logs" \
  --results "${SCRIPT_DIR}/../data/agentdojo/none/results.json"

echo "Running AgentDojo with embedding defense"
uv run "${SCRIPT_DIR}/../agentdojo/main.py" run-benchmark \
  --defense embedding \
  --logs "${SCRIPT_DIR}/../data/agentdojo/embedding/logs" \
  --defense-embedding-model-name sentence-transformers/all-MiniLM-L6-v2 \
  --defense-embedding-classifier-path "${SCRIPT_DIR}/../data/notebooks/classifier-embedding/jayavibhav-prompt-injection-safety/sentence-transformers-all-MiniLM-L6-v2/random-forest.joblib" \
uv run "${SCRIPT_DIR}/../agentdojo/main.py" compute-results \
  --logs "${SCRIPT_DIR}/../data/agentdojo/embedding/logs" \
  --results "${SCRIPT_DIR}/../data/agentdojo/embedding/results.json"

echo "Running AgentDojo with hybrid defense (protectai/deberta-v3-base-prompt-injection-v2)"
uv run "${SCRIPT_DIR}/../agentdojo/main.py" run-benchmark \
  --defense hybrid \
  --logs "${SCRIPT_DIR}/../data/agentdojo/hybrid/protectai-deberta-v3-base-prompt-injection-v2/logs" \
  --defense-embedding-model-name sentence-transformers/all-MiniLM-L6-v2 \
  --defense-embedding-classifier-path "${SCRIPT_DIR}/../data/notebooks/classifier-embedding/jayavibhav-prompt-injection-safety/sentence-transformers-all-MiniLM-L6-v2/random-forest.joblib" \
  --defense-transformer-model-name qualifire/prompt-injection-jailbreak-sentinel-v2 \
  --defense-transformer-model-benign-label benign
uv run "${SCRIPT_DIR}/../agentdojo/main.py" compute-results \
  --logs "${SCRIPT_DIR}/../data/agentdojo/hybrid/protectai-deberta-v3-base-prompt-injection-v2/logs" \
  --results "${SCRIPT_DIR}/../data/agentdojo/hybrid/protectai-deberta-v3-base-prompt-injection-v2/results.json"

echo "Running AgentDojo with hybrid defense (qualifire/prompt-injection-jailbreak-sentinel-v2)"
uv run "${SCRIPT_DIR}/../agentdojo/main.py" run-benchmark \
  --defense hybrid \
  --logs "${SCRIPT_DIR}/../data/agentdojo/hybrid/qualifire-prompt-injection-jailbreak-sentinel-v2/logs" \
  --defense-embedding-model-name sentence-transformers/all-MiniLM-L6-v2 \
  --defense-embedding-classifier-path "${SCRIPT_DIR}/../data/notebooks/classifier-embedding/jayavibhav-prompt-injection-safety/sentence-transformers-all-MiniLM-L6-v2/random-forest.joblib" \
  --defense-transformer-model-name qualifire/prompt-injection-jailbreak-sentinel-v2 \
  --defense-transformer-model-benign-label benign
uv run "${SCRIPT_DIR}/../agentdojo/main.py" compute-results \
  --logs "${SCRIPT_DIR}/../data/agentdojo/hybrid/qualifire-prompt-injection-jailbreak-sentinel-v2/logs" \
  --results "${SCRIPT_DIR}/../data/agentdojo/hybrid/qualifire-prompt-injection-jailbreak-sentinel-v2/results.json"

echo "Running AgentDojo with multiclass defense"
uv run "${SCRIPT_DIR}/../agentdojo/main.py" run-benchmark \
  --defense multiclass \
  --logs "${SCRIPT_DIR}/../data/agentdojo/multiclass/logs" \
  --defense-embedding-model-name sentence-transformers/all-MiniLM-L6-v2 \
  --defense-embedding-classifier-path "${SCRIPT_DIR}/../data/notebooks/classifier-multiclass/sentence-transformers-all-MiniLM-L6-v2/random-forest-goal.joblib" \
  --defense-llm-model-name "hf.co/unsloth/Qwen3-8B-GGUF:Q5_K_XL"
uv run "${SCRIPT_DIR}/../agentdojo/main.py" compute-results \
  --logs "${SCRIPT_DIR}/../data/agentdojo/multiclass/logs" \
  --results "${SCRIPT_DIR}/../data/agentdojo/multiclass/results.json"

echo "Running AgentDojo with LLM defense"
uv run "${SCRIPT_DIR}/../agentdojo/main.py" run-benchmark \
  --defense llm \
  --logs "${SCRIPT_DIR}/../data/agentdojo/llm/logs" \
  --defense-llm-model-name "hf.co/unsloth/Qwen3-8B-GGUF:Q5_K_XL"
uv run "${SCRIPT_DIR}/../agentdojo/main.py" compute-results \
  --logs "${SCRIPT_DIR}/../data/agentdojo/llm/logs" \
  --results "${SCRIPT_DIR}/../data/agentdojo/llm/results.json"

echo "Running AgentDojo with transformer (DeBERTa V2) defense"
uv run "${SCRIPT_DIR}/../agentdojo/main.py" run-benchmark \
  --defense transformer \
  --logs "${SCRIPT_DIR}/../data/agentdojo/transformer/protectai-deberta-v3-base-prompt-injection-v2/logs" \
  --defense-transformer-model-name protectai/deberta-v3-base-prompt-injection-v2 \
  --defense-transformer-model-benign-label SAFE
uv run "${SCRIPT_DIR}/../agentdojo/main.py" compute-results \
  --logs "${SCRIPT_DIR}/../data/agentdojo/transformer/protectai-deberta-v3-base-prompt-injection-v2/logs" \
  --results "${SCRIPT_DIR}/../data/agentdojo/transformer/protectai-deberta-v3-base-prompt-injection-v2/results.json"

echo "Running AgentDojo with transformer (Sentinel V2) defense"
uv run "${SCRIPT_DIR}/../agentdojo/main.py" run-benchmark \
  --defense transformer \
  --logs "${SCRIPT_DIR}/../data/agentdojo/transformer/qualifire-prompt-injection-jailbreak-sentinel-v2/logs" \
  --defense-transformer-model-name qualifire/prompt-injection-jailbreak-sentinel-v2 \
  --defense-transformer-model-benign-label benign
uv run "${SCRIPT_DIR}/../agentdojo/main.py" compute-results \
  --logs "${SCRIPT_DIR}/../data/agentdojo/transformer/qualifire-prompt-injection-jailbreak-sentinel-v2/logs" \
  --results "${SCRIPT_DIR}/../data/agentdojo/transformer/qualifire-prompt-injection-jailbreak-sentinel-v2/results.json"

echo "Running AgentDojo with data delimiter defense"
uv run "${SCRIPT_DIR}/../agentdojo/main.py" run-benchmark \
  --defense data_delimiter \
  --logs "${SCRIPT_DIR}/../data/agentdojo/data-delimiter/logs"
uv run "${SCRIPT_DIR}/../agentdojo/main.py" compute-results \
  --logs "${SCRIPT_DIR}/../data/agentdojo/data-delimiter/logs" \
  --results "${SCRIPT_DIR}/../data/agentdojo/data-delimiter/results.json"


# Finish up running the AgentDojo benchmarks
"${SCRIPT_DIR}/stop-ollama.sh"

# Prepare to run the final notebooks
pushd "${SCRIPT_DIR}/.."

# Run the results analysis notebook
echo "Running results-analysis notebook..."
uv run --with jupyter jupyter execute "${SCRIPT_DIR}/../notebooks/results-analysis.ipynb"

# Finish up running the final notebooks
popd

# Finish up
echo "All done."
