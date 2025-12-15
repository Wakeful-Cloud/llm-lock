# LLMLock

This is the final project for CS562.

## Documentation

### Setup

Note: the below steps assume you are running on Linux with an NVIDIA GPU.

1. Install dependencies:

   - [Git](https://git-scm.com/downloads) + [Git LFS](https://git-lfs.com/)
   - [Python 3.12](https://www.python.org/downloads/) + [UV](https://docs.astral.sh/uv/getting-started/installation/)
   - [Ollama](https://ollama.com/download)

2. Clone the repository:

   ```shell
   git clone https://github.com/Wakeful-Cloud/llm-lock.git
   ```

3. Run all experiments:

   ```
   ./scripts/run.sh
   ```

   Note: this script takes upwards of 2 weeks to complete on a single NVIDIA A100 80 GB GPU. You can
   modify the script to run only specific parts if desired.

### Project Structure

- `agentdojo/`: custom [AgentDojo](https://github.com/ethz-spylab/agentdojo) evaluation benchmark code
  - `defense_utils.py`: Defense utilities
  - `hybrid_defense.py`: Hybrid defense
  - `main.py`: AgentDojo evaluation benchmark entrypoint
  - `multiclass_defense.py`: Multiclass defense
  - `transformer_defense.py`: Transformer-only defense
- `data/`: data
  - `agentdojo/`: AgentDojo results
    - `[run name]/`: results for a specific benchmark
      - `README.md`: benchmark description, including notes about the run
      - `results.json`: summary results
      - `logs/`: detailed logs
  - `notebooks/`: notebook outputs (Please refer to the individual notebooks for details)
    - `[notebook name]/`: outputs for a specific notebook
  - `synthetic-dataset/`: synthetic dataset
    - `aggregated.json`: aggregated synthetic dataset
    - `raw/`: raw synthetic dataset
- `notebooks/`: Jupyter notebooks for experimentation and analysis
  - `classifier-embedding.ipynb`: embedding-only random forest classifier experiment
  - `classifier-hybrid.ipynb`: hybrid embedding + transformer classifier experiment
  - `classifier-multiclass.ipynb`: multiclass classifier experiment
  - `classifier-transformer.ipynb`: transformer-only classifier experiment
  - `parameter-search-embedding.ipynb`: parameter search for embedding-only classifier
  - `parameter-search-hybrid.ipynb`: parameter search for hybrid classifier
  - `prepare-dataset.ipynb`: prepare datasets for `classifier-embedding.ipynb`, `classifier-hybrid.ipynb`, `classifier-transformer.ipynb`, `parameter-search-embedding.ipynb`, and `parameter-search-hybrid.ipynb` notebooks
  - `results-analysis.ipynb`: compute and visualize results from the experiments
- `scripts/`: miscellaneous scripts
  - `run.sh`: main script to reproduce all artifacts
  - `start-ollama.sh`: start Ollama local server
  - `stop-ollama.sh`: stop Ollama local server

# Acknowledgements

## Illinois Computes Research Notebook

This research was supported in part by the
[Illinois Computes project](https://computes.illinois.edu), which is supported by the
[University of Illinois Urbana-Champaign](https://illinois.edu) and the
[University of Illinois System](https://uillinois.edu).

## Campus Cluster

This research made use of the Illinois Campus Cluster, a computing resource that is operated by the
[Illinois Campus Cluster Program (ICCP)](https://campuscluster.illinois.edu) in conjunction with the
[National Center for Supercomputing Applications (NCSA)](https://www.ncsa.illinois.edu) and which is
supported by funds from the [University of Illinois Urbana-Champaign](https://illinois.edu).
