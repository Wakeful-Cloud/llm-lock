# AgentDojo Run

- Defense: embedding-only
  - Embedding model: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
  - Embedding classifier minimum benign threshold: `0.5`
  - Embedding classifier minimum malicious threshold: `0.5`
- Environment
  - System: [Illinois Computes Research Notebook (ICRN)](https://computes.illinois.edu/resources/icrn/)
  - Hardware
    - CPU: AMD EPYC 7413
    - Memory: 32 GiB DDR4
    - GPU: 1x NVIDIA A100 (80 GB VRAM, SXM4 form factor)
    - Storage: [NCSA Harbor](https://docs.ncsa.illinois.edu/systems/harbor/en/latest/index.html)
  - Software
    - OS: Ubuntu 22.04.3 LTS
    - Kernel: 5.15.0-156-generic
    - Python: 3.12.12
- Notes
  - The runs was interrupted once due to AgentDojo failing on travel suite with no attack on user
    task 12 or 13.
