# AgentDojo Run

- Defense: transformer-only
  - Transformer model: [Sentinel V2](https://huggingface.co/qualifire/prompt-injection-jailbreak-sentinel-v2)
  - Transformer malicious threshold: `0.5`
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
  - There was a bug in the code that caused the classification probabilities to be inverted. These have been fixed after the fact.
