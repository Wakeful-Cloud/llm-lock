# AgentDojo Run

- Defense: N/A
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
  - The runs were interrupted quite a few times due to ICRN time limits, [this bug](https://github.com/ethz-spylab/agentdojo/commit/4069297e57604e7d054f61c1be25fb1c697fee0e), and [this bug](https://github.com/ethz-spylab/agentdojo/commit/6ebf27e7648d8f866e8ee8c850e0c3a78b9d203f).
  - We started this run with an overly-ambitious set of attacks. Due to time constrains, we reduce the number of attacks part of the way through. For transparency, we leave the half-complete attacks in the results, though we do not use them in our analysis.
