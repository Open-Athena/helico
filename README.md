# Project helico
Helico's goal is to enable robust experimentation around modeling and data improvements for AlphaFold3-like architectures.

## Motivation and key ideas

Open source [AlphaFold3](https://www.nature.com/articles/s41586-024-07487-w) clones have so far kept the AF3 architecture remarkably intact with mostly small tweaks and limited ablations.

If we are to match or eventually exceed more recent proprietary models like [IsoDDE](https://www.isomorphiclabs.com/articles/the-isomorphic-labs-drug-design-engine-unlocks-a-new-frontier), we need organized efforts to find better architectures. Helico is an opinionated take on how to do this. We prioritize:

**Open development**. We want to capture and share with the community not only the final best-performing model, but also the incremental and failed experiments that got us there, in real time.

**Automated workflows**. We want to configure compute environments (e.g. Lambda Labs, or AWS) with code that lives in this codebase. It should be possible for anyone to kick off training and evals on any supported compute environment that they have access to.

**Everything lives on github, wandb, or huggingface**. The source of truth on datasets, code, checkpoints, an so on is on public services not private filesystems (or in people's brains). For example, it should be possible for anyone to tell exactly what dataset and code was used to train a given checkpoint.

**Agentic coding**.
We aim for a low-abstraction codebase that is easy for agents to work with. Tests are prioritized over code. It should be possible for agents to autonomously run experiments and analyze the results. We try to document in this repo everything an agent has done wrong so it doesn't do it in the future. We also need to have good guardrails in place to monitor compute usage and data transfer costs.


## Project status

We are just getting started. Our initial implementation closely follows [Protenix](https://github.com/bytedance/Protenix) and our model can load protenix weights. Before we do expensive training runs from scratch we are planning to iterate on modeling improvements starting from these weights. (Code coming soon!)
