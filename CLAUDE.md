# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Helico is an AlphaFold3 clone built from scratch in PyTorch for experimentation.

## Project Structure

```
src/helico/
  __init__.py    Package entry (exports Helico, HelicoConfig)
  model.py       All neural network modules in a single file
  data.py        Data pipeline (CCD, mmCIF, tokenizer, MSA, cropping)
  train.py       Training loop, DDP, checkpointing, inference
tests/
  test_data.py   Integration tests for the data pipeline
  test_model.py  Integration tests for all model components
```

## Build & Test Commands

- **Install**: `uv pip install -e ".[dev]"`
- **Run all tests**: `uv run pytest`
- **Run fast tests** (skip CCD/seqres): `uv run pytest -k "not CCD and not Seqres"`
- **Run a single test**: `uv run pytest tests/test_model.py::TestTriangleOps::test_tri_mul_outgoing_shape -v`
- **Train (synthetic)**: `helico-train --synthetic --n-blocks 2 --n-diffusion-token-blocks 2 --max-steps 100`

## Architecture

- The model lives in `src/helico/model.py` using PyTorch.
- Target GPUs: **H100 / B200 only**. No other architectures.
- Always use **cuEquivariance** kernels directly — no PyTorch-only fallback code paths.
- Three cuEquivariance kernels are used: `triangle_multiplicative_update`, `triangle_attention`, `attention_pair_bias`.
- Prioritize simplicity and single code paths over flexibility.

## Testing

- Unit tests for all non-trivial functionality.
- Always full integration tests — **never use stubs or mocks**.
- Tests run on GPU with bfloat16 precision.

## Training Data

- Set `HELICO_RAW_DIR` to the directory containing raw data (components.cif, mmCIF/, MSA tars, etc.)
- Set `HELICO_PROCESSED_DIR` to the directory for processed outputs (pickles, manifest, tar indices)
- Both env vars are **required** for any data operations. No paths are hardcoded.
- Preprocessing: `helico-preprocess all` (structures + MSA tar indices)
- See `LOG.md` for actual paths and commands used on our machines.
- Processing follows the [Boltz2 flow](https://github.com/jwohlwend/boltz).

## Reference Material

Key papers and repos to be familiar with:

- **AlphaFold3**: [paper](https://www.nature.com/articles/s41586-024-07487-w) / [code](https://github.com/google-deepmind/alphafold3)
- **Boltz2**: [paper](https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1.full) / [code](https://github.com/jwohlwend/boltz)
- **OpenFold3**: [whitepaper](https://github.com/aqlaboratory/openfold-3/blob/main/assets/whitepaper.pdf) / [code](https://github.com/aqlaboratory/openfold-3) / [docs](https://openfold-3.readthedocs.io/en/latest/)
- **cuEquivariance**: [code](https://github.com/NVIDIA/cuEquivariance) / [docs](https://docs.nvidia.com/cuda/cuequivariance/)
- **Protenix**: [code](https://github.com/bytedance/Protenix) / [paper](https://github.com/bytedance/Protenix/blob/main/docs/PTX_V1_Technical_Report_202602042356.pdf)
