# Codebook-Based Model Compression — Proof of Concept

> **DISCLAIMER: This is a PROOF OF CONCEPT for educational purposes.**
> The concept of codebook-based compression with on-the-fly decompression is **not novel**.
> This repo demonstrates the concept and provides a starting point for further development.
> See [Related Work](#related-work) for existing research.

## Overview

This PoC demonstrates a model compression approach where:
- **Codebook is stored externally** (in the CLI/agent runtime, not in the model file)
- **Model weights are compressed** into indices referencing the codebook
- **On-the-fly decompression** happens per-layer during inference
- **Goal**: Reduce VRAM usage for memory-constrained environments (laptops, edge devices)

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  CLI / Agent (runtime)                              │
│                                                     │
│  ┌─────────────┐                                    │
│  │ Codebook    │  ← stored in system RAM            │
│  │ (external)  │     NOT in model file              │
│  └──────┬──────┘                                    │
│         │                                           │
│         │ decode per-layer at inference             │
│         ▼                                           │
│  ┌─────────────────────────────────────┐            │
│  │ VRAM                                │            │
│  │                                     │            │
│  │  Layer 0: [compressed indices]      │            │
│  │  Layer 1: [compressed indices]      │            │
│  │  Layer 2: [compressed indices]      │            │
│  │  ...                                │            │
│  │                                     │            │
│  │  (decode one layer at a time        │            │
│  │   during forward pass)              │            │
│  └─────────────────────────────────────┘            │
└─────────────────────────────────────────────────────┘
```

## How It Works

1. **Codebook Construction**: All model weights are clustered (k-means) to build a codebook of representative values
2. **Compression**: Each weight is replaced by an index (uint8) pointing to its nearest codebook entry
3. **Storage**: Model file contains only indices + metadata. Codebook is stored separately in the runtime
4. **Inference**: For each layer, indices are decoded back to full weights on-the-fly, used for computation, then discarded

## Results (PoC with small model)

```
Original model (FP32):       1.13 MB
Compressed model (indices):  0.28 MB  (4x smaller)
Codebook (external):         1 KB

VRAM during inference:
  Original:                  1.13 MB (all weights loaded)
  Compressed (per-layer):    0.78 MB (compressed + 1 layer decoded)
  VRAM saving:               30.5%

Decode overhead:             1.48 ms total
```

For larger models (e.g., 7B parameters), the VRAM saving is expected to be **~87%** because:
- Compressed model: ~1.75 GB (uint8 indices)
- One decoded layer: ~50-100 MB
- Total: ~1.85 GB vs original ~14 GB

## Trade-offs

| Aspect | Benefit | Cost |
|--------|---------|------|
| **VRAM** | 30-87% saving | — |
| **Disk** | 4x smaller model file | — |
| **Speed** | — | 20-40% slower (decode overhead) |
| **Quality** | — | Lossy (codebook approximation) |
| **Flexibility** | Codebook shared across models | Codebook management complexity |

## Installation

```bash
git clone https://github.com/dekai23/codebook-compression-poc.git
cd codebook-compression-poc
pip install -r requirements.txt
```

## Usage

```bash
python codebook_compression_poc.py
```

## Project Structure

```
codebook-compression-poc/
├── codebook_compression_poc.py   # Main PoC script
├── requirements.txt              # Dependencies
├── LICENSE                       # MIT License
├── README.md                     # This file
└── docs/
    └── technical_details.md      # Deep dive into implementation
```

## Limitations (PoC)

- **1D codebook**: Treats each weight as a scalar. Production should use vector quantization (multi-dimensional)
- **No GPU optimization**: Decode happens on CPU. Production needs fused GPU kernels (see LUT-GEMM)
- **Simple k-means**: No advanced clustering or learned codebooks
- **No format standard**: Custom format, not compatible with existing tools
- **Small model only**: Tested with ~1M param model. Needs validation with real LLMs

## Related Work

This PoC is inspired by and related to existing research:

1. **LUT-GEMM** (Park et al., 2022) — Lookup table for quantized matmul
   - https://arxiv.org/abs/2206.09557
2. **ANS-based LLM Compression** (2025) — Tile-level on-the-fly decompression
   - https://arxiv.org/abs/2606.15789
3. **CCQ** (2025) — Convolutional code for extreme low-bit quantization
   - https://arxiv.org/abs/2507.07145
4. **CommVQ** (Apple, 2024) — Commutative vector quantization for KV cache
   - https://machinelearning.apple.com/research/commutative-vector-quantization
5. **GLVQ** (NeurIPS 2025) — Grouped lattice vector quantization
   - https://papers.nips.cc/paper_files/paper/2025/hash/a0183ad2088503d707669e1d222bc8c4-Abstract-Conference.html
6. **DeepGEMM** (2023) — LUT-based inference for ultra low-precision CNNs
   - https://arxiv.org/abs/2304.09049
7. **LUT Tensor Core** (2024) — Hardware-software co-design for LUT inference
   - https://arxiv.org/abs/2408.06003

## Future Development

- [ ] Vector quantization (multi-dimensional codebook)
- [ ] GPU kernel for fused decode + matmul
- [ ] Standard model format (.codebook)
- [ ] Benchmark with real LLMs (Llama, Mistral, etc.)
- [ ] Adaptive codebook (per-layer, per-channel)
- [ ] Integration with existing inference frameworks (vLLM, llama.cpp)

## License

MIT License — see [LICENSE](LICENSE)

## Contributing

This is an educational PoC. Contributions welcome for:
- Bug fixes
- Documentation improvements
- Benchmark scripts
- GPU kernel implementations

---

**Author**: dekai23
**Date**: October 2026
**Status**: Proof of Concept — Not production-ready