# Technical Details: Codebook-Based Model Compression

## 1. Codebook Construction

The codebook is built using k-means clustering over all model weights:

```python
def kmeans_codebook(weights, k, n_iter=10):
    # Initialize: sample k random weights
    indices = np.random.choice(len(weights), size=k, replace=False)
    centroids = weights[indices].copy()
    
    for _ in range(n_iter):
        # Assign each weight to nearest centroid
        distances = np.abs(weights[:, None] - centroids[None, :])
        assignments = np.argmin(distances, axis=1)
        
        # Update centroids
        for i in range(k):
            mask = assignments == i
            if mask.any():
                centroids[i] = weights[mask].mean()
    
    return centroids
```

**Key parameters:**
- `k` (codebook size): Number of representative values. Larger k = better quality, less compression
- `n_iter` (iterations): More iterations = better convergence, slower build time

## 2. Compression (Encoding)

Each weight is replaced by the index of its nearest codebook entry:

```python
def encode(weights, codebook):
    flat = weights.flatten()
    distances = np.abs(flat[:, None] - codebook[None, :])
    indices = np.argmin(distances, axis=1).astype(np.uint8)
    return indices
```

**Compression ratio:**
- Original: 4 bytes per weight (FP32)
- Compressed: 1 byte per weight (uint8 index)
- Ratio: 4x (for codebook_size <= 256)

## 3. Decompression (Decoding)

On-the-fly decoding during inference:

```python
def decode(indices, codebook, shape):
    decoded = codebook[indices.flatten()].flatten()
    return decoded.reshape(shape)
```

## 4. VRAM Analysis

### Original Model
```
VRAM = sum(layer_size for all layers)
     = num_layers * avg_layer_size
```

### Compressed Model (Per-Layer Decode)
```
VRAM = compressed_model_size + max_layer_size
     = (num_layers * avg_layer_size / compression_ratio) + max_layer_size
```

### Break-Even Point
Compression is beneficial when:
```
compressed_model_size + max_layer_size < original_model_size
```

For a model with N layers and compression ratio R:
```
(original_size / R) + max_layer < original_size
=> max_layer < original_size * (1 - 1/R)
```

For R=4: max_layer must be < 75% of original model size (almost always true)

## 5. GPU Optimization Opportunities

### Current (PoC)
- Decode on CPU
- Transfer decoded weights to GPU per layer
- High latency due to CPU-GPU transfer

### Production Approach
- **Fused decode + GEMM kernel**: Decode directly in GPU shared memory
- **Double buffering**: Decode next layer while computing current layer
- **Codebook in constant memory**: Fast broadcast access
- **Vectorized lookup**: Use GPU warp-level parallelism for decode

### LUT-GEMM Style
- Precompute partial products in lookup table
- Eliminate dequantization entirely
- Requires custom CUDA kernel

## 6. Codebook Size vs Quality

| Codebook Size | Bits per Index | Compression Ratio | Quality |
|---------------|----------------|-------------------|---------|
| 64 | 6 bits | ~6.4x | Lower |
| 128 | 7 bits | ~5.7x | Good |
| 256 | 8 bits | 4.0x | Better |
| 512 | 9 bits | 3.6x | High |
| 1024 | 10 bits | 3.2x | Very High |

**Note**: These are theoretical ratios. Actual ratios depend on weight distribution and clustering quality.

## 7. Comparison with Existing Methods

| Method | Codebook Location | Decode Timing | GPU Optimized |
|--------|-------------------|---------------|---------------|
| **This PoC** | External (CLI) | Per-layer, CPU | No |
| **GPTQ** | Internal (model) | Pre-decode | Yes |
| **AWQ** | Internal (model) | Pre-decode | Yes |
| **LUT-GEMM** | Internal (kernel) | Fused | Yes |
| **ANS-based** | Internal (model) | Per-tile, GPU | Yes |

## 8. Potential Improvements

1. **Vector Quantization**: Cluster weight vectors (e.g., 4D) instead of scalars
2. **Residual VQ**: Multiple codebooks for hierarchical refinement
3. **Learned Codebook**: End-to-end training with codebook as part of model
4. **Adaptive Codebook**: Different codebooks per layer based on weight distribution
5. **Sparse Codebook**: Only store non-zero entries for sparse models
6. **Quantization-Aware Training**: Fine-tune model with quantization in the loop