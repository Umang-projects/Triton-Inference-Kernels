# Fused Rotary Positional Embedding (RoPE) - Triton Implementation

## What is RoPE?

Rotary Positional Embedding (RoPE) is a positional encoding technique used in modern Large Language Models (Llama, Mistral, DeepSeek). It encodes position information by rotating query and key vectors in the embedding space.

The rotation operation splits the embedding dimension in half and applies: 
```
y₁ = x₁ · cos θ - x₂ · sin θ
y₂ = x₁ · sin θ + x₂ · cos θ
```

## Why This Implementation?

Traditional PyTorch implementations involve:
- Slicing the tensor into two halves
- Computing rotations separately
- Concatenating results back together

This creates **high memory overhead** and multiple memory passes.

This **fused Triton kernel** performs the entire operation in a **single memory pass**, eliminating intermediate allocations and significantly improving performance.

## How It Works

The kernel implementation:

1. **Maps program IDs to tokens**: Each CUDA program processes one token (row)
2. **Loads data in parallel**: Simultaneously loads the first half `x₁` and second half `x₂` of the embedding dimension
3. **Applies rotation**: Computes `y₁` and `y₂` using precomputed cos/sin values
4. **Stores results directly**: Writes output in a single pass without intermediate buffers

The block size is automatically set to the next power of 2 for optimal memory coalescing.

## Performance Benchmarks

**Test Configuration**: DIM=128, CUDA GPU

| Tokens (N) | Triton (GB/s) | PyTorch (GB/s) | Speedup |
|------------|---------------|----------------|---------|
| 2,048      | 14.63         | 8.17           | **1.79x**   |
| 6,144      | 188.08        | 96.00          | **1.96x**   |
| 10,240     | 192.00        | 121.90         | **1.58x**   |
| 14,336     | 238.93        | 125.75         | **1.90x**   |
| 18,432     | 240.42        | 112.85         | **2.13x**   |
| 22,528     | 243.11        | 97.95          | **2.48x**   |
| 26,624     | 240.58        | 85.33          | **2.82x**   |
| 30,720     | 241.27        | 75.29          | **3.20x**   |
| 34,816     | 245.18        | 69.63          | **3.52x**   |
| 38,912     | 247.32        | 64.57          | **3.83x**   |
| 43,008     | 246.23        | 62.88          | **3.92x**   |
| 47,104     | 248.79        | 61.98          | **4.01x**   |
| 51,200     | 249.35        | 60.57          | **4.12x**   |
| 55,296     | 248.34        | 60.99          | **4.07x**   |
| 59,392     | 250.25        | 59.14          | **4.23x**   |
| 63,488     | 251.27        | 59.15          | **4.25x**   |

### Key Observations

- **Triton maintains consistent throughput**: ~240-250 GB/s regardless of batch size
- **PyTorch degrades with scale**: Performance drops from 121 GB/s to 59 GB/s as tokens increase
- **Speedup increases with scale**: From 1.79x at small batches to **4.25x at large batches**
- **Memory bandwidth efficiency**: Triton achieves near-optimal memory bandwidth utilization

## Running the Code

```bash
# Run correctness test
python rope_kernel.py

# Expected output: "Correctness Check Passed!"
```

The script will also generate benchmark plots comparing Triton vs PyTorch performance.

## Requirements

- PyTorch
- Triton
- CUDA-capable GPU