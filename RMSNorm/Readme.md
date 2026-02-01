# RMS Normalization - Triton Implementation

## What is RMS Norm?

Root Mean Square (RMS) Normalization is a normalization technique used in modern Large Language Models like Llama, Mistral, and DeepSeek. It's simpler and more efficient than LayerNorm as it doesn't subtract the mean.

The mathematical formula:
```
RMS(x) = sqrt(mean(x²) + ε)
output = (x / RMS(x)) * weight
```

## Why This Implementation?

Traditional PyTorch implementations involve multiple operations:
- Computing squares of all elements
- Calculating mean across the dimension
- Taking reciprocal square root
- Element-wise multiplication with input and weight

This creates **multiple memory passes** and intermediate tensors.

This **fused Triton kernel** combines all operations into a **single GPU kernel launch**, processing one row (token) per CUDA program for optimal parallelization.

## How It Works

The kernel implementation:

1. **Row-wise parallelization**: Each CUDA program processes one token (row) independently
2. **Upcast to FP32**: Loads data and casts to float32 for numerical stability (follows Llama-3 implementation)
3. **Compute RMS**: 
   - Squares each element: `x²`
   - Sums and divides by dimension: `mean(x²)`
   - Computes reciprocal square root: `1/sqrt(mean(x²) + ε)`
4. **Normalize and scale**: Multiplies input by RMS inverse and weight
5. **Store result**: Writes output back to original dtype (fp16/bf16)

The implementation ensures numerical stability by performing all intermediate calculations in float32, even when input is in half precision.

## Performance Benchmarks

**Test Configuration**: M=4096 tokens (batch size), dtype=float16, CUDA GPU

| Hidden Dim (N) | Triton (GB/s) | PyTorch (GB/s) | Speedup |
|----------------|---------------|----------------|---------|
| 1,024          | 135.46        | 19.84          | **6.83x**   |
| 2,048          | 166.34        | 17.42          | **9.55x**   |
| 3,072          | 166.62        | 17.29          | **9.64x**   |
| 4,096          | 165.91        | 17.33          | **9.58x**   |
| 5,120          | 167.87        | 17.34          | **9.68x**   |
| 6,144          | 167.75        | 17.34          | **9.68x**   |
| 7,168          | 168.16        | 17.34          | **9.70x**   |
| 8,192          | 167.83        | 17.45          | **9.62x**   |

### Key Observations

- **Massive speedup**: Triton achieves **~9.6x faster** performance across all hidden dimensions
- **Consistent Triton throughput**: Maintains ~165-168 GB/s regardless of dimension size
- **PyTorch bottleneck**: Limited to ~17-19 GB/s due to multiple kernel launches and memory passes
- **Scaling efficiency**: Performance remains stable even as hidden dimension increases from 1K to 8K

## Why Such a Large Speedup?

1. **Kernel fusion**: Single kernel vs multiple PyTorch operations (pow, mean, rsqrt, mul)
2. **Memory efficiency**: One read, one write vs multiple intermediate tensors
3. **No kernel launch overhead**: Single GPU kernel vs ~4-5 separate kernels in PyTorch
4. **Optimized memory access**: Coalesced memory reads/writes with proper blocking

## Running the Code

```bash
# Run correctness test and benchmark
python rms_norm_kernel.py
```

Expected output:
```
--- Testing Correctness ---
Correctness Check PASSED! (Error < 1e-2)

--- Running Benchmark ---
[Benchmark results and plots]
```

## Numerical Stability Notes

- The implementation follows Llama-3 style: **always upcast to float32** for intermediate calculations
- This prevents numerical instability in fp16/bf16 accumulation
- Final output is cast back to original dtype
- Tolerance `atol=1e-2` is used in tests due to fp16 accumulation differences (acceptable for LLM inference)

## Requirements

- PyTorch
- Triton
- CUDA-capable GPU