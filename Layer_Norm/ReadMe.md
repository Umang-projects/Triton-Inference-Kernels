# Triton Layer Normalization Kernel

> A high-performance Layer Normalization implementation using Triton GPU programming language

## 📋 Overview

This project implements Layer Normalization using [Triton](https://github.com/openai/triton), a language and compiler for parallel programming. The implementation demonstrates how to write custom GPU kernels that can match or exceed PyTorch's native performance.

## 🚀 Features

- **Custom Triton Kernel**: Hand-written GPU kernel for Layer Normalization
- **Performance Benchmarking**: Comprehensive comparison with PyTorch's native implementation
- **Profiling Support**: Includes Chrome trace export for detailed performance analysis
- **Scalability Testing**: Benchmarks across various embedding dimensions (256 to 32,768)

## 📂 Repository Contents

- `layer_norm_kernel.py` - Main implementation file containing:
  - Triton kernel implementation
  - Python wrapper function
  - Benchmarking code
  - Profiling utilities

## 🛠️ Installation

```bash
pip install torch
pip install triton
```

## 💻 Usage

### Basic Usage

```python
import torch
from layer_norm_kernel import triton_layerNorm_kernel

# Create input tensor
batch_size, embedding_dim = 64, 512
X = torch.randn(batch_size, embedding_dim, device='cuda')
weight = torch.ones(embedding_dim, device='cuda')
bias = torch.zeros(embedding_dim, device='cuda')

# Run Triton kernel
output = triton_layerNorm_kernel(X, weight, bias)
```

### Run Benchmarks

```bash
python layer_norm_kernel.py
```

This will:
- Verify correctness against PyTorch implementation
- Measure execution time for both implementations
- Generate profiling traces (`triton_profile.json`, `pytorch_profile.json`)
- Create performance plots comparing bandwidth (GB/s) across different sizes

## 📊 Performance

The implementation includes benchmarking across embedding dimensions from 256 to 32,768 with a fixed batch size of 4,096. Results show performance in terms of memory bandwidth (GB/s).

View profiling results in Chrome by opening `chrome://tracing` and loading the generated JSON files.

## 🔧 How It Works

### Layer Normalization Formula

For each input row:
1. Calculate mean: `μ = sum(x) / n`
2. Calculate variance: `σ² = sum((x - μ)²) / n`
3. Normalize: `x̂ = (x - μ) / sqrt(σ² + ε)`
4. Scale and shift: `y = γ * x̂ + β`

### Triton Kernel Implementation

The kernel processes each row (batch element) independently:
- Each thread block handles one complete row
- Memory coalescing for efficient GPU memory access
- Uses block size as power of 2 for optimal performance

## 🎯 Key Implementation Details

- **Block Size**: Automatically set to next power of 2 of embedding dimension
- **Memory Access Pattern**: Row-wise processing with masked loads/stores
- **Numerical Stability**: Uses epsilon (1e-5) to prevent division by zero
- **Bounds Checking**: Ensures safe access with masking

## 📈 Benchmarking Results

Run the benchmark to see performance comparison:
- Execution time (milliseconds)
- Memory bandwidth (GB/s)
- Speedup factor over PyTorch

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements!

## 📄 License

MIT License

## 🙏 Acknowledgments

- [OpenAI Triton](https://github.com/openai/triton) for the GPU programming framework
- PyTorch team for the reference implementation

## 📚 References

- [Layer Normalization Paper](https://arxiv.org/abs/1607.06450)
- [Triton Documentation](https://triton-lang.org/)

---

**Note**: This project requires an NVIDIA GPU with CUDA support.