# Triton Inference Kernels

Custom OpenAI Triton kernels for high-performance model inference. This repository accelerates deep learning models on NVIDIA GPUs by leveraging Triton's ease of development with CUDA-level performance.

## Overview

Triton is a language and compiler for writing highly efficient custom Deep Learning operations. This repository provides optimized Triton kernels specifically designed for model inference workloads, offering significant performance improvements over standard implementations.

## Features

- **High Performance**: CUDA-level performance with Python-like productivity
- **GPU Acceleration**: Optimized for NVIDIA GPUs
- **Easy Integration**: Simple Python API for seamless integration into existing workflows
- **Custom Kernels**: Hand-tuned implementations for common inference operations

## Repository Structure

```
Triton-Inference-Kernels/
├── Vector_add/          # Vector addition kernel implementation
└── .gitignore
```

## Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support
- PyTorch
- OpenAI Triton

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Umang-projects/Triton-Inference-Kernels.git
cd Triton-Inference-Kernels
```

2. Install required dependencies:
```bash
pip install torch triton
```

## Available Kernels

### Vector Addition
A basic Triton kernel implementation for vector addition operations, demonstrating the fundamentals of Triton programming.

## Usage

```python
import torch
from Vector_add import vector_add_triton

# Create sample tensors
x = torch.randn(10000, device='cuda')
y = torch.randn(10000, device='cuda')

# Run the Triton kernel
result = vector_add_triton(x, y)
```

## Performance Benefits

Triton kernels provide several advantages:
- Automatic memory coalescing
- Optimized register usage
- Reduced memory bandwidth requirements
- Better cache utilization
- Simplified kernel development compared to raw CUDA

## Development

To add new kernels:

1. Create a new directory for your kernel
2. Implement the Triton kernel code
3. Add benchmarking and validation tests
4. Update this README with usage examples

## Benchmarking

Each kernel includes benchmarking utilities to measure performance against baseline implementations. Run benchmarks with:

```bash
python Vector_add/benchmark.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Resources

- [Triton Documentation](https://triton-lang.org/)
- [Triton GitHub Repository](https://github.com/openai/triton)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)

## License

This project is open source and available under the MIT License.

## Contact

For questions or suggestions, please open an issue on GitHub.

## Acknowledgments

- OpenAI for developing and maintaining the Triton language and compiler
- The CUDA and PyTorch communities for their extensive documentation and support