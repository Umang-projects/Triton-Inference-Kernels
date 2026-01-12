import torch
import pytest
from triton_core.vector_ops import add 

@pytest.mark.parametrize("size", [128, 1024, 40960])
def test_vector_add(size):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    torch.manual_seed(0)
    x = torch.randn(size, device='cuda')
    y = torch.randn(size, device='cuda')

    # Expected (PyTorch)
    expected = x + y

    # Actual (Triton)
    actual = add(x, y)

    # Compare
    torch.testing.assert_close(actual, expected)
    print(f"Pass for size {size}")
