import torch
import triton
import triton.language as tl

# ==========================================
# 1. TERA TRITON KERNEL (As provided)
# ==========================================

@triton.jit
def rms_norm_kernel(
    x_ptr, w_ptr, output_ptr,
    stride_x_row, stride_w, stride_out_row,
    N_COLS, eps,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start_ptr = x_ptr + (row_idx * stride_x_row)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_COLS

    # Load Data (Important: Cast to float32 for stability like Llama/DeepSeek)
    x_row = tl.load(row_start_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    w_row = tl.load(w_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Math
    x_sq = x_row * x_row
    sum_sq = tl.sum(x_sq, axis=0)
    mean_sq = sum_sq / N_COLS
    rms_inv = tl.rsqrt(mean_sq + eps)
    
    output = x_row * rms_inv * w_row

    # Store (Convert back to original dtype implicitly by pointer type)
    output_row_ptr = output_ptr + (row_idx * stride_out_row)
    tl.store(output_row_ptr + offsets, output, mask=mask)

def rms_norm_triton(x, weight, eps=1e-6):
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    
    rms_norm_kernel[grid](
        x, weight, y,
        x.stride(0), weight.stride(0), y.stride(0),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return y

# ==========================================
# 2. PYTORCH REFERENCE (The Gold Standard)
# ==========================================

def rms_norm_torch(x, weight, eps=1e-6):
    # Llama-3 style: Always upcast to float32 for calculation
    x_fp32 = x.float()
    
    # Square -> Mean
    mean_sq = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    
    # Rsqrt
    rms_inv = torch.rsqrt(mean_sq + eps)
    
    # Normalize & Scale
    out = x_fp32 * rms_inv * weight.float()
    
    # Cast back to original type (fp16/bf16)
    return out.to(x.dtype)

# ==========================================
# 3. CORRECTNESS TEST
# ==========================================

def test_correctness():
    torch.manual_seed(0)
    M, N = 4, 4096  # Batch size 4, Hidden dim 4096
    dtype = torch.float16 # T4 pe float16 best hai

    # Inputs
    x = torch.randn((M, N), device='cuda', dtype=dtype)
    w = torch.randn((N), device='cuda', dtype=dtype)

    # 1. Run PyTorch
    y_ref = rms_norm_torch(x, w)

    # 2. Run Triton
    y_tri = rms_norm_triton(x, w)

    # 3. Compare
    # atol=1e-2 thoda loose rakha hai kyunki fp16 accumulation mein thoda diff aata hai
    if torch.allclose(y_ref, y_tri, atol=1e-2, rtol=1e-2):
        print(f"Correctness Check PASSED! (Error < 1e-2)")
    else:
        print(f"Correctness Check FAILED!")
        print(f"Max Diff: {(y_ref - y_tri).abs().max().item()}")

# ==========================================
# 4. BENCHMARKING (GB/s Calculation)
# ==========================================

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # X-axis: Vector Size (Hidden Dim)
        x_vals=[1024 * i for i in range(1, 9)], # 1024, 2048 ... 8192
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'PyTorch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='rms-norm-performance',
        args={'M': 4096}, # Batch Size fixed at 4096 tokens
    )
)
def benchmark(M, N, provider):
    # Setup Data
    dtype = torch.float16
    x = torch.randn((M, N), device='cuda', dtype=dtype)
    w = torch.randn((N), device='cuda', dtype=dtype)
    
    quantiles = [0.5, 0.2, 0.8]
    
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: rms_norm_torch(x, w), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: rms_norm_triton(x, w), quantiles=quantiles)
    
    # Calculate GB/s
    # Formula: 2 reads (x, w) + 1 write (y). 
    # NOTE: w is negligible for large M, so mainly 2 * M * N * element_size
    gbps = lambda ms: (2 * M * N * x.element_size()) * 1e-9 / (ms * 1e-3)
    
    return gbps(ms), gbps(max_ms), gbps(min_ms)

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("--- Testing Correctness ---")
    test_correctness()
    
    print("\n--- Running Benchmark ---")
    benchmark.run(show_plots=True, print_data=True)