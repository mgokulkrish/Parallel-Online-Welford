# Parallel-Online-Welford
CUDA kernel (GPU) for calculating mean and variance in single pass.

# Algorithm used
Welford-Online:

```
def parallel_variance(n_a, avg_a, M2_a, n_b, avg_b, M2_b):
    n = n_a + n_b
    delta = avg_b - avg_a
    M2 = M2_a + M2_b + delta**2 * n_a * n_b / n
    var_ab = M2 / (n - 1)
    return var_ab
```

where each mean is calculated on a thread and combined using the above algorithm

# Details
Current implementation only supports 1D Tensor as of now. Can be generalized to multiple dimensions.

Modify `#define N`, in the file to modify the size. Add the data in a variable. Tested for most of the cases. More Testing is Welcome `:)`.

# Command
```
nvcc stddv.cu -o std
```

# TODO
Further parallelization can be done, while combining across blocks.