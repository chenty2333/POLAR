"""Row-major CSR SpMM: one warp per row, lanes over columns; the matrix is streamed once per 32 columns."""
import cupy as cp
_src = r'''
extern "C" __global__ void csr_spmm(const int n, const int R, const int* __restrict__ ptr,
    const int* __restrict__ idx, const double* __restrict__ val, const double* __restrict__ X, double* __restrict__ Y){
  int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; int lane = threadIdx.x & 31;
  if (warp >= n) return;
  int lo = ptr[warp], hi = ptr[warp+1];
  for (int c0 = 0; c0 < R; c0 += 32){
    int c = c0 + lane; double acc = 0.0;
    if (c < R) for (int k = lo; k < hi; ++k) acc += val[k] * X[(long long)idx[k] * R + c];
    if (c < R) Y[(long long)warp * R + c] = acc;
  }
}'''
_k = cp.RawKernel(_src, 'csr_spmm')


def spmm(A, X):
    """A: cupyx csr_matrix (int32 indices, float64), X: C-contiguous (n, R) float64."""
    if X.ndim == 1 or X.shape[1] < 8:
        return A @ X
    X = cp.ascontiguousarray(X); n, R = A.shape[0], X.shape[1]; Y = cp.empty((n, R))
    threads = 256; blocks = (n * 32 + threads - 1) // threads
    _k((blocks,), (threads,), (cp.int32(n), cp.int32(R), A.indptr, A.indices, A.data, X, Y))
    return Y
