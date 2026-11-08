"""GPU arithmetic sanity check: fp64 GEMV/GEMM/SpMM/reductions against NumPy, repeated."""
import numpy as np, cupy as cp, cupyx.scipy.sparse as csp, scipy.sparse as sp
rng=np.random.default_rng(0); out={}
A=rng.normal(size=(1005,1005)); x=rng.normal(size=(1005,1)); ref=A@x; Ad=cp.asarray(A); xd=cp.asarray(x)
out['gemv']=max(float(np.abs((Ad@xd).get()-ref).max()) for _ in range(50))
B=rng.normal(size=(2000,2000)); refB=B@B; Bd=cp.asarray(B)
out['gemm']=max(float(np.abs((Bd@Bd).get()-refB).max()) for _ in range(10))
nnz=2_000_000; S=sp.csr_matrix((rng.normal(size=nnz),(rng.integers(200000,size=nnz),rng.integers(200000,size=nnz))),shape=(200000,200000)); X=rng.normal(size=(200000,48)); refS=S@X; Sd=csp.csr_matrix(S); Xd=cp.asarray(X)
out['spmm']=max(float(np.abs((Sd@Xd).get()-refS).max()) for _ in range(20))
v=rng.normal(size=50_000_000); vd=cp.asarray(v); s=[float(cp.sum(vd*vd)) for _ in range(20)]
out['reduce_rel']=abs(np.mean(s)-v@v)/(v@v); out['reduce_spread']=max(s)-min(s)
f=cp.asarray(rng.normal(size=(4096,4096)).astype(np.float32)); g=f@f
out['fp32_gemm_repeat_equal']=sum(bool(cp.array_equal(f@f,g)) for _ in range(20))
print(out); assert out['gemv']<1e-10 and out['gemm']<1e-8 and out['spmm']<1e-10 and out['reduce_rel']<1e-12, 'GPU FAILED'
print('GPU OK')
