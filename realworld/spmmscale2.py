import numpy as np, cupy as cp, polar_gpu as pg, graphs, sys, time
from spmm import spmm
for name in sys.argv[1:]:
    n,E,meta=graphs.load(name); G=pg.Graph(n,E,1.0); A=G.M0
    print(A.indices.dtype, A.indptr.dtype, A.data.dtype)
    for R in (8,16,32,48,64,128):
        X=cp.random.standard_normal((n,R)); ref=A@X; Y=spmm(A,X); err=float(cp.abs(Y-ref).max())
        cp.cuda.Device().synchronize(); t=time.perf_counter()
        for _ in range(20): Y=spmm(A,X)
        cp.cuda.Device().synchronize(); dt=(time.perf_counter()-t)/20
        print(name,'R',R,'custom ms %.3f'%(dt*1e3),'ms/col %.3f'%(dt*1e3/R),'err',err,flush=True)
