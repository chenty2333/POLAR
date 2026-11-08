import numpy as np, cupy as cp, polar_gpu as pg, graphs, sys, time
for name in sys.argv[1:]:
    n,E,meta=graphs.load(name); G=pg.Graph(n,E,1.0)
    for R in (1,4,8,16,32,48,64,128):
        X=cp.random.standard_normal((n,R)); 
        for _ in range(3): Y=G.M0@X
        cp.cuda.Device().synchronize(); t=time.perf_counter()
        for _ in range(20): Y=G.M0@X
        cp.cuda.Device().synchronize(); dt=(time.perf_counter()-t)/20
        print(name,'n',n,'nnz',G.M0.nnz,'R',R,'ms %.3f'%(dt*1e3),'ms/col %.3f'%(dt*1e3/R),flush=True)
