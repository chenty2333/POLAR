import numpy as np, cupy as cp, polar_gpu as pg, run_real, exact, sys, time
s=0.5; k=20
for name in sys.argv[1:]:
    G,cand,meta=run_real.setup(name,2000); Md=G.M0.get().toarray()
    ref=exact.exact_greedy(Md,cand,k,s)['sequence']; refg=exact.eval_prefix(Md,cand,ref,s)[-1]
    for q in (24,64,128):
        sk=pg.sketches(G,cand,q,2026)
        for B in (4,8,16,32):
            out=pg.polar(G,cand,k,s,q=q,B=B,sketch=sk); g=exact.eval_prefix(Md,cand,out['sequence'],s)[-1]
            print(name,'q',q,'B',B,'ratio %.4f'%(g/refg),'sec %.2f'%out['seconds'],'spmv',out['spmv'],flush=True)
