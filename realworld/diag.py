import numpy as np, cupy as cp, polar_gpu as pg, run_real, sys
name=sys.argv[1] if len(sys.argv)>1 else 'ca-GrQc'; s=0.5; q=int(sys.argv[2]) if len(sys.argv)>2 else 24
G,cand,meta=run_real.setup(name,2000); n=G.n; Md=cp.asarray(G.M0.get().toarray())
lam,V=cp.linalg.eigh(Md); W=(V[cand[:,0]]-V[cand[:,1]])**2
tau,om,_=pg.composite(s,1.0,G.Lam(20),1e-2)
D=1/(lam[:,None]+cp.asarray(tau)[None]); Gp=cp.asnumpy(((W@(D*D))/(1+W@D))@cp.asarray(om))
# true marginals by eigen for top-50 by Gp
old=float(cp.sum(lam**-s)); top=np.argsort(-Gp)[:50]; tru={}
for e in top:
    A=Md.copy(); a,b=cand[e]; A[a,a]+=1;A[b,b]+=1;A[a,b]-=1;A[b,a]-=1; tru[e]=old-float(cp.sum(cp.linalg.eigvalsh(A)**-s))
print('rational vs true (top50) max rel err', max(abs(Gp[e]-tru[e])/tru[e] for e in top))
for kind in ('block','gauss'):
    Z,F,Gc=pg.sketches(G,cand,q,2026,kind)
    act=np.unique(cand); es=np.searchsorted(act,cand)
    XY,_,_=pg.msolve(pg.Op(G),cp.asarray(np.c_[Z,F]),tau,cp.asarray(act),1e-7)
    X,Y=XY[:,:,:q],XY[:,:,q:]
    h=cp.sum((X[:,es[:,0]]-X[:,es[:,1]])**2,-1); t=cp.sum((Y[:,es[:,0]]-Y[:,es[:,1]])**2,-1)
    taud=cp.asarray(tau)[:,None]; proxy=cp.asnumpy(cp.sum(cp.asarray(om)[:,None]*h/(1+t+(1+taud)*h),0))
    # exact h,t
    he=cp.asnumpy(((W)@(D*D)).T); de=cp.asnumpy((W@D).T)
    rel=np.abs(proxy-Gp)/Gp; best=int(np.argmax(Gp)); rank=int((proxy>proxy[best]).sum())
    hrel=np.abs(cp.asnumpy(h)-he)/he; drel=np.abs(cp.asnumpy(t+(1+taud)*h)-de)/de
    print(kind,'q',q,'proxy rel err median %.3f p90 %.3f max %.3f'%(np.median(rel),np.quantile(rel,.9),rel.max()),
          '| rank of true best in proxy:',rank,'| h relerr med %.3f d relerr med %.3f'%(np.median(hrel),np.median(drel)))
    o=np.argsort(-Gp)[:10]; print('  top10 Gp',np.round(Gp[o],4)); print('  their proxy',np.round(proxy[o],4))
