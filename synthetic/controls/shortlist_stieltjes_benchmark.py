import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np, scipy.sparse as sp
from heapq import heapify, heappop
from time import perf_counter
q,k,N,tol=24,15,3000,1e-7; started=perf_counter()
def quad(s,r=20,c=1.):
    if s==1: return np.zeros(1),np.ones(1)
    i=np.arange(r); a=(1-2*s)/(4*i*i-1.); i=np.arange(1,r); b=np.sqrt((i-s)*(i+s-1))/(2*i-1); b[0]*=np.sqrt(2)
    y,V=np.linalg.eigh(np.diag(a)+np.diag(b,1)+np.diag(b,-1)); return c*((1-y)/(1+y))[::-1],c**(1-s)*(2*V[0]**2/(1+y))[::-1]
def orth(n,rng,center=False,blocks=False):
    W=rng.normal(size=(n,q))
    if not blocks:
        if center: W-=W.mean(0)
        return np.linalg.qr(W,mode='reduced')[0]*np.sqrt((n-int(center))/q)
    for i in range(0,n,q):
        Q,R=np.linalg.qr(W[i:i+q].T,mode='reduced'); W[i:i+q]=(Q*np.sign(np.diag(R))).T
    if center: W-=W.mean(0)
    return W
def graph(kind):
    n=(400,800)[kind]; sig=(1.,.01)[kind]; rng=np.random.default_rng((407,711)[kind]); edges=[]; weights=[]
    if kind:
        h=(n-20)//2
        for off in (0,h):
            for _ in range(3):
                o=rng.permutation(h)+off; edges.extend(zip(o,np.roll(o,-1))); weights.extend([10.]*h)
        edges+=[(0,h)]+[(2*h+i,i) for i in range(20)]; weights += [.5]+[.3]*20
    else:
        edges=np.sort(np.vstack((np.c_[np.arange(n),np.roll(np.arange(n),-1)],rng.integers(n,size=(2*n,2)))),axis=1)
        edges=np.unique(edges[edges[:,0]!=edges[:,1]],axis=0); weights=np.ones(len(edges))
    a,b=np.array(edges).T; w=np.array(weights); M=sp.coo_matrix((np.r_[w,w,-w,-w],(np.r_[a,b,a,b],np.r_[a,b,b,a])),shape=(n,n)).tocsr()+sig*sp.eye(n,format='csr')
    if kind: E=np.array([(a,b) for a in range(60) for b in range(h,h+40) if (a,b)!=(0,h)]+[(2*h+i,a) for i in range(20) for a in range(60) if a!=i])
    else: blocked=set(map(tuple,edges)); E=np.array([(a,b) for a in range(96) for b in range(a+1,96) if (a,b) not in blocked])
    return M,E[rng.choice(len(E),N,False)],sig
def prepare(M,E,blocks=False):
    rng=np.random.default_rng(2026); Z=orth(M.shape[0],rng,True,blocks); C=sp.triu(M,k=1).tocoo(); U=orth(C.nnz+N,rng,False,blocks)
    F=np.zeros_like(Z); W=U[:C.nnz]*np.sqrt(-C.data[:,None]); np.add.at(F,C.row,W); np.add.at(F,C.col,-W); return Z,F,U[C.nnz:]
def cg(M,w):
    r=w.copy(); d=r.copy(); rr=w@w; stop=tol*np.sqrt(rr); p=len(tau); X=np.zeros((p,len(active))); P=np.tile(w[active],(p,1))
    z=np.ones(p); zp=z.copy(); live=np.ones(p,bool); h=np.zeros(p); ip=h.copy(); pn=np.full(p,rr)
    if rr==0: return X,h,0
    for it in range(2*len(w)+500):
        v=M@d+tau[0]*d; a=rr/(d@v); rn=r-a*v; rrn=rn@rn; be=rrn/rr; c=0 if it==0 else a*bp/ap; ids=np.flatnonzero(live)
        zn=z[ids]/(1+a*(tau[ids]-tau[0])+c*(1-z[ids]/zp[ids])); fac=zn/z[ids]; aj=a*fac; bj=be*fac**2
        X[ids]+=aj[:,None]*P[ids]; P[ids]=zn[:,None]*rn[active]+bj[:,None]*P[ids]
        h[ids]+=2*aj*ip[ids]+aj**2*pn[ids]; ip[ids]=bj*(ip[ids]+aj*pn[ids]); pn[ids]=zn**2*rrn+bj**2*pn[ids]
        zp[ids]=z[ids]; z[ids]=zn; live[ids]=abs(zn)*np.sqrt(rrn)>stop
        if np.sqrt(rrn)<=stop: return X,h,it+1
        d=rn+be*d; r=rn; rr=rrn; ap=a; bp=be
    raise RuntimeError('CG cap')
def run(B=1,naive=False):
    M=M0.copy(); F=F0.copy(); live=np.ones(N,bool); seq=[]; mv=ev=0; start=perf_counter(); h=np.zeros((len(tau),N)); t=h.copy()
    def bank():
        nonlocal mv
        xs=[]
        for w in np.c_[Z,F].T: x,_,steps=cg(M,w); mv+=steps; xs.append(x)
        return np.split(np.stack(xs,axis=-1),2,axis=-1)
    for r in range(k):
        ids=np.flatnonzero(live)
        if naive or r==0:
            X,Y=bank(); h[:,ids]=np.sum((X[:,es[ids,0]]-X[:,es[ids,1]])**2,axis=-1); t[:,ids]=np.sum((Y[:,es[ids,0]]-Y[:,es[ids,1]])**2,axis=-1); ev+=len(ids)
        vals=np.sum(omega[:,None]*h/(1+t+(sigma+tau[:,None])*h),axis=0); heap=[(-vals[e],int(e)) for e in ids]; heapify(heap); short=[heappop(heap)[1] for _ in range(B)]
        if naive: e=short[0]
        else:
            best=-np.inf
            for candidate in short:
                w=np.zeros(n); w[E[candidate]]=[1,-1]; u,hh,steps=cg(M,w); mv+=steps; gain=np.sum(omega*hh/(1+u[:,es[candidate,0]]-u[:,es[candidate,1]]))
                if gain>best: best=gain; e=candidate; champion=u
        seq.append(e); live[e]=False; a,b=E[e]; aa,bb=es[e]
        if not naive and r+1<k:
            u=champion; xi=X[:,aa]-X[:,bb]; eta=G[e]-Y[:,aa]+Y[:,bb]; den=1+u[:,aa]-u[:,bb]; c=(u[:,es[:,0]]-u[:,es[:,1]])/den[:,None]
            wx=np.einsum('pnq,pq->pn',X,xi); wy=np.einsum('pnq,pq->pn',Y,eta)
            h=np.maximum(0,h-2*c*(wx[:,es[:,0]]-wx[:,es[:,1]])+c*c*np.sum(xi*xi,axis=1)[:,None])
            t=np.maximum(0,t+2*c*(wy[:,es[:,0]]-wy[:,es[:,1]])+c*c*np.sum(eta*eta,axis=1)[:,None])
            X-=u[:,:,None]*xi[:,None,:]/den[:,None,None]; Y+=u[:,:,None]*eta[:,None,:]/den[:,None,None]
        F[a]+=G[e]; F[b]-=G[e]; M+=sp.csr_matrix(([1.,1.,-1.,-1.],([a,b,a,b],[a,b,b,a])),shape=M.shape)
    return seq,mv,ev,perf_counter()-start
def reference(s):
    M=M0.toarray(); live=np.ones(N,bool); seq=[]
    for _ in range(k):
        if s==0: vals=-np.diag(M)[E].sum(1); ids=np.array([np.argmax(np.where(live,vals,-np.inf))]); vals=vals[ids]
        else:
            lam,V=np.linalg.eigh(M); W=(V[E[:,0]]-V[E[:,1]])**2
            if s==1: vals=(W@(lam**-2))/(1+W@(lam**-1)); ids=np.array([np.argmax(np.where(live,vals,-np.inf))]); vals=vals[ids]
            else:
                top=lam[-1]+2; ts,ws=quad(s,128,np.sqrt(sigma*top)); D=1/(lam[:,None]+ts); g=((W@(D*D))/(1+W@D))@ws
                b=(np.sqrt(top/sigma)-1)/(np.sqrt(top/sigma)+1); rho=(1/b+np.sqrt(1/b**2-1))*128/129; R=(rho+1/rho)/2
                err=max(1e-10,2*(1+R)*(1+b)**2/(s*(rho-1)*(1-b*R)**2)*rho**-255)
                ids=np.flatnonzero(live&(g/(1-err)>=np.max(g[live]/(1+err)))); vals=[]
                for e in ids:
                    v=np.zeros(n); v[E[e]]=[1,-1]; vals.append(np.sum(lam**(-s))-np.sum(np.linalg.eigvalsh(M+np.outer(v,v))**(-s)))
        e=int(ids[np.argmax(vals)]); seq.append(e); live[e]=False; a,b=E[e]; M[a,a]+=1; M[b,b]+=1; M[a,b]-=1; M[b,a]-=1
    return seq
def truth(seq,s):
    A=M0.toarray()
    for e in seq: a,b=E[e]; A[a,a]+=1; A[b,b]+=1; A[a,b]-=1; A[b,a]-=1
    return np.sum(lam0**(-s))-np.sum(np.linalg.eigvalsh(A)**(-s))
for kind in (0,1):
    q=24; M0,E,sigma=graph(kind); n=M0.shape[0]; active=np.unique(E); es=np.searchsorted(active,E); Z,F0,G=prepare(M0,E); lam0=np.linalg.eigvalsh(M0.toarray())
    if kind: fixed={'Random':np.random.default_rng(2026).choice(N,k,False),'Degree':reference(0),'Exact-ER':reference(1)}; tau,omega=quad(1); fixed['Sketched-ER']=run()[0]
    for s in ((.5,) if not kind else (.25,.5)):
        tau,omega=quad(s); ref=reference(s); refgain=truth(ref,s); baseline=run(naive=True); saved={}
        print('BASE',n,s,'exact',refgain,'SpMV',baseline[1],'seconds',baseline[3])
        for B in (1,3,5):
            ans=run(B); saved[B]=ans[0]; gain=truth(ans[0],s)
            if B==1: assert ans[0]==baseline[0]
            print('T1',n,s,q,'column',B,gain,100*gain/refgain,ans[1],ans[2],100*(1-ans[2]/(k*N-k*(k-1)/2)),ans[3],baseline[1]/ans[1],baseline[3]/ans[3])
        if kind:
            methods=dict(fixed); methods.update({'Inc-Lazy-B1':saved[1],'Inc-Lazy-Cert-B4':run(4)[0],'Exact-Greedy':ref})
            for name,seq in methods.items(): print('T2',s,name,truth(seq,s),'cross',sum(max(E[e])<n-20 for e in seq))
    if not kind:
        for q in (24,64):
            Z,F0,G=prepare(M0,E,True); base=run(naive=True); ans=run(5); gain=truth(ans[0],s)
            print('T1',n,s,q,'block',5,gain,100*gain/refgain,ans[1],ans[2],100*(1-ans[2]/(k*N-k*(k-1)/2)),ans[3],base[1]/ans[1],base[3]/ans[3])
print('total_seconds',perf_counter()-started)
