import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import scipy.sparse as sp
import heapq
from time import perf_counter
p,q,k,N,nc,tol,sigma=20,24,15,3000,96,1e-7,1.0
def quad(s,r):
    i=np.arange(r); a=(1-2*s)/(4*i*i-1.); i=np.arange(1,r); b=np.sqrt((i-s)*(i+s-1))/(2*i-1)
    b[0]*=np.sqrt(2); y,V=np.linalg.eigh(np.diag(a)+np.diag(b,1)+np.diag(b,-1))
    return ((1-y)/(1+y))[::-1], (2*V[0]**2/(1+y))[::-1]
spmvs=0; started=perf_counter()
def data(n):
    rng=np.random.default_rng(7+n)
    edge=np.sort(np.vstack((np.c_[np.arange(n),np.roll(np.arange(n),-1)],rng.integers(n,size=(2*n,2)))),axis=1)
    edge=np.unique(edge[edge[:,0]!=edge[:,1]],axis=0); a,b=edge.T
    M=sp.diags(np.bincount(edge.ravel(),minlength=n)+sigma)-sp.coo_matrix((np.ones(2*len(a)),(np.r_[a,b],np.r_[b,a])),shape=(n,n)).tocsr()
    blocked=set(map(tuple,edge)); E=np.array([(i,j) for i in range(nc) for j in range(i+1,nc) if (i,j) not in blocked])
    E=E[rng.choice(len(E),N,replace=False)]; Z=rng.normal(size=(n,q))/np.sqrt(q)
    U=rng.normal(size=(len(a),q))/np.sqrt(q); B=np.zeros((n,q))
    np.add.at(B,a,U); np.add.at(B,b,-U)
    G=rng.normal(size=(N,q))/np.sqrt(q)
    return M.tocsr(),E,Z,B,G
def cg(M,w,freeze):
    global spmvs
    r=w.copy(); d=r.copy(); rr=r@r; stop=tol*np.sqrt(rr)
    if rr==0: return np.zeros((p,nc))
    X=np.zeros((p,nc)); P=np.tile(w[:nc],(p,1)); z=np.ones(p); zp=z.copy(); live=np.ones(p,bool)
    for it in range(500):
        v=M@d+tau[0]*d; spmvs+=1
        a=rr/(d@v); rn=r-a*v; rrn=rn@rn; be=rrn/rr; c=0 if it==0 else a*bp/ap
        ids=np.flatnonzero(live); zn=z[ids]/(1+a*shift[ids]+c*(1-z[ids]/zp[ids])); fac=zn/z[ids]
        X[ids]+=a*fac[:,None]*P[ids]
        P[ids]=zn[:,None]*rn[:nc]+(be*fac**2)[:,None]*P[ids]
        zp[ids]=z[ids]; z[ids]=zn
        if freeze: live[ids]=np.abs(zn)*np.sqrt(rrn)>stop
        if np.sqrt(rrn)<=stop: break
        d=rn+be*d; r=rn; rr=rrn; ap=a; bp=be
    else: raise RuntimeError('CG iteration cap')
    return X
def bank(M,F,freeze):
    return np.stack([cg(M,F[:,i],freeze) for i in range(F.shape[1])],axis=-1)
def norms(X,E,ids):
    return np.linalg.norm(X[:,E[ids,0]]-X[:,E[ids,1]],axis=-1)
def score(H,L):
    h=H*H
    return np.sum(omega[:,None]*h/(1+L*L+(sigma+tau[:,None])*h),axis=0)
def run(inputs,mode):
    global spmvs
    M,E,Z,B,G=inputs; M=M.copy(); B=B.copy(); live=np.ones(N,bool); seq=[]; count=0; spmvs=0; start=perf_counter()
    if mode:
        X,Y=np.split(bank(M,np.c_[Z,B],mode==2),2,axis=-1)
        if mode==2: H=norms(X,E,np.arange(N)); L=norms(Y,E,np.arange(N)); stamp=np.zeros(N,int); count=N
    for r in range(k):
        ids=np.flatnonzero(live)
        if mode==0: X,Y=np.split(bank(M,np.c_[Z,B],False),2,axis=-1)
        if mode<2:
            vals=score(norms(X,E,ids),norms(Y,E,ids)); e=ids[np.argmax(vals)]; count+=len(ids)
        else:
            vals=score(H[:,ids],L[:,ids])*(1+1e-12); heap=[(-v,int(e)) for v,e in zip(vals,ids)]; heapq.heapify(heap)
            while True:
                _,e=heapq.heappop(heap)
                if stamp[e]==r: break
                H[:,e:e+1]=norms(X,E,np.array([e])); L[:,e:e+1]=norms(Y,E,np.array([e])); stamp[e]=r; count+=1
                heapq.heappush(heap,(-score(H[:,e:e+1],L[:,e:e+1])[0],e))
        seq.append(e); live[e]=False; a,b=E[e]; w=np.zeros(M.shape[0]); w[a]=1;w[b]=-1
        if mode and r+1<k:
            U=cg(M,w,mode==2); vx=X[:,a]-X[:,b]; vy=G[e]-Y[:,a]+Y[:,b]; den=1+U[:,a]-U[:,b]; cross=np.abs(U[:,E[:,0]]-U[:,E[:,1]])/den[:,None]
            if mode==2:
                H+=cross*np.linalg.norm(vx,axis=1)[:,None]; L=np.maximum(0,L-cross*np.linalg.norm(vy,axis=1)[:,None])
            X-=U[:,:,None]*vx[:,None,:]/den[:,None,None]; Y+=U[:,:,None]*vy[:,None,:]/den[:,None,None]
        B[a]+=G[e]; B[b]-=G[e]
        M+=sp.csr_matrix((np.array([1.,1.,-1.,-1.]),([a,b,a,b],[a,b,b,a])),shape=M.shape)
    return seq,spmvs,count,perf_counter()-start
def reference(M0,E,s):
    M=M0.toarray(); left=np.ones(N,bool); seq=[]; checks=0; rows=np.arange(N)
    for r in range(k):
        if s<=0:
            g=(2*(s<0)-1)*np.diag(M)[E].sum(1)
            ids=np.array([np.argmax(np.where(left,g,-np.inf))]); vals=g[ids]
        elif s==1:
            R=np.linalg.inv(M); X=R[:,E[:,0]]-R[:,E[:,1]]
            g=np.sum(X*X,axis=0)/(1+X[E[:,0],rows]-X[E[:,1],rows])
            ids=np.array([np.argmax(np.where(left,g,-np.inf))]); vals=g[ids]
        else:
            lam,V=np.linalg.eigh(M); B=2*np.diag(M).max()-sigma+2; c=np.sqrt(sigma*B)
            t,w=quad(s,48); t*=c; w*=c**(1-s)
            X=(V[E[:,0]]-V[E[:,1]])**2; D=1/(lam[:,None]+t)
            g=np.sum(w*(X@(D*D))/(1+X@D),axis=1)
            beta=(np.sqrt(B/sigma)-1)/(np.sqrt(B/sigma)+1)
            eps=max(1e-10,2*(1+beta)**2*beta**95*(96-95*beta)/(1-beta)**2)
            ids=np.flatnonzero(left & (g/(1-eps)>=np.max(np.where(left,g/(1+eps),-np.inf))))
            vals=[]
            for e in ids:
                b=np.zeros(len(M)); b[E[e]]=[1,-1]; vals.append(np.sum(lam**(-s))-np.sum(np.linalg.eigvalsh(M+np.outer(b,b))**(-s)))
            checks+=len(ids)
        e=int(ids[np.argmax(vals)]); seq.append(e); left[e]=False; b=np.zeros(len(M)); b[E[e]]=[1,-1]; M+=np.outer(b,b)
    return seq,checks
def truth(M,E,seq,s):
    A=M.toarray(); base=np.sum(np.linalg.eigvalsh(A)**(-s))
    for e in seq:
        b=np.zeros(len(A)); b[E[e]]=[1,-1]; A+=np.outer(b,b)
    return base-np.sum(np.linalg.eigvalsh(A)**(-s))
tau,omega=quad(.5,p); shift=tau-tau[0]
for n in (400,1500,4000):
    inputs=data(n); ref=None
    for mode,name in enumerate(('Naive-ShiftedCG','Inc-Full','Inc-Lazy')):
        seq,smv,ev,seconds=run(inputs,mode)
        if ref is None: ref=seq; time0=seconds; mv0=smv
        assert seq==ref, 'selected sequence mismatch'
        print('A',n,name,smv,ev,round(100*(1-ev/(k*N-k*(k-1)/2)),2),round(seconds,4),round(time0/seconds,3),round(mv0/smv,3),sum(a==b for a,b in zip(seq,ref)))
inputs=data(400); M,E=inputs[:2]; ds=(M.diagonal()-sigma)[E].sum(1)
fixed={'Random':np.random.default_rng(2026).choice(N,k,False),'Degree-low':reference(M,E,0)[0],'Degree-high':reference(M,E,-1)[0],'Standard-ER':reference(M,E,1)[0]}
for s in (.3,.5):
    tau,omega=quad(s,p); shift=tau-tau[0]; methods=dict(fixed)
    methods['Inc-Lazy']=run(inputs,2)[0]; methods['Exact-Greedy'],checks=reference(M,E,s)
    print('B',s,'certified survivor eigencomputations',checks)
    for name,seq in methods.items(): print(name,truth(M,E,seq,s))
print('total_seconds',round(perf_counter()-started,3))
