"""POLAR sparse scalability extension; deterministic inputs and raw timings.

One CPU, one BLAS thread. Timing excludes topology/sketch generation but
includes initial solves, scoring, sparse rank-one graph additions and updates.
Every mode uses the same sketches and candidate set within a configuration.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
CPU=min(os.sched_getaffinity(0)); os.sched_setaffinity(0,{CPU})
import json, platform, sys, heapq, gc
from pathlib import Path
from time import perf_counter
import numpy as np
import scipy
import scipy.sparse as sp
p,q,k,N,tol,sigma=20,24,15,3000,1e-7,1.0
OUT=Path(__file__).resolve().parent
def quad(s,r):
    i=np.arange(r); a=(1-2*s)/(4*i*i-1.)
    i=np.arange(1,r); b=np.sqrt((i-s)*(i+s-1))/(2*i-1); b[0]*=np.sqrt(2)
    y,V=np.linalg.eigh(np.diag(a)+np.diag(b,1)+np.diag(b,-1))
    return ((1-y)/(1+y))[::-1], (2*V[0]**2/(1+y))[::-1]
tau,omega=quad(.5,p); shift=tau-tau[0]
spmvs=0; active=None
def topology(n,kind,rng):
    if kind=='ring-random':
        edge=np.sort(np.vstack((np.c_[np.arange(n),np.roll(np.arange(n),-1)],rng.integers(n,size=(2*n,2)))),axis=1)
        return np.unique(edge[edge[:,0]!=edge[:,1]],axis=0)
    if kind=='small-world':
        ed={(min(i,(i+d)%n),max(i,(i+d)%n)) for i in range(n) for d in (1,2,3)}
        for i in range(n):
            for d in (1,2,3):
                old=(min(i,(i+d)%n),max(i,(i+d)%n))
                if rng.random()<.1:
                    ed.remove(old)
                    while True:
                        j=int(rng.integers(n)); new=(min(i,j),max(i,j))
                        if j!=i and new not in ed: break
                    ed.add(new)
        return np.array(sorted(ed),dtype=int)
    if kind=='preferential':
        ed=[(i,j) for i in range(4) for j in range(i+1,4)]
        urn=[v for e in ed for v in e]
        for v in range(4,n):
            targets=set()
            while len(targets)<3: targets.add(urn[int(rng.integers(len(urn)))])
            for u in sorted(targets): ed.append((u,v)); urn.extend((u,v))
        return np.array(ed,dtype=int)
    raise ValueError(kind)
def data(n,kind,support):
    rng=np.random.default_rng(7+n)
    edge=topology(n,kind,rng); a,b=edge.T
    M=sp.diags(np.bincount(edge.ravel(),minlength=n)+sigma)-sp.coo_matrix((np.ones(2*len(a)),(np.r_[a,b],np.r_[b,a])),shape=(n,n)).tocsr()
    blocked=set(map(tuple,edge))
    if support=='local':
        E=np.array([(i,j) for i in range(96) for j in range(i+1,96) if (i,j) not in blocked])
        E=E[rng.choice(len(E),N,replace=False)]
    else:
        candidates=set()
        while len(candidates)<N:
            i,j=sorted(map(int,rng.integers(n,size=2)))
            if i!=j and (i,j) not in blocked: candidates.add((i,j))
        E=np.array(sorted(candidates),dtype=int)
    vertices=np.unique(E); es=np.searchsorted(vertices,E)
    Z=rng.normal(size=(n,q))/np.sqrt(q)
    U=rng.normal(size=(len(a),q))/np.sqrt(q); F=np.zeros((n,q))
    np.add.at(F,a,U); np.add.at(F,b,-U)
    G=rng.normal(size=(N,q))/np.sqrt(q)
    return (M.tocsr(),E,es,Z,F,G,vertices),len(edge)
def cg(M,w,freeze):
    global spmvs
    r=w.copy(); d=r.copy(); rr=r@r; stop=tol*np.sqrt(rr); nc=len(active)
    if rr==0: return np.zeros((p,nc))
    X=np.zeros((p,nc)); P=np.tile(w[active],(p,1)); z=np.ones(p); zp=z.copy(); live=np.ones(p,bool)
    for it in range(2000):
        v=M@d+tau[0]*d; spmvs+=1
        a=rr/(d@v); rn=r-a*v; rrn=rn@rn; be=rrn/rr; c=0 if it==0 else a*bp/ap
        ids=np.flatnonzero(live); zn=z[ids]/(1+a*shift[ids]+c*(1-z[ids]/zp[ids])); fac=zn/z[ids]
        X[ids]+=a*fac[:,None]*P[ids]
        P[ids]=zn[:,None]*rn[active]+(be*fac**2)[:,None]*P[ids]
        zp[ids]=z[ids]; z[ids]=zn
        if freeze: live[ids]=np.abs(zn)*np.sqrt(rrn)>stop
        if np.sqrt(rrn)<=stop: return X
        d=rn+be*d; r=rn; rr=rrn; ap=a; bp=be
    raise RuntimeError('CG iteration cap')
def bank(M,F,freeze):
    return np.stack([cg(M,F[:,i],freeze) for i in range(F.shape[1])],axis=-1)
def norms(X,es,ids):
    return np.linalg.norm(X[:,es[ids,0]]-X[:,es[ids,1]],axis=-1)
def score(H,L):
    h=H*H
    return np.sum(omega[:,None]*h/(1+L*L+(sigma+tau[:,None])*h),axis=0)
def run(inputs,mode):
    global spmvs,active
    M,E,es,Z,F,G,active=inputs; M=M.copy(); F=F.copy(); live=np.ones(N,bool); seq=[]; count=0; spmvs=0
    start=perf_counter()
    if mode:
        X,Y=np.split(bank(M,np.c_[Z,F],mode==2),2,axis=-1)
        if mode==2: H=norms(X,es,np.arange(N)); L=norms(Y,es,np.arange(N)); stamp=np.zeros(N,int); count=N
    for r in range(k):
        ids=np.flatnonzero(live)
        if mode==0: X,Y=np.split(bank(M,np.c_[Z,F],False),2,axis=-1)
        if mode<2:
            vals=score(norms(X,es,ids),norms(Y,es,ids)); e=ids[np.argmax(vals)]; count+=len(ids)
        else:
            vals=score(H[:,ids],L[:,ids])*(1+1e-12); heap=[(-v,int(e)) for v,e in zip(vals,ids)]; heapq.heapify(heap)
            while True:
                _,e=heapq.heappop(heap)
                if stamp[e]==r: break
                H[:,e:e+1]=norms(X,es,np.array([e])); L[:,e:e+1]=norms(Y,es,np.array([e])); stamp[e]=r; count+=1
                heapq.heappush(heap,(-score(H[:,e:e+1],L[:,e:e+1])[0],e))
        seq.append(int(e)); live[e]=False; a,b=E[e]; aa,bb=es[e]; w=np.zeros(M.shape[0]); w[a]=1;w[b]=-1
        if mode and r+1<k:
            U=cg(M,w,mode==2); vx=X[:,aa]-X[:,bb]; vy=G[e]-Y[:,aa]+Y[:,bb]; den=1+U[:,aa]-U[:,bb]
            cross=np.abs(U[:,es[:,0]]-U[:,es[:,1]])/den[:,None]
            if mode==2:
                H+=cross*np.linalg.norm(vx,axis=1)[:,None]; L=np.maximum(0,L-cross*np.linalg.norm(vy,axis=1)[:,None])
            X-=U[:,:,None]*vx[:,None,:]/den[:,None,None]; Y+=U[:,:,None]*vy[:,None,:]/den[:,None,None]
        F[a]+=G[e]; F[b]-=G[e]
        M+=sp.csr_matrix((np.array([1.,1.,-1.,-1.]),([a,b,a,b],[a,b,b,a])),shape=M.shape)
    return {'sequence':seq,'spmvs':spmvs,'norm_evaluations':count,'seconds':perf_counter()-start}
def main():
    configs=[(n,'ring-random','local') for n in (400,1500,4000,10000,20000)]
    configs += [(10000,'small-world','local'),(10000,'preferential','local'),(10000,'ring-random','global')]
    names=['Naive-ShiftedCG','Inc-Full','POLAR-Lazy']
    result={'environment':{'cpu_affinity':CPU,'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},'parameters':{'p':p,'q':q,'k':k,'N':N,'sigma':sigma,'s':.5,'tol':tol,'repetitions':3},'configurations':[]}
    cpuinfo=Path('/proc/cpuinfo').read_text(); result['environment']['cpu_model']=next(l.split(':',1)[1].strip() for l in cpuinfo.splitlines() if l.startswith('model name'))
    for n,kind,support in configs:
        inputs,m=data(n,kind,support); rows={name:[] for name in names}
        for repeat in range(3):
            for mode in np.roll(np.arange(3),-repeat):
                gc.collect(); out=run(inputs,int(mode)); rows[names[mode]].append(out)
                print(n,kind,support,repeat,names[mode],out['spmvs'],out['norm_evaluations'],round(out['seconds'],6),flush=True)
        ref=rows[names[0]][0]['sequence']; baseline=np.median([x['seconds'] for x in rows[names[0]]]); mv0=rows[names[0]][0]['spmvs']
        summaries=[]
        for name in names:
            r=rows[name]; smv=r[0]['spmvs']; ev=r[0]['norm_evaluations']; tm=float(np.median([x['seconds'] for x in r]))
            assert all(x['sequence']==ref for x in r), 'sequence mismatch'
            assert all(x['spmvs']==smv and x['norm_evaluations']==ev for x in r)
            summaries.append({'method':name,'seconds':tm,'spmvs':smv,'spmv_reduction':mv0/smv,'norm_evaluations':ev,'pruning_percent':100*(1-ev/(k*N-k*(k-1)/2)),'speedup':baseline/tm,'sequence_match':k})
        result['configurations'].append({'n':n,'m':m,'topology':kind,'support':support,'n_candidates_vertices':len(inputs[-1]),'seed':7+n,'raw':rows,'summary':summaries})
        (OUT/'results.json').write_text(json.dumps(result,indent=2))
        print('SUMMARY',json.dumps(result['configurations'][-1]['summary']),flush=True)
    print('DONE',flush=True)
if __name__=='__main__': main()
