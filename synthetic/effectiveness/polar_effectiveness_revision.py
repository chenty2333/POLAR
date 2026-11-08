"""Reproduce POLAR degree-heterogeneity experiment and original control.
Run: python3 polar_effectiveness_revision.py
Requires NumPy/SciPy; pins one permitted CPU and uses one BLAS thread.
Outputs results.json and results.csv beside this script.
The exact fractional reference uses Theorem1 composite positive quadrature
with relative derivative bound <=1e-9, then eigvalsh on every contender.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
if hasattr(os,'sched_getaffinity'):
    allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{allowed[min(1,len(allowed)-1)]})
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

def make(seed=711,cw=.5,aw=.5):
 n=805;sig=.01;rng=np.random.default_rng(seed);h=390;edges=[];weights=[]
 for off in (0,h):
  for _ in range(3):
   o=rng.permutation(h)+off;edges.extend(zip(o,np.roll(o,-1)));weights.extend([10.]*h)
 edges += [(0,h)]+[(2*h+i,i) for i in range(20)];weights += [.5]+[.3]*20
 edges += [(800+i,800+(i+1)%5) for i in range(5)]+[(800+i,20+i) for i in range(5)];weights +=[cw]*5+[aw]*5
 a,z=np.array(edges).T;w=np.array(weights);M=sp.coo_matrix((np.r_[w,w,-w,-w],(np.r_[a,z,a,z],np.r_[a,z,z,a])),shape=(n,n)).tocsr()+sig*sp.eye(n,format='csr')
 pool=np.array([(a,z) for a in range(60) for z in range(h,h+40) if (a,z)!=(0,h)]+[(2*h+i,a) for i in range(20) for a in range(60) if a!=i])
 decoys=np.array([(800+i,800+j) for i in range(5) for j in range(i+1,5) if (j-i) not in (1,4)])
 E=np.vstack((pool[rng.choice(len(pool),3000-len(decoys),False)],decoys));rng.shuffle(E)
 return M,E,sig


def setup(seed=711,control=False):
    global q,k,N,M0,E,sigma,n,active,es,Z,F0,G,lam0
    q,k,N=24,15,3000
    M0,E,sigma=graph(1) if control else make(seed,.5,.5)
    n=M0.shape[0];active=np.unique(E);es=np.searchsorted(active,E)
    Z,F0,G=prepare(M0,E);lam0=np.linalg.eigvalsh(M0.toarray())

def categories(seq):
    return {'cross':int(sum(max(E[e])<780 for e in seq)),
            'leaf':int(sum(780<=max(E[e])<800 for e in seq)),
            'cycle':int(sum(max(E[e])>=800 for e in seq))}

def composite(s,a,z,eps=1e-9):
    r=int(np.ceil(np.log(256/eps)/np.log(4)));K=int(np.ceil(np.log2(z/a)));cs=np.sin(np.pi*s)/np.pi
    def rule(beta):
        i=np.arange(r,dtype=float);aa=np.empty(r);aa[0]=beta/(beta+2);i=i[1:]
        aa[1:]=beta*beta/((2*i+beta)*(2*i+beta+2))
        bb=2/(2*i+beta)*np.sqrt(i*i*(i+beta)**2/((2*i+beta-1)*(2*i+beta+1)))
        x,Q=np.linalg.eigh(np.diag(aa)+np.diag(bb,1)+np.diag(bb,-1))
        return (1+x)/2,Q[0]**2/(beta+1)
    lo,A=rule(-s);hi,B=rule(s-1);mid,W=np.polynomial.legendre.leggauss(r)
    ts=[a*lo];ws=[cs*a**(1-s)*A]
    for ell in range(K):
        L=a*2**ell;t=L*(3+mid)/2;ts.append(t);ws.append(cs*L/2*W*t**(-s))
    C=a*2**K;ts.append(C/hi);ws.append(cs*C**(1-s)*B/hi)
    return np.concatenate(ts),np.concatenate(ws),256*4.**(-r)

def exact_fractional(s):
    M=M0.toarray();live=np.ones(N,bool);seq=[];cert=[];beg=perf_counter()
    for r in range(k):
        lam,V=np.linalg.eigh(M);W=(V[E[:,0]]-V[E[:,1]])**2
        ts,ws,eps=composite(s,sigma,lam[-1]+2)
        D=1/(lam[:,None]+ts);scores=((W@(D*D))/(1+W@D))@ws
        eps=max(eps,1e-9)
        ids=np.flatnonzero(live&(scores/(1-eps)>=np.max(scores[live]/(1+eps))))
        vals=[];old=np.sum(lam**(-s))
        for e in ids:
            A=M.copy();a,z=E[e];A[a,a]+=1;A[z,z]+=1;A[a,z]-=1;A[z,a]-=1
            vals.append(old-np.sum(np.linalg.eigvalsh(A)**(-s)))
        e=int(ids[np.argmax(vals)]);seq.append(e);live[e]=False;a,z=E[e];M[a,a]+=1;M[z,z]+=1;M[a,z]-=1;M[z,a]-=1
        cert.append({'round':r+1,'p':len(ts),'derivative_relative_bound':float(256*4.**(-int(np.ceil(np.log(256/1e-9)/np.log(4))))),'screen_guard':eps,'eigenvalue_contenders':len(ids),'selected_quad_score':float(scores[e]),'selected_true_score':float(max(vals))})
    return seq,perf_counter()-beg,cert


if __name__=='__main__':
    import json,csv,platform
    from pathlib import Path
    output=Path(__file__).resolve().parent
    out={'construction':{'n_stress':805,'n_control':800,'core_sizes':[390,390],
        'cycles_per_core':3,'core_cycle_weight':10,'intercore_weight':.5,
        'leaves':20,'leaf_weight':.3,'local_cycle_nodes':5,
        'local_cycle_weight':.5,'local_anchor_weight':.5,
        'local_anchor_vertices':[20,21,22,23,24],
        'sigma':.01,'N':3000,'q':24,'B':4,'p':20,'k':15,'tol':1e-7,
        'stress_seeds':[711,712,713],'control_seed':711,'projection_seed':2026,
        'candidate_rule':'Stress: sample2995 intercore/leaf links uniformly, add all5 missing C5 chords, shuffle; same pool for all methods. Control: original3000-link pool.'},
        'environment':{'cpu_affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else [],'python':platform.python_version(),'numpy':np.__version__},'rows':[],'references':[]}
    for case,seed in [('control',711),('stress',711),('stress',712),('stress',713)]:
        setup(seed,case=='control');degree=reference(0);er=reference(1)
        tau,omega=quad(1);sketched=run(1)
        fixed={'Random':np.random.default_rng(2026).choice(N,k,False).tolist(),
               'Degree':degree,'Exact-ER':er,'Sketched-ER':sketched[0]}
        for s in (.25,.5):
            exact,seconds,cert=exact_fractional(s);tau,omega=quad(s);ans=run(4)
            sequences={**fixed,'POLAR':ans[0],'Exact-Greedy':exact};refgain=truth(exact,s)
            for method,seq in sequences.items():
                gain=float(truth(seq,s));row={'case':case,'seed':seed,'s':s,'method':method,
                'gain':gain,'recovery_percent':100*gain/refgain,'counts':categories(seq),
                'sequence':[int(x) for x in seq],'selected_edges':E[seq].tolist()}
                if method=='POLAR':row.update({'SpMV':ans[1],'norm_evaluations':ans[2],'seconds':ans[3]})
                out['rows'].append(row)
            out['references'].append({'case':case,'seed':seed,'s':s,'seconds':seconds,'certificate':cert})
            (output/'results.json').write_text(json.dumps(out,indent=2))
            print(case,seed,s,'POLAR',truth(ans[0],s),'Exact',refgain,flush=True)
    out['summary']=[]
    for case in ('control','stress'):
        for s in (.25,.5):
            for method in ('Random','Degree','Exact-ER','Sketched-ER','POLAR','Exact-Greedy'):
                rows=[r for r in out['rows'] if r['case']==case and r['s']==s and r['method']==method]
                g=np.array([r['gain'] for r in rows]);out['summary'].append({'case':case,'s':s,'method':method,
                    'mean_gain':float(g.mean()),'std_gain':float(g.std(ddof=1)) if len(g)>1 else 0.,
                    'min_gain':float(g.min()),'max_gain':float(g.max()),
                    'mean_recovery':float(np.mean([r['recovery_percent'] for r in rows])),
                    'counts_by_seed':[r['counts'] for r in rows]})
    out['comparisons']=[]
    for exponent in (.25,.5):
        means={r['method']:r['mean_gain'] for r in out['summary'] if r['case']=='stress' and r['s']==exponent}
        out['comparisons'].append({'s':exponent,'POLAR_percent_over':{name:100*(means['POLAR']/means[name]-1) for name in ('Degree','Exact-ER','Sketched-ER')},'recovery_ratio_of_means':100*means['POLAR']/means['Exact-Greedy']})
    (output/'results.json').write_text(json.dumps(out,indent=2))
    with open(output/'results.csv','w') as f:
        wr=csv.DictWriter(f,fieldnames=['case','seed','s','method','gain','recovery_percent','cross','leaf','cycle']);wr.writeheader()
        for row in out['rows']:
            wr.writerow({**{k:row[k] for k in ['case','seed','s','method','gain','recovery_percent']},**row['counts']})
    print(json.dumps(out['summary'],indent=2))
