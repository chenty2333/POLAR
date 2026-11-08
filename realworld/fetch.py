"""Parallel HTTP range downloader for the SNAP/KONECT inputs.
Usage: python fetch.py OUTDIR URL[=SHA256] ...   (48 connections per file, 1 MiB chunks)."""
import sys, os, hashlib, urllib.request, concurrent.futures as cf, time
CHUNK=1<<20; CONN=48
def size(url):
    with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=60) as r:
        return int(r.headers['Content-Length'])
def part(url,path,lo,hi):
    for attempt in range(20):
        try:
            req=urllib.request.Request(url,headers={'Range':f'bytes={lo}-{hi}'})
            with urllib.request.urlopen(req,timeout=120) as r: data=r.read()
            if len(data)!=hi-lo+1: raise IOError('short read')
            with open(path,'r+b') as f: f.seek(lo); f.write(data)
            return
        except Exception as e: time.sleep(2+attempt)
    raise RuntimeError(f'failed {url} {lo}-{hi}')
def fetch(out,spec):
    url,_,want=spec.partition('='); path=os.path.join(out,url.rsplit('/',1)[1])
    if want and os.path.exists(path) and hashlib.sha256(open(path,'rb').read()).hexdigest()==want:
        print('cached',path,flush=True); return
    n=size(url); t=time.time(); tmp=path+'.part'
    with open(tmp,'wb') as f: f.truncate(n)
    with cf.ThreadPoolExecutor(CONN) as ex:
        list(ex.map(lambda lo: part(url,tmp,lo,min(n,lo+CHUNK)-1),range(0,n,CHUNK)))
    got=hashlib.sha256(open(tmp,'rb').read()).hexdigest()
    if want and got!=want: raise RuntimeError(f'sha mismatch {url} {got}')
    os.replace(tmp,path); print(f'done {path} {n} bytes {time.time()-t:.0f}s sha256={got}',flush=True)
if __name__=='__main__':
    os.makedirs(sys.argv[1],exist_ok=True)
    for spec in sys.argv[2:]: fetch(sys.argv[1],spec)
    print('ALLDONE',flush=True)
