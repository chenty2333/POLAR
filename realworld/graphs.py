"""Load SNAP / KONECT edge lists into a cached, relabelled, undirected simple edge array.
Preprocessing (same as the FastFrac companion): retain all endpoint vertices, drop edge directions,
duplicates and self-loops. Candidate pools: uniform non-edges (FastFrac rule) with a fixed seed."""
import gzip, tarfile, io, hashlib, json
from pathlib import Path
import numpy as np

import os
_DATA = Path(os.environ.get('POLAR_DATA', Path(__file__).resolve().parent / 'data'))
RAW = _DATA / 'raw'; NPZ = _DATA / 'npz'
SNAP = {
    'email-Eu-core': 'email-Eu-core.txt.gz', 'ca-GrQc': 'ca-GrQc.txt.gz', 'ca-HepPh': 'ca-HepPh.txt.gz',
    'ca-CondMat': 'ca-CondMat.txt.gz', 'email-Enron': 'email-Enron.txt.gz',
    'com-DBLP': 'com-dblp.ungraph.txt.gz', 'com-Amazon': 'com-amazon.ungraph.txt.gz',
    'web-Google': 'web-Google.txt.gz', 'roadNet-PA': 'roadNet-PA.txt.gz',
    'com-Youtube': 'com-youtube.ungraph.txt.gz', 'as-Skitter': 'as-skitter.txt.gz',
    'roadNet-CA': 'roadNet-CA.txt.gz', 'com-LiveJournal': 'com-lj.ungraph.txt.gz',
}
KONECT = {
    'Zachary': 'download.tsv.ucidata-zachary.tar.bz2', 'Dolphins': 'download.tsv.dolphins.tar.bz2',
    'LesMis': 'download.tsv.moreno_lesmis.tar.bz2', 'Polbooks': 'download.tsv.dimacs10-polbooks.tar.bz2',
    'Hamster': 'download.tsv.petster-hamster-household.tar.bz2',
}


def _parse(text_bytes, comment):
    try:
        import pandas as pd
        df = pd.read_csv(io.BytesIO(text_bytes), comment=comment, sep=r'\s+', header=None,
                         usecols=[0, 1], dtype=np.int64, engine='c')
        return df.to_numpy()
    except ImportError:
        return np.loadtxt(io.BytesIO(text_bytes), comments=comment, usecols=(0, 1), dtype=np.int64)


def _raw_edges(name):
    if name in SNAP:
        path = RAW / SNAP[name]
        return _parse(gzip.open(path).read(), '#'), path
    path = RAW / KONECT[name]
    with tarfile.open(path, 'r:bz2') as tf:
        mem = next(m for m in tf.getmembers() if Path(m.name).name.startswith('out.'))
        return _parse(tf.extractfile(mem).read(), '%'), path


def load(name):
    NPZ.mkdir(parents=True, exist_ok=True); cache = NPZ / f'{name}.npz'
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return int(z['n']), z['edges'], json.loads(str(z['meta']))
    raw, path = _raw_edges(name)
    ids, inv = np.unique(raw.ravel(), return_inverse=True); e = inv.reshape(-1, 2)
    e = np.sort(e, axis=1); e = e[e[:, 0] != e[:, 1]]; e = np.unique(e, axis=0)
    meta = dict(name=name, source=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                n=int(len(ids)), m=int(len(e)),
                preprocessing='retain all endpoint vertices; undirect; deduplicate; remove self-loops')
    np.savez(cache, n=len(ids), edges=e.astype(np.int64), meta=json.dumps(meta))
    return len(ids), e.astype(np.int64), meta


def candidates(n, edges, N, seed):
    keys = np.sort(edges[:, 0] * n + edges[:, 1]); rng = np.random.default_rng(seed); C = set()
    while len(C) < N:
        u, v = sorted(map(int, rng.integers(n, size=2)))
        i = np.searchsorted(keys, u * n + v)
        if u != v and not (i < len(keys) and keys[i] == u * n + v):
            C.add((u, v))
    c = np.asarray(sorted(C), dtype=np.int64)
    return np.ascontiguousarray(c[rng.permutation(N)])


def all_nonedges(n, edges):
    present = set(map(tuple, edges.tolist()))
    return np.array([(u, v) for u in range(n) for v in range(u + 1, n) if (u, v) not in present], dtype=np.int64)


def candidates_2hop(n, edges, N, seed):
    """Friend-of-friend pool: missing pairs (u, w) joined by a path u-v-w, sampled by random 2-step walks
    from a uniformly random start vertex (the standard link-recommendation candidate source)."""
    import scipy.sparse as sp
    A = sp.coo_matrix((np.ones(2 * len(edges)), (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])),
                      shape=(n, n)).tocsr()
    ptr, nbr = A.indptr, A.indices; deg = np.diff(ptr)
    keys = np.sort(edges[:, 0] * n + edges[:, 1]); rng = np.random.default_rng(seed); C = set()
    starts = np.flatnonzero(deg > 0)
    while len(C) < N:
        u = int(starts[rng.integers(len(starts))]); v = int(nbr[ptr[u] + rng.integers(deg[u])])
        w = int(nbr[ptr[v] + rng.integers(deg[v])])
        a, b = min(u, w), max(u, w); i = np.searchsorted(keys, a * n + b)
        if a != b and not (i < len(keys) and keys[i] == a * n + b):
            C.add((a, b))
    c = np.asarray(sorted(C), dtype=np.int64)
    return np.ascontiguousarray(c[rng.permutation(N)])
