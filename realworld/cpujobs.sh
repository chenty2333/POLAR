#!/bin/bash
# CPU-only exact references (12 LAPACK threads, nice 19) alongside the GPU pipeline.
cd /root/autodl-tmp/polar; export OPENBLAS_NUM_THREADS=12 OMP_NUM_THREADS=12; PY="nice -n 19 env/bin/python -u"
r(){ echo "[$(date '+%F %T')] START $*"; $PY exact_cpu.py "$@" > "logs/exactcpu_$1_$2_$(date +%s).log" 2>&1 && echo "[$(date '+%F %T')] OK $*" || echo "[$(date '+%F %T')] FAIL $*"; }
r ca-HepPh uniform
r ca-HepPh 2hop
r ca-CondMat uniform 2000 20 frac:0.5 er
r ca-CondMat 2hop 2000 20 frac:0.5
echo "[$(date "+%F %T")] CPU DONE"
