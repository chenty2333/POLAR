#!/bin/bash
cd /root/autodl-tmp/polar; export OPENBLAS_NUM_THREADS=12 OMP_NUM_THREADS=12; PY="nice -n 19 env/bin/python -u"
r(){ echo "[$(date '+%F %T')] START dcpu $*"; $PY downstream_cpu.py "$@" > "logs/dcpu_$1_$2_$(date +%s).log" 2>&1 && echo "[$(date '+%F %T')] OK dcpu $*" || echo "[$(date '+%F %T')] FAIL dcpu $*"; }
r ca-HepPh uniform
r ca-HepPh 2hop
r ca-CondMat uniform
echo "[$(date '+%F %T')] CPU2 DONE"
