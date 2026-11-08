#!/bin/bash
cd /root/autodl-tmp/polar; export OPENBLAS_NUM_THREADS=8
for m in grid round sigma; do for g in email-Eu-core Hamster ca-GrQc; do echo "[$(date "+%F %T")] START $m $g"; env/bin/python -u diag2.py $m $g > logs/diag_${m}_${g}.log 2>&1 && echo "[$(date "+%F %T")] OK $m $g" || echo "[$(date "+%F %T")] FAIL $m $g"; done; done
echo "[$(date "+%F %T")] DIAG DONE"
