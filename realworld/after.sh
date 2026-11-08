#!/bin/bash
# Runs after pipeline.sh finishes: ca-GrQc 2-hop with k=100 (k=20 only reaches the tied pendant pairs).
cd /root/autodl-tmp/polar; PY=env/bin/python
until grep -q "PIPELINE DONE" pipeline.log; do sleep 60; done
echo "[$(date '+%F %T')] START effect ca-GrQc 2hop k100"
$PY -u run_real.py effect --graphs ca-GrQc --N 2000 --k 100 --random 10 --pool 2hop --tag k100 > logs/effect_ca-GrQc_2hop_k100.log 2>&1 \
  && echo "[$(date '+%F %T')] OK effect ca-GrQc 2hop k100" || echo "[$(date '+%F %T')] FAIL effect ca-GrQc 2hop k100"
echo "[$(date '+%F %T')] AFTER DONE"
