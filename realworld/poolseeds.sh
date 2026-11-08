#!/bin/bash
# Candidate-pool robustness: two extra uniform pools (seed offsets 1, 2) on the medium graphs.
cd /root/autodl-tmp/polar; PY=env/bin/python
for ps in 1 2; do for g in email-Eu-core Hamster ca-GrQc; do
  echo "[$(date '+%F %T')] START effect $g ps$ps"
  $PY -u run_real.py effect --graphs $g --N 2000 --k 20 --random 10 --pool-seed $ps > logs/effect_${g}_ps${ps}.log 2>&1 \
    && echo "[$(date '+%F %T')] OK effect $g ps$ps" || echo "[$(date '+%F %T')] FAIL effect $g ps$ps"
done; done
echo "[$(date '+%F %T')] POOLSEEDS DONE"
