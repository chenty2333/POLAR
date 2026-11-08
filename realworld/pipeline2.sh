#!/bin/bash
# Full POLAR real-network pipeline (defaults q=128, B=16), sequential on one GPU. Logs in logs/.
cd /root/autodl-tmp/polar; export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
PY=env/bin/python; mkdir -p logs
run(){ echo "[$(date '+%F %T')] START $*"; $PY -u run_real.py "$@" > "logs/$1_$3_$(date +%s).log" 2>&1 && echo "[$(date '+%F %T')] OK $*" || echo "[$(date '+%F %T')] FAIL $*"; }
dn(){ echo "[$(date '+%F %T')] START downstream $*"; $PY -u downstream.py "$@" > "logs/downstream_$(date +%s).log" 2>&1 && echo "[$(date '+%F %T')] OK downstream $*" || echo "[$(date '+%F %T')] FAIL downstream $*"; }
# pipeline2 (2026-11-08 05:20): LiveJournal timing alone (bank width capped at 16 for n > 3e6), then two streams.
run time --graphs com-LiveJournal --N 2000 --k 20 --s 0.5 --reps 1 --naive-limit 1800
(
for g in ca-HepPh ca-CondMat; do run effect --graphs $g --N 2000 --k 20 --random 3 --cross-only; run effect --graphs $g --N 2000 --k 20 --random 3 --cross-only --pool 2hop; done
for f in results/effect_email-Eu-core.json results/effect_Hamster.json results/effect_ca-GrQc.json \
         results/effect_email-Eu-core_2hop.json results/effect_Hamster_2hop.json results/effect_ca-GrQc_2hop.json; do dn $f; done
run effect --graphs ca-GrQc --N 2000 --k 100 --random 10 --pool 2hop --tag k100
for f in results/effect_ca-HepPh.json results/effect_ca-HepPh_2hop.json; do dn $f; done
) &
(
for g in email-Enron com-DBLP com-Amazon roadNet-PA roadNet-CA; do run effect --graphs $g --N 2000 --k 20 --random 3 --cross-only; done
for g in web-Google com-Youtube as-Skitter com-LiveJournal; do run effect --graphs $g --N 2000 --k 20 --random 3 --cross-only --s 0.5; done
) &
wait
for f in results/effect_ca-CondMat.json results/effect_ca-CondMat_2hop.json; do dn $f; done
echo "[$(date '+%F %T')] PIPELINE DONE"
