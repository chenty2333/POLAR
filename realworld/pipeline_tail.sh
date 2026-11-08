# --- reordered remainder (2026-11-08 01:50): timing alone on the GPU, then two concurrent streams ---
for g in as-Skitter com-LiveJournal; do run time --graphs $g --N 2000 --k 20 --s 0.5 --reps 3; done
(
for g in ca-HepPh ca-CondMat; do run effect --graphs $g --N 2000 --k 20 --random 3 --cross-only; run effect --graphs $g --N 2000 --k 20 --random 3 --cross-only --pool 2hop; done
for f in results/effect_email-Eu-core.json results/effect_Hamster.json results/effect_ca-GrQc.json \
         results/effect_email-Eu-core_2hop.json results/effect_Hamster_2hop.json results/effect_ca-GrQc_2hop.json; do dn $f; done
run effect --graphs ca-GrQc --N 2000 --k 100 --random 10 --pool 2hop --tag k100
for f in results/effect_ca-HepPh.json results/effect_ca-HepPh_2hop.json; do dn $f; done
) &
(
for g in email-Enron com-DBLP com-Amazon web-Google roadNet-PA com-Youtube roadNet-CA as-Skitter com-LiveJournal; do run effect --graphs $g --N 2000 --k 20 --random 3 --cross-only; done
) &
wait
for f in results/effect_ca-CondMat.json results/effect_ca-CondMat_2hop.json; do dn $f; done
echo "[$(date '+%F %T')] PIPELINE DONE"
