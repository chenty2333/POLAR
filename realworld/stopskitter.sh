#!/bin/bash
# Stop the as-Skitter timing process once two repetitions of both POLAR variants are saved.
cd /root/autodl-tmp/polar
until env/bin/python -c "import json,sys; r=json.load(open('results/time_as-Skitter.json'))['runs']; sys.exit(0 if len(r.get('POLAR(B=1)',[]))>=2 and len(r.get('POLAR(B=16)',[]))>=2 else 1)" 2>/dev/null; do sleep 20; done
for p in $(ps -eo pid,args | awk '/run_real[.]py time --graphs as-Skitte[r]/{print $1}'); do kill $p; echo "[$(date '+%F %T')] stopped as-Skitter timing after 2 reps (pid $p)"; done
