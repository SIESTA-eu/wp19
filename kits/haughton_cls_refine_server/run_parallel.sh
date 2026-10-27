#!/bin/bash
# Spreads the jobs over N GPUs, one process per GPU (job i goes to GPU i mod N).
# Usage: ./run_parallel.sh [N_GPUS] [extra run.py options]   e.g. ./run_parallel.sh 8 --schemes 10c
cd "$(dirname "$0")"
. .venv/bin/activate
N=${1:-$(nvidia-smi -L | wc -l)}; shift || true; EXTRA="$*"   # e.g. --schemes 10c
mkdir -p logs
for k in $(seq 0 $((N - 1))); do
  CUDA_VISIBLE_DEVICES=$k python run.py --shard $k/$N $EXTRA > logs/gpu$k.log 2>&1 &
  echo "GPU $k: shard $k/$N (logs/gpu$k.log)"
done
wait
echo "All shards finished. Check: grep -L 'ALL DONE' logs/gpu*.log  (lists any that did not finish)"
