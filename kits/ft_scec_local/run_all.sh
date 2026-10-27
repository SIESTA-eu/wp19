#!/bin/bash
# Six jobs (3 WP17 models x 2 label sets), one per GPU, in parallel.
# Usage: ./run_all.sh [first GPU index, default 0]   -- needs 6 free GPUs from there;
# with fewer, run the lines of JOBS by hand with --gpu.
cd "$(dirname "$0")"; G=${1:-0}; mkdir -p logs
JOBS=("dslim/bert-base-NER conll4" "dbmdz/bert-large-cased-finetuned-conll03-english conll4"
      "Jean-Baptiste/roberta-large-ner-english conll4" "dslim/bert-base-NER native"
      "dbmdz/bert-large-cased-finetuned-conll03-english native" "Jean-Baptiste/roberta-large-ner-english native")
for k in "${!JOBS[@]}"; do
  set -- ${JOBS[$k]}
  .venv/bin/python run.py --model "$1" --labels "$2" --gpu $((G + k)) > "logs/${1//\//__}__$2.log" 2>&1 &
done
wait
grep -h "ALL DONE" logs/*.log
echo "finished: $(grep -l 'ALL DONE' logs/*.log | wc -l) of 6"
