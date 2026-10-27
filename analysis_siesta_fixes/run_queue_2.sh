#!/bin/bash
# Phase C, part 1: fine-tune dslim/bert-base-NER on the S-CECILIA training seeds with the
# kit's recipe (kits/ft_scec_local), once with conll4 labels and once with SIESTA's entity
# types, then run the four SIESTA variants on S-CECILIA. The choice is made afterwards on
# the 4 validation seeds only (rule in run_siesta_ft.py).
# Run from WP17/eval.
cd "$(dirname "$0")/.."
for L in conll4 siesta; do
  (cd kits/ft_scec_local && ../../.venv/bin/python run.py --model dslim/bert-base-NER --labels $L --gpu 0) \
    > logs/ft_scec_local_dslim_$L.log 2>&1 || echo "FAILED training $L"
  echo "train $L: $(tail -1 logs/ft_scec_local_dslim_$L.log)"
done
export SIESTA_DEVICE=0
FIXC="$PWD/siesta_aad6c5b_fixC/Anonimizacion"
for v in aad-fixC-ft-conll4-rep aad-fixC-ft-conll4-add aad-fixC-ft-siesta-rep aad-fixC-ft-siesta-add; do
  EVAL_DATASET=scec SIESTA_REPO="$FIXC" .venv/bin/python analysis_siesta_fixes/run_siesta_ft.py $v > logs/${v}_scec.log 2>&1 || echo "FAILED $v"
  echo "$v scec: $(tail -1 logs/${v}_scec.log)"
done
