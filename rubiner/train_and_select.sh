#!/bin/bash
# Fine-tune dslim/bert-base-NER on the S-CECILIA training seeds (train_general_ner/),
# once with conll4 labels and once with SIESTA's entity types, then run the four
# RubiNER variants on S-CECILIA. The choice is made afterwards on the 4 validation
# seeds only (rule in run_rubiner.py); the variant kept is aad-fixC-ft-siesta-rep.
# Training data: the S-CECILIA training seeds written by
# experiments/tables_2_3_detection/export_ft_kit.py. Logs in logs/.
cd "$(dirname "$0")/.."
mkdir -p logs
[ -d rubiner/train_general_ner/data ] || cp -r experiments/tables_2_3_detection/d17_on_scecilia_kit/data rubiner/train_general_ner/data
for L in conll4 siesta; do
  (cd rubiner/train_general_ner && ../../.venv/bin/python run.py --model dslim/bert-base-NER --labels $L --gpu 0) \
    > logs/train_general_ner_$L.log 2>&1 || echo "FAILED training $L"
  echo "train $L: $(tail -1 logs/train_general_ner_$L.log)"
done
export SIESTA_DEVICE=0
for v in aad-fixC-ft-conll4-rep aad-fixC-ft-conll4-add aad-fixC-ft-siesta-rep aad-fixC-ft-siesta-add; do
  EVAL_DATASET=scec .venv/bin/python rubiner/run_rubiner.py $v > logs/${v}_scec.log 2>&1 || echo "FAILED $v"
  echo "$v scec: $(tail -1 logs/${v}_scec.log)"
done
