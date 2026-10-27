#!/bin/bash
# Step 1: corrected SIESTA without the application-number pattern on all test sets.
# Step 2: WP17 configuration (CECILIA-fine-tuned checkpoints) on the remaining corpora.
# Step 3: owner-defined application-number type on TAB.
# Run from WP17/eval. Logs in logs/.
cd "$(dirname "$0")/.."
export SIESTA_DEVICE=0
FIXC="$PWD/siesta_aad6c5b_fixC/Anonimizacion"
for c in scec cecilia cyner tab cyberpii casie anonlfi_cases; do
  EVAL_DATASET=$c SIESTA_REPO="$FIXC" .venv/bin/python run_siesta.py aad-fixC > logs/aad-fixC_$c.log 2>&1 || echo "FAILED fixC $c"
  echo "fixC $c: $(tail -1 logs/aad-fixC_$c.log)"
done
.venv/bin/python analysis_siesta_fixes/predict_ft_cecilia.py tab cyberpii casie anonlfi_cases > logs/ft_cecilia_more.log 2>&1 || echo "FAILED ft-cecilia"
echo "ft-cecilia: $(grep -c wrote logs/ft_cecilia_more.log) files written"
EVAL_DATASET=tab SIESTA_REPO="$FIXC" .venv/bin/python analysis_siesta_fixes/run_custom_type_tab.py > logs/aad-fixC-appno_tab.log 2>&1 || echo "FAILED appno"
echo "appno: $(tail -1 logs/aad-fixC-appno_tab.log)"
