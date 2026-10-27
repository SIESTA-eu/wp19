# RubiNER: evaluation code for EOSC SIESTA Deliverable D19.4

Code behind the results of *D19.4 Anonymising Cyber Incident Reports: Text
Anonymisation Tool Evaluation*: the RubiNER ensemble NER model used by the
SIESTA text anonymisation service, and the scripts that evaluate it against
the Deliverable 17 (D17) models.

The fine-tuned weights of RubiNER's general NER component are published at
[GVIS-ULE/rubiner](https://huggingface.co/GVIS-ULE/rubiner) and downloaded
automatically (pinned to the evaluated revision).

## What produced which result

| D19.4 | Scripts |
|---|---|
| RubiNER (Section 2) | `siesta_aad6c5b_fixC/Anonimizacion/` (the anonymiser evaluated), `run_siesta.py`, `analysis_siesta_fixes/run_siesta_ft.py` (adds the fine-tuned general NER) |
| Fine-tuning of the general NER (Appendix 1) | `kits/ft_scec_local/run.py --labels siesta`, `analysis_siesta_fixes/run_queue_2.sh` |
| GLiNER2 thresholds | `gliner_cache.py`, `gliner_exp.py deployed` (reproduces `results_gliner_base_tuned.json`) |
| Cyber-model threshold (0.7) | `choose_threshold.py` |
| Splits | `split.py` (`split.json`, S-CECILIA seed-grouped), `corpora.py` (`split_cecilia.json`, D17's own split) |
| Metrics (Table 1) | `score.py`, `analysis_siesta_fixes/score_variants.py` |
| Table 2: D17 configuration and D17 models as published | `run_wp17_zeroshot.py`, `finetune.py`, `analysis_siesta_fixes/predict_ft_cecilia.py` |
| Table 3: D17 models fine-tuned on S-CECILIA | `export_ft_kit.py`, `kits/ft_scec/`, `import_ft_kit.py` |
| Tables 2 and 3 (spreadsheet) | `analysis_siesta_fixes/build_rubiner_tables.py` |
| Table 4: incident classification | `analysis_siesta_fixes/refinement_stage3.py`, `kits/haughton_cls_refine_server/` (RoBERTa-base, 5 seeds), with `haughton_compare.py`, `scec_utility.py`, `cecilia_utility.py`, `analysis_siesta_fixes/utility_rubiner.py` |
| Table 5: SPY | `spy_benchmark.py`, `analysis_siesta_fixes/run_spy_variants.py` |
| Limitations: name bias | `name_confidence.py`, `template_name_bias.py`, `mehrabi_bias.py` |

Loaders: `gold.py` (S-CECILIA), `cecilia.py` (CECILIA-10C-900), `benchmarks.py` (CyNER).

## Setup

Python 3.10, one CUDA GPU for training (an 8 GB GPU is enough except for the
large models in Tables 3 and 4, which were run on a server GPU).

```bash
python3.10 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m spacy download en_core_web_md   # and _lg, _trf for Table 2
```

Scripts are run from the repository root. Docstrings that say "run from
WP17/eval" mean this directory.

## Data

No data is included. Place it as follows:

| Path | Corpus |
|---|---|
| `data/S-CECILIA.csv`, `data/S-CECILIA-NER-GOLD.csv`, `data/S-CECILIA-NER-GOLD.conll` | S-CECILIA and S-CECILIA-Gold (under review; to be released on acceptance) |
| `data_cecilia/cecilia100_fixed_final.csv` | CECILIA-10C-900, as used by D17 |
| `data_haughton/source/Loya_NER_Cib.xlsx` | CECILIA-10C-900 with manual de-identification (Haughton et al., 2026) |
| `third_party/CyNER/` | `git clone https://github.com/aiforsec/CyNER` (commit 37aff53) |
| `third_party/spy_src/repo/` | `git clone https://github.com/LogicZMaksimka/SPY_Dataset` (commit 5baabd6) |

`split.json` and `split_cecilia.json` hold document identifiers only and are
included, so the exact test sets can be used without re-running the splits.

## Reproducing

Prediction files go to `preds/` (S-CECILIA) or `preds_<corpus>/`; the corpus
is chosen with `EVAL_DATASET` (`scec`, `cecilia`, `cyner`).

```bash
export SIESTA_DEVICE=0        # GPU index; -1 for CPU

# RubiNER and the anonymiser without fine-tuning, on each corpus
for c in scec cecilia cyner; do
  EVAL_DATASET=$c .venv/bin/python run_siesta.py aad-fixC
  EVAL_DATASET=$c .venv/bin/python analysis_siesta_fixes/run_siesta_ft.py aad-fixC-ft-siesta-rep
done

# D17 models: as published (and spaCy), fine-tuned on CECILIA, fine-tuned on S-CECILIA
for c in scec cecilia cyner; do EVAL_DATASET=$c .venv/bin/python run_wp17_zeroshot.py; done
.venv/bin/python finetune.py
.venv/bin/python analysis_siesta_fixes/predict_ft_cecilia.py cyner
.venv/bin/python export_ft_kit.py      # then run kits/ft_scec (see its README)
.venv/bin/python import_ft_kit.py <results dir or .tar.gz>

# Tables 2 and 3
.venv/bin/python score.py scec test; .venv/bin/python score.py cecilia test; .venv/bin/python score.py cyner all
.venv/bin/python analysis_siesta_fixes/score_variants.py aad-fixC aad-fixC-ft-siesta-rep \
  ft-cecilia__dslim__bert-base-NER ft-cecilia__dbmdz__bert-large-cased-finetuned-conll03-english \
  ft-cecilia__Jean-Baptiste__roberta-large-ner-english
.venv/bin/python analysis_siesta_fixes/build_rubiner_tables.py     # RubiNER_comparison.xlsx

# Table 4
.venv/bin/python analysis_siesta_fixes/refinement_stage3.py anonymise
.venv/bin/python analysis_siesta_fixes/refinement_stage3.py classical
.venv/bin/python analysis_siesta_fixes/refinement_stage3.py kit    # then run kits/haughton_cls_refine_server
.venv/bin/python analysis_siesta_fixes/refinement_stage3.py report

# Table 5
.venv/bin/python spy_benchmark.py
.venv/bin/python analysis_siesta_fixes/run_spy_variants.py
.venv/bin/python analysis_siesta_fixes/run_spy_variants.py --report

# GLiNER2 thresholds
.venv/bin/python gliner_cache.py
.venv/bin/python gliner_exp.py deployed
```

To retrain the general NER instead of downloading it, run
`kits/ft_scec_local/run.py --model dslim/bert-base-NER --labels siesta --gpu 0`;
`run_siesta_ft.py` uses a local checkpoint when one exists.

## Notes

- RubiNER as evaluated is the anonymiser in `siesta_aad6c5b_fixC/` with its
  general NER replaced at run time by the fine-tuned model
  (`run_siesta_ft.py`); the anonymiser's own files are not edited.
- Deviations from D17's fine-tuning recipe (warm-up, learning rate and batch
  size for large models, windowing instead of truncation) are listed in
  `finetune.py`.
- S-CECILIA results use seed-grouped splits: the 247 test reports come from
  15 seed incidents not used for training, tuning or model selection.
