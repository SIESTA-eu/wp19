# RubiNER: evaluation code for EOSC SIESTA Deliverable D19.4

Code behind the results of *D19.4 Anonymising Cyber Incident Reports: Text
Anonymisation Tool Evaluation*: the RubiNER ensemble NER model used by the
SIESTA text anonymisation service, and the scripts that evaluate it against
the Deliverable 17 (D17) models.

The fine-tuned weights of RubiNER's general NER component are published at
[GVIS-ULE/rubiner](https://huggingface.co/GVIS-ULE/rubiner) and downloaded
automatically, pinned to the evaluated revision.

## Layout

```
rubiner/                     RubiNER itself
  anonymiser/                the SIESTA anonymiser as evaluated (cyber NER, GLiNER2 types,
                             patterns, dates, merging)
  run_rubiner.py             RubiNER = the anonymiser with the fine-tuned general NER
  run_siesta.py              builds the anonymiser and runs it over a corpus
  train_general_ner/         fine-tuning of dslim/bert-base-NER on S-CECILIA (Appendix 1)
  train_and_select.sh        training, then the variant choice on validation seeds
  thresholds/                tuning of the GLiNER2 type thresholds
corpora/                     the three corpora and their splits
  corpus.py                  which corpus a script works on (EVAL_DATASET) and where
                             its predictions go
  s_cecilia.py, cecilia.py,  one loader per corpus: S-CECILIA-Gold, CECILIA-10C-900,
  cyner.py                   CyNER
  splits.py                  the train/test splits, fixed in split_s_cecilia.json
                             and split_cecilia.json
scoring/                     metrics of Table 1 (P/R/F1, exact, leak)
experiments/
  tables_2_3_detection/      RubiNER against the D17 models (Tables 2 and 3)
  table_4_classification/    incident classification after redaction (Table 4)
  table_5_spy/               medical and legal text (Table 5)
paths.py                     imported first by every script; puts these folders on sys.path
```

## Names in the code

"SIESTA" is the project, and in the code it means the SIESTA anonymisation
service whose anonymiser RubiNER is built on. In particular:

- `rubiner/run_siesta.py` builds that anonymiser and runs it over a corpus.
  Its variant `aad-fixC` is the pipeline **without** the fine-tuned general
  NER ("RubiNER without S-CECILIA fine-tuning" in Table 2); `run_rubiner.py`
  builds RubiNER on top of it. Its other variants are earlier experiments,
  not reported in D19.4.
- `aad-fixC` names that version of the anonymiser; `aad-fixC-ft-siesta-rep`
  is RubiNER: the general NER fine-tuned (`ft`) with SIESTA's entity types as
  labels (`siesta`), replacing (`rep`) the original one.
- The label set `siesta` (`--labels siesta`, `SCEC_TO_SIESTA`) is SIESTA's
  entity types; `conll4` and `native` are D17's four types and S-CECILIA's own.
- `SIESTA_DEVICE` (GPU index, -1 for CPU) and `SIESTA_REPO` (another copy of
  the anonymiser) are environment variables read by `run_siesta.py`.

## What produced which result

| D19.4 | Scripts |
|---|---|
| RubiNER (Section 2) | `rubiner/run_rubiner.py`, `rubiner/run_siesta.py`, `rubiner/anonymiser/` |
| Fine-tuning of the general NER, label mapping (Appendix 1) | `rubiner/train_general_ner/`, `rubiner/train_and_select.sh`; mapping in `experiments/tables_2_3_detection/export_ft_kit.py` (`SCEC_TO_SIESTA`) |
| GLiNER2 thresholds | `rubiner/thresholds/gliner_cache.py`, `gliner_exp.py deployed` (reproduces `results_gliner_base_tuned.json`) |
| Splits | `corpora/splits.py`: `split_s_cecilia.json` (S-CECILIA, seed-grouped), `split_cecilia.json` (D17's own split) |
| Metrics (Table 1) | `scoring/score.py`, `scoring/score_variants.py` |
| Table 2: D17 configuration, D17 models as published | `run_wp17_zeroshot.py`, `finetune.py`, `predict_ft_cecilia.py` |
| Table 3: D17 models fine-tuned on S-CECILIA | `export_ft_kit.py`, `d17_on_scecilia_kit/`, `import_ft_kit.py` |
| Tables 2 and 3 (spreadsheet) | `build_rubiner_tables.py` |
| Table 4 | `refinement_stage3.py` (anonymisation, XGBoost, intervals), `roberta_kit/` (RoBERTa-base, 5 seeds) |
| Table 5 | `run_spy_variants.py`, `spy_corpus.py` |

## Setup

Python 3.10. A CUDA GPU is needed for training; an 8 GB GPU is enough except
for the large models in Tables 3 and 4, which were run on a server GPU.

```bash
python3.10 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m spacy download en_core_web_md   # and _lg, _trf for Table 2
```

Run every script from the repository root.

## Data

No data is included. Place it as follows:

| Path | Corpus |
|---|---|
| `data/S-CECILIA.csv`, `data/S-CECILIA-NER-GOLD.csv`, `data/S-CECILIA-NER-GOLD.conll` | S-CECILIA and S-CECILIA-Gold (under review; to be released on acceptance) |
| `data_cecilia/cecilia100_fixed_final.csv` | CECILIA-10C-900, as used by D17 |
| `data_haughton/source/Loya_NER_Cib.xlsx` | CECILIA-10C-900 with manual de-identification (Haughton et al., 2026) Request here - **https://gvis.unileon.es/datasets-cecilia-10c-900-ner/** |
| `third_party/CyNER/` | `git clone https://github.com/aiforsec/CyNER` (commit 37aff53) |
| `third_party/spy_src/repo/` | `git clone https://github.com/LogicZMaksimka/SPY_Dataset` (commit 5baabd6) |

`corpora/split_s_cecilia.json` and `corpora/split_cecilia.json` hold
document identifiers only, so the exact test sets are fixed without
re-running the splits.

## Reproducing

Predictions go to `preds/` (S-CECILIA) or `preds_<corpus>/`; the corpus is
chosen with `EVAL_DATASET` (`scec`, `cecilia`, `cyner`). `SIESTA_DEVICE` is a
GPU index, or -1 for CPU.

```bash
export SIESTA_DEVICE=0
X=experiments/tables_2_3_detection

# RubiNER, and the same pipeline without fine-tuning
for c in scec cecilia cyner; do
  EVAL_DATASET=$c .venv/bin/python rubiner/run_siesta.py aad-fixC
  EVAL_DATASET=$c .venv/bin/python rubiner/run_rubiner.py aad-fixC-ft-siesta-rep
done

# D17 models: as published (and spaCy), fine-tuned on CECILIA, fine-tuned on S-CECILIA
for c in scec cecilia cyner; do EVAL_DATASET=$c .venv/bin/python $X/run_wp17_zeroshot.py; done
.venv/bin/python $X/finetune.py
.venv/bin/python $X/predict_ft_cecilia.py cyner
.venv/bin/python $X/export_ft_kit.py       # then run $X/d17_on_scecilia_kit (see its README)
.venv/bin/python $X/import_ft_kit.py <results dir or .tar.gz>

# Tables 2 and 3 -> RubiNER_comparison.xlsx
for c in "scec test" "cecilia test" "cyner all"; do .venv/bin/python scoring/score.py $c; done
.venv/bin/python scoring/score_variants.py aad-fixC aad-fixC-ft-siesta-rep \
  ft-cecilia__dslim__bert-base-NER ft-cecilia__dbmdz__bert-large-cased-finetuned-conll03-english \
  ft-cecilia__Jean-Baptiste__roberta-large-ner-english
.venv/bin/python $X/build_rubiner_tables.py

# Table 4
T=experiments/table_4_classification
.venv/bin/python $T/refinement_stage3.py anonymise
.venv/bin/python $T/refinement_stage3.py classical
.venv/bin/python $T/refinement_stage3.py kit      # then run $T/roberta_kit (see its README)
.venv/bin/python $T/refinement_stage3.py report

# Table 5
.venv/bin/python experiments/table_5_spy/run_spy_variants.py
.venv/bin/python experiments/table_5_spy/run_spy_variants.py --report

# GLiNER2 thresholds
.venv/bin/python rubiner/thresholds/gliner_cache.py
.venv/bin/python rubiner/thresholds/gliner_exp.py deployed
```

To retrain the general NER instead of downloading it, run
`rubiner/train_and_select.sh`; `run_rubiner.py` uses a local checkpoint when
one exists.

## Notes

- RubiNER as evaluated is the anonymiser in `rubiner/anonymiser/` with its
  general NER replaced at run time by the fine-tuned model (`run_rubiner.py`);
  the anonymiser's own files are not edited.
- Deviations from D17's fine-tuning recipe (warm-up, learning rate and batch
  size for large models, windowing instead of truncation) are listed in
  `finetune.py`.
- S-CECILIA results use seed-grouped splits: the 247 test reports come from
  15 seed incidents not used for training, tuning or model selection.
