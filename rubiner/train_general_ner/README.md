# WP17 models fine-tuned on S-CECILIA (server kit)

This kit answers the question: what if the WP17 models were fine-tuned on the same incident reports that SIESTA was tuned on? It trains the three WP17 models from D17.1's fine-tuning experiment on the S-CECILIA **training seeds** with WP17's own recipe, then predicts on the held-out seeds, CECILIA and CyNER. When it finishes, send back `results/` and `logs/`. Scoring happens on the project machine, so the held-out annotations never leave it.

## What it does

- **Models:** `dslim/bert-base-NER`, `dbmdz/bert-large-cased-finetuned-conll03-english`, `Jean-Baptiste/roberta-large-ner-english`.
- **Two label sets per model** (six jobs in all):
  - `conll4`: WP17's recipe, with S-CECILIA types collapsed to PER / ORG / LOC / MISC;
  - `native`: S-CECILIA's 31 types, which gives the models the best chance.
- **Training** is the code used on the project machine for the CECILIA fine-tuning (`experiments/tables_2_3_detection/finetune.py`): 8 epochs, AdamW, effective batch 16, 10% warmup, best epoch by validation loss, seed 42. The batch settings are kept as on an 8 GB GPU, so the results match the method already reported.
- **Data** (`data/`; `MANIFEST.json` has checksums):

  | File | What | Documents |
  |---|---|---|
  | `train.jsonl` | S-CECILIA training-seed reports, with annotations | 663 (37 seeds) |
  | `val.jsonl` | 4 further training seeds, with annotations, for choosing the epoch | 64 |
  | `predict_scec.jsonl` | S-CECILIA held-out seeds, text only | 247 |
  | `predict_cecilia.jsonl` | CECILIA, text only | 879 |
  | `predict_cyner.jsonl` | CyNER, text only | 94 |

## Steps

1. **Set up** (installs PyTorch and transformers 4.53.2 in `.venv`, downloads the models):
   ```bash
   tar xzf ft_scec_kit.tar.gz && cd ft_scec
   ./setup.sh
   ```
2. **Smoke test** (a minute; should end with `ALL DONE`):
   ```bash
   .venv/bin/python run.py --model dslim/bert-base-NER --labels conll4 --gpu 0 --smoke
   rm -rf results/SMOKE__*
   ```
3. **Run** in tmux. With six free GPUs starting at index 0 (pass another start index if needed):
   ```bash
   tmux new -s ft
   ./run_all.sh 0
   ```
   With fewer GPUs, run the jobs from `run_all.sh` one at a time with `--gpu N`. Each run resumes after an interruption: a finished training is not repeated, and finished corpora are skipped. Expect minutes per model on a large GPU.
4. **Send back:**
   ```bash
   tar czf ft_scec_results.tar.gz results/*/train_log.json results/*/env.json results/*/*.jsonl logs
   ```
   This leaves out the model checkpoints, which are large and not needed.

## Notes

- The training files include S-CECILIA annotations, and all files are project data. Keep the kit on machines you control.
- Back on the project machine, `experiments/tables_2_3_detection/import_ft_kit.py ft_scec_results.tar.gz` turns the results into prediction files for the project's scorer.

**Local copy (19-20 Oct 2026).** Adds the label set `siesta` (S-CECILIA types relabelled as SIESTA entity types; three types without an equivalent left as O), and predicts only on the S-CECILIA held-out seeds. Used by `rubiner/train_and_select.sh`; the original kit is `experiments/tables_2_3_detection/d17_on_scecilia_kit`.
