# Refinement evaluation: incident classification with RoBERTa-base (server kit)

Fine-tunes RoBERTa-base, the transformer of Haughton et al. (2026, NLPAICS, *"When Privacy Helps…"*), on the same CECILIA incidents in ten text versions, all on one split:

- `original`: the original text;
- `manual_labels`, `manual_masking`: the paper's manual pseudonymisation (labels, `xxxx`);
- `rubiner_tokenisation`, `rubiner_masking`, `rubiner_pseudonymisation`: RubiNER (SIESTA with corrections, general NER fine-tuned on S-CECILIA), default settings;
- `wp17_dslim_tokenisation`, `wp17_dslim_masking`, `wp17_roberta_tokenisation`, `wp17_roberta_masking`: WP17's fine-tuning recipe trained on the same S-CECILIA seeds, native labels, every detected span removed.

When it finishes, send back `results/`.

## Settings

As in the paper's notebook (`Transformers_Clean (2).ipynb`) and its Section 4.1: 5 epochs, batch 8, learning rate 5e-5, 10% warm-up, weight decay 0.01, unweighted loss, 512 tokens; 70/30 stratified split, seed 42, 6 classes (595 training, 256 test incidents). Five training seeds (42-46), run seed-major. Each job records test predictions and loss after every epoch, so both the paper's rule (lowest test loss) and the last epoch are available. Every training argument, model revision, data and split hash, library version and the GPU are saved with each result.

The same 50 jobs are also running on a laptop GPU with identical settings; results from either machine can be used.

## Steps

```bash
tar xzf haughton_cls_refine_kit.tar.gz && cd haughton_cls_refine_server
./setup.sh https://download.pytorch.org/whl/cu124   # or your CUDA version
python run.py --list                                  # 50 jobs
./run_parallel.sh 8                                   # one process per GPU; about 10 minutes on 8 GPUs
grep -L 'ALL DONE' logs/gpu*.log                      # lists any shard that did not finish
tar czf haughton_cls_refine_results.tar.gz results logs
```

Finished jobs are skipped, so an interrupted run can be restarted with the same command.
