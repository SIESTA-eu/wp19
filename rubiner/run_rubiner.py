"""SIESTA (corrected copy) with a general NER model fine-tuned on the S-CECILIA
training seeds (rubiner/train_general_ner, dslim/bert-base-NER).

Variants, all patched at run time; the SIESTA copy is not edited:
  aad-fixC-ft-<labels>-rep   the fine-tuned model replaces dslim/bert-base-NER
  aad-fixC-ft-<labels>-add   the fine-tuned model runs beside it
  <labels> = conll4 (PER/ORG/LOC/MISC) or siesta (25 SIESTA entity types)

The fine-tuned model's labels are read by exact name, not by detect_ner's
substring rules (under which OPERATING_SYSTEM would read as PER): a label that is
a SIESTA entity type is kept as that type, MISC is dropped. Same confidence floor
(0.65) and 1-2 character filter as detect_ner.

Which variant to keep is decided on the 4 validation seeds the fine-tuning used to
pick its epoch (SEED_004, 023, 037, 041), never on the held-out seeds: highest
detection F1 with private leak at most 0.5 point above the corrected SIESTA's;
within 0.5 point of F1, the simpler one (replace before add, conll4 before siesta).

Run from the repository root:
  EVAL_DATASET=scec SIESTA_DEVICE=0 .venv/bin/python rubiner/run_rubiner.py aad-fixC-ft-conll4-rep
"""
import json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); import paths  # noqa: E402,F401  (puts every code folder on sys.path)
import run_siesta  # noqa: E402  (puts SIESTA_REPO on the path and changes directory to it)
import corpus, custom_entities, diccionarios  # noqa: E402
from base_anonymizer import windowed_ner_pipeline  # noqa: E402

RESULTS = ROOT / "rubiner" / "train_general_ner" / "results"
KNOWN = {t for t, *_ in diccionarios.ENTITY_CATALOGUE} - {"MISC"}
# RubiNER's general NER (labels "siesta") as published, pinned to the evaluated weights
RUBINER_MODEL, RUBINER_REVISION = "GVIS-ULE/rubiner", "b08d9f0ba1e9e97655d3c7e0aae9ea1e2fb87d3e"


def checkpoint(labels):
    """A local rubiner/train_general_ner training run if there is one, else RubiNER's published weights."""
    local = RESULTS / f"dslim__bert-base-NER__{labels}" / "checkpoint"
    if local.exists() or labels != "siesta":
        return local
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(RUBINER_MODEL, revision=RUBINER_REVISION))


def ft_detector(pipe):
    def detect(text):
        out = []
        for ent in pipe(text):
            label = ent["entity_group"]
            span = text[ent["start"]:ent["end"]].strip()
            if label in KNOWN and ent["score"] >= 0.65 and len(span) > 2:
                out.append({"start": ent["start"], "end": ent["end"], "text": span, "entity_group": label, "score": ent["score"]})
        return out
    return detect


def main(variant):
    _, _, _, labels, mode = variant.split("-")          # aad-fixC-ft-<labels>-<rep|add>
    ckpt = checkpoint(labels)
    custom_entities._load()
    if run_siesta.DEV >= 0 and custom_entities._model is not None:
        custom_entities._model = custom_entities._model.to("cuda")
    a = run_siesta.build("aad-fixC")
    if mode == "rep":
        a.pipelines.pop("general")
    ft = windowed_ner_pipeline(str(ckpt))
    if run_siesta.DEV >= 0:
        import torch
        for pipe in list(a.pipelines.values()) + [ft]:
            pipe.model.to("cuda"); pipe.device = torch.device("cuda")
    base_detect, ft_detect = a.detect_ner, ft_detector(ft)
    a.detect_ner = lambda text: base_detect(text) + ft_detect(text)
    out = corpus.preds_dir() / f"{variant}.jsonl"
    t0 = time.time()
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"_meta": {"name": f"SIESTA + corrections, general NER fine-tuned on S-CECILIA ({labels}, {'replacing' if mode == 'rep' else 'added'})",
                                      "taxonomy": "native", "family": "SIESTA", "checkpoint": str(ckpt.relative_to(ROOT)) if ckpt.is_relative_to(ROOT) else f"{RUBINER_MODEL}@{RUBINER_REVISION}"}}) + "\n")
        for d in corpus.load():
            _, ents, _ = a.anonymize(d["text"], 0)
            f.write(json.dumps({"id": d["id"], "spans": [[e["start"], e["end"], e["entity_group"]] for e in ents]}) + "\n")
    print(variant, "done", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for v in sys.argv[1:]:
        main(v)
