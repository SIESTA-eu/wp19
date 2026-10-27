"""Independent public benchmarks, as gold units.

TAB   Text Anonymization Benchmark (Pilan et al. 2022, arXiv 2202.00443),
      echr_test.json: 127 European Court of Human Rights judgments. Gold =
      annotator1's mentions marked DIRECT or QUASI (NO_MASK is not to be
      masked, by the benchmark's own definition). The official metrics come
      from TAB's evaluation.py (tab_official.py), which uses every annotator.
      None of the tested systems was trained on it.

CyNER Alam et al. 2022, dataset/mitre/test.txt: 747 sentences of MITRE
      ATT&CK threat-intel text, labels Malware / Indicator / System /
      Organization / Vulnerability. No document boundaries, so sentences are
      grouped into passages of 8, tokens joined with light detokenisation.
      cisco-ai/SecureBERT2.0-NER was most likely trained on it (its card
      describes a benchmark cyber-NER corpus with exactly these five labels),
      so the SecureBERT 2.0 variants are contaminated here.
"""
import json, re
from pathlib import Path

TP = Path(__file__).resolve().parent / "third_party"
NO_SPACE_BEFORE = {".", ",", ";", ":", "!", "?", ")", "]", "}", "'s", "n't", "'re", "'ve", "'ll", "'d", "'m", "%"}
NO_SPACE_AFTER = {"(", "[", "{", "$"}


def tab():
    out = []
    for d in json.load(open(TP / "TAB" / "echr_test.json", encoding="utf-8")):
        ann = list(d["annotations"].values())[0]["entity_mentions"]
        spans = [(m["start_offset"], m["end_offset"], m["entity_type"]) for m in ann
                 if m.get("identifier_type") in ("DIRECT", "QUASI")]
        out.append(dict(id=d["doc_id"], text=d["text"], spans=sorted(set(spans))))
    return out


def cyner(per_passage=8):
    sents, cur = [], []
    for line in open(TP / "CyNER" / "dataset" / "mitre" / "test.txt", encoding="utf-8"):
        line = line.rstrip("\r\n")
        if not line.strip():
            if cur: sents.append(cur); cur = []
            continue
        tok, tag = line.rsplit("\t", 1) if "\t" in line else line.rsplit(" ", 1)
        cur.append((tok, tag))
    if cur: sents.append(cur)
    out = []
    for k in range(0, len(sents), per_passage):
        text, spans, open_ = "", [], None
        for sent in sents[k:k + per_passage]:
            for tok, tag in sent:
                if text and tok not in NO_SPACE_BEFORE and text[-1] not in NO_SPACE_AFTER: text += " "
                s = len(text); text += tok; e = len(text)
                if tag.startswith("B-") or (tag.startswith("I-") and (open_ is None or open_[2] != tag[2:])):
                    if open_: spans.append(tuple(open_))
                    open_ = [s, e, tag[2:]]
                elif tag.startswith("I-"):
                    open_[1] = e
                else:
                    if open_: spans.append(tuple(open_)); open_ = None
            if open_: spans.append(tuple(open_)); open_ = None
        out.append(dict(id=f"cyner-{k // per_passage:03d}", text=text, spans=spans))
    return out


if __name__ == "__main__":
    import collections
    for name, f in [("TAB", tab), ("CyNER", cyner)]:
        d = f(); c = collections.Counter(s[2] for x in d for s in x["spans"])
        print(name, len(d), "units,", sum(c.values()), "gold spans", dict(c.most_common()))
    x = cyner()[0]; print(repr(x["text"][:200])); print([x["text"][s:e] for s, e, _ in x["spans"][:8]])
