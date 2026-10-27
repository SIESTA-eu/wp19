"""CyNER (Alam et al. 2022), dataset/mitre/test.txt: 747 sentences of MITRE
ATT&CK threat-intel text, labels Malware / Indicator / System / Organization /
Vulnerability. No document boundaries, so sentences are grouped into passages
of 8, tokens joined with light detokenisation.
"""
from pathlib import Path

TP = Path(__file__).resolve().parents[1] / "third_party"
NO_SPACE_BEFORE = {".", ",", ";", ":", "!", "?", ")", "]", "}", "'s", "n't", "'re", "'ve", "'ll", "'d", "'m", "%"}
NO_SPACE_AFTER = {"(", "[", "{", "$"}


def load(per_passage=8):
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
    d = load(); c = collections.Counter(s[2] for x in d for s in x["spans"])
    print("CyNER", len(d), "passages,", sum(c.values()), "gold spans", dict(c.most_common()))
