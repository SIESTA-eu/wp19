"""Gender bias in person-name detection on the published benchmark of
Mehrabi et al. (2020), "Man is to Person as Woman is to Location"
(https://github.com/Ninarehm/NERGenderBias, commit cebd776, cloned into
third_party/NERGenderBias_src/repo; used as data only, its code not run).

The benchmark fills nine templates with U.S. Social Security baby names for
1880-2018. Running all of it through the release pipeline on CPU would take
days, so this uses 2018 names and four templates:
  1  "<Name>"              (no context)
  4  "<Name> is a person"  (the paper's headline template)
  8  "<Name> is a nurse"
  9  "<Name> is a doctor"
Names per gender: a seeded random 1,000 of all 2018 names (unweighted
errors, as in the paper) plus the 500 most frequent (frequency-weighted
errors; they cover 62% of female and 74% of male 2018 births).

Errors, as defined in the paper:
  type 1  name not tagged PERSON (type 2 + type 3)
  type 2  name tagged, but as something other than PERSON
  type 3  name not tagged at all
plus, for SIESTA, "leak": the name is not fully covered by a redacted
(non-preserved) entity. Scored for the release pipeline and for its general
model (dslim/bert-base-NER) alone. A name counts as PERSON only if one PER
span covers all of it.

Usage: [SIESTA_DEVICE=0] [MEHRABI_YEAR=1985] [GENERAL_FIRST=1] mehrabi_bias.py [--report]
"""
import json, os, random, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE / "third_party" / "NERGenderBias_src" / "repo" / "benchmark"
REPO = HERE / "siesta_aad6c5b_fixC" / "Anonimizacion"
# MEHRABI_YEAR picks the census year (default 2018); other years get their
# own output files, suffixed with the year.
YEAR = int(os.environ.get("MEHRABI_YEAR", "2018"))
SUFFIX = "" if YEAR == 2018 else f"_{YEAR}"
# GENERAL_FIRST=1: general model with aggregation_strategy="first" (see
# run_siesta.patch_general_first); outputs get a "_first" suffix.
FIRST = os.environ.get("GENERAL_FIRST") == "1"
SUFFIX += "_first" if FIRST else ""
OUT = HERE / "logs" / f"mehrabi_bias{SUFFIX}.jsonl"
TEMPLATES = {1: "", 4: "_withsent_isaperson", 8: "_withsent_isanurse", 9: "_withsent_isadoctor"}
N_RANDOM, N_TOP = 1000, 500


def load(template, gender):
    f = BENCH / f"Template_{template}" / f"{'female' if gender == 'F' else 'male'}{YEAR}.txt{TEMPLATES[template]}"
    rows = []
    for line in f.read_text(encoding="utf-8").splitlines():
        sent, _, freq = line.rsplit(",", 2)
        rows.append((sent, int(freq)))
    return rows


def sample(gender):
    """Names to test: random N_RANDOM plus top N_TOP, from template 1's list."""
    rows = load(1, gender)
    rnd = set(random.Random(f"{YEAR}{gender}").sample([n for n, _ in rows], N_RANDOM))
    top = {n for n, _ in sorted(rows, key=lambda r: -r[1])[:N_TOP]}
    return rnd, top, dict(rows)


def classify(ents, s, e, label_key):
    """'PER', 'OTHER' or 'NONE' for the name span [s, e)."""
    over = [x for x in ents if x["start"] < e and x["end"] > s]
    if any(x[label_key] in ("PER",) and x["start"] <= s and x["end"] >= e for x in over):
        return "PER"
    return "OTHER" if over else "NONE"


def run():
    sys.path.insert(0, str(REPO)); os.chdir(REPO)
    from en_anonymizer import EnglishAnonymizer
    a = EnglishAnonymizer()
    if FIRST:
        import run_siesta
        run_siesta.patch_general_first(a)
    # SIESTA_DEVICE=0 runs the models on the GPU, as in run_siesta.py; same
    # output as CPU. Default -1 (CPU), as deployed.
    if int(os.environ.get("SIESTA_DEVICE", "-1")) >= 0:
        import torch, custom_entities
        for pipe in list(a.pipelines.values()) + ([a.cisco_pipe] if a.cisco_pipe else []):
            pipe.model.to("cuda"); pipe.device = torch.device("cuda")
        if custom_entities._load() is not None:
            custom_entities._model = custom_entities._model.to("cuda")
    general = a.pipelines["general"]
    rows, t0 = [], time.time()
    for g in ("F", "M"):
        rnd, top, freq = sample(g)
        names = rnd | top
        for t in TEMPLATES:
            sents = {sent: f for sent, f in load(t, g)}
            for name in sorted(names):
                sent = name if t == 1 else next(x for x in (f"{name} {s}" for s in
                       ("is a person", "is a nurse", "is a doctor")) if x in sents)
                _, ents, _ = a.anonymize(sent, 0)
                s, e = 0, len(name)
                red = set()
                for x in ents:
                    if not x.get("preserved"):
                        red.update(range(max(x["start"], s), min(x["end"], e)))
                gen = [dict(start=x["start"], end=x["end"], entity_group=x["entity_group"]) for x in general(sent)]
                rows.append(dict(gender=g, template=t, name=name, freq=freq[name],
                                 random=name in rnd, top=name in top,
                                 siesta=classify(ents, s, e, "readable_type"),
                                 general=classify(gen, s, e, "entity_group"),
                                 leak=len(red) < e - s,
                                 types=sorted({x.get("readable_type") for x in ents if x["start"] < e and x["end"] > s})))
            print(g, t, f"{time.time() - t0:.0f}s", flush=True)
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return rows


def rates(rows, sys_key, weighted):
    w = [r["freq"] if weighted else 1 for r in rows]
    tot = sum(w)
    err = lambda cond: sum(wi for r, wi in zip(rows, w) if cond(r)) / tot
    out = dict(t1=err(lambda r: r[sys_key] != "PER"), t2=err(lambda r: r[sys_key] == "OTHER"),
               t3=err(lambda r: r[sys_key] == "NONE"))
    if sys_key == "siesta":
        out["leak"] = err(lambda r: r["leak"])
    return out


def boot_diff(rows, sys_key, key, n=2000):
    """95% bootstrap CI of (female - male) unweighted error, resampling names."""
    rng = random.Random(0)
    v = {g: [(r[sys_key] != "PER") if key == "t1" else r["leak"] for r in rows if r["gender"] == g] for g in "FM"}
    d = sorted(sum(rng.choice(v["F"]) for _ in v["F"]) / len(v["F"]) -
               sum(rng.choice(v["M"]) for _ in v["M"]) / len(v["M"]) for _ in range(n))
    return d[int(0.025 * n)], d[int(0.975 * n) - 1]


def boot_diff_weighted(rows, sys_key, key, n=2000):
    """95% bootstrap CI of (female - male) birth-weighted error, resampling
    the top names with their weights."""
    rng = random.Random(0)
    v = {g: [(r["freq"], (r[sys_key] != "PER") if key == "t1" else r["leak"]) for r in rows if r["gender"] == g] for g in "FM"}

    def wmean(xs):
        return sum(w * e for w, e in xs) / sum(w for w, _ in xs)
    d = sorted(wmean([rng.choice(v["F"]) for _ in v["F"]]) - wmean([rng.choice(v["M"]) for _ in v["M"]]) for _ in range(n))
    return d[int(0.025 * n)], d[int(0.975 * n) - 1]


def report(rows):
    L = ["# Gender bias on the Mehrabi et al. (2020) benchmark", "",
         f"{YEAR} U.S. names; {N_RANDOM} random names per gender (unweighted) and the top {N_TOP} (weighted by births). "
         "Error types as in the paper: 1 = not tagged PERSON, 2 = tagged as another type, 3 = not tagged. "
         "Leak = name not fully redacted by SIESTA. Δ = female − male, with 95% bootstrap CI over names.", ""]
    for sys_key, title in (("siesta", "SIESTA release pipeline"), ("general", "dslim/bert-base-NER alone")):
        L += [f"## {title}", "",
              "| Template | Weighting | Type-1 F | Type-1 M | Type-2 F | Type-2 M | Type-3 F | Type-3 M |"
              + (" Leak F | Leak M |" if sys_key == "siesta" else "") + " Δ type-1 (CI) |" + (" Δ leak (CI) |" if sys_key == "siesta" else ""),
              "|---|---|" + "---|" * (6 + (2 if sys_key == "siesta" else 0) + 1 + (1 if sys_key == "siesta" else 0))]
        for t in TEMPLATES:
            for weighted in (False, True):
                sel = lambda r, g: r["template"] == t and r["gender"] == g and (r["top"] if weighted else r["random"])
                F = rates([r for r in rows if sel(r, "F")], sys_key, weighted)
                M = rates([r for r in rows if sel(r, "M")], sys_key, weighted)
                cells = [f"{F['t1']:.1%}", f"{M['t1']:.1%}", f"{F['t2']:.1%}", f"{M['t2']:.1%}", f"{F['t3']:.1%}", f"{M['t3']:.1%}"]
                if sys_key == "siesta":
                    cells += [f"{F['leak']:.1%}", f"{M['leak']:.1%}"]
                if weighted:
                    top = [r for r in rows if r["template"] == t and r["top"]]
                    lo, hi = boot_diff_weighted(top, sys_key, "t1")
                    cells.append(f"{F['t1'] - M['t1']:+.1%} ({lo:+.1%}, {hi:+.1%})")
                    if sys_key == "siesta":
                        lo, hi = boot_diff_weighted(top, sys_key, "leak")
                        cells.append(f"{F['leak'] - M['leak']:+.1%} ({lo:+.1%}, {hi:+.1%})")
                else:
                    unw = [r for r in rows if r["template"] == t and r["random"]]
                    lo, hi = boot_diff(unw, sys_key, "t1")
                    cells.append(f"{F['t1'] - M['t1']:+.1%} ({lo:+.1%}, {hi:+.1%})")
                    if sys_key == "siesta":
                        lo, hi = boot_diff(unw, sys_key, "leak")
                        cells.append(f"{F['leak'] - M['leak']:+.1%} ({lo:+.1%}, {hi:+.1%})")
                L.append(f"| {t} | {'births' if weighted else 'names'} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## Types SIESTA gave to names not typed as person (template 4, random names)", ""]
    for g in "FM":
        c = {}
        for r in rows:
            if r["template"] == 4 and r["random"] and r["gender"] == g and r["siesta"] != "PER":
                for ty in r["types"] or ["(none)"]:
                    c[ty] = c.get(ty, 0) + 1
        L.append(f"- {g}: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items(), key=lambda x: -x[1])))
    L.append("")
    L += ["## Most frequent names leaked by SIESTA (template 4)", "", "| Name | Gender | 2018 births | Typed as |", "|---|---|---|---|"]
    leaked = sorted((r for r in rows if r["template"] == 4 and r["leak"]), key=lambda r: -r["freq"])[:20]
    L += [f"| {r['name']} | {r['gender']} | {r['freq']:,} | {', '.join(r['types']) or '(none)'} |" for r in leaked]
    (HERE / f"MEHRABI_BIAS{SUFFIX}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    if "--report" in sys.argv and OUT.exists():
        report([json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines()])
    else:
        report(run())
