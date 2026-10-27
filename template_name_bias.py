"""Demographic bias in person-name detection (template test, origin x gender).

Following Mehrabi et al. (2020) and Mishra et al. (2020): the same
incident-report sentences are filled with names from different origins and
genders, and we measure how often each name is (a) fully redacted by the
release pipeline, (b) typed as a person, (c) found as a person by the
general model (dslim/bert-base-NER) alone. The gender comparison on a
published name list is in mehrabi_bias.py.

Design
  7 origins x 2 genders x 20 given names. Within an origin, female name i
  and male name i share surname i, so gender is compared with surnames held
  fixed. Full names go into 8 templates, given names alone into 4. Templates
  avoid gendered pronouns.

  The name lists are common given names and surnames for each origin,
  written for this test. They are an approximation of each group, not a
  census sample; the result is a screening test, not a population estimate.

CIs are 95% bootstrap intervals over names (2,000 resamples), since
sentences sharing a name are not independent.

Runs the repo's release pipeline on CPU, as deployed, default configuration
(GLiNER pack on, default preserved types).

Usage: [SIESTA_DEVICE=0] [GENERAL_FIRST=1] template_name_bias.py [--report]
"""
import json, os, random, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE / "siesta_aad6c5b_fixC" / "Anonimizacion"
# GENERAL_FIRST=1: general model with aggregation_strategy="first" (see
# run_siesta.patch_general_first); outputs get a "_first" suffix.
FIRST = os.environ.get("GENERAL_FIRST") == "1"
SUFFIX = "_first" if FIRST else ""
OUT = HERE / "logs" / f"template_name_bias{SUFFIX}.jsonl"

NAMES = {
    "Anglo": dict(
        F="Emily Sarah Jessica Hannah Megan Laura Rachel Emma Olivia Charlotte Amy Rebecca Katherine Lauren Abigail Sophie Lucy Grace Alice Claire".split(),
        M="James John Michael David Robert William Thomas Daniel Matthew Andrew Christopher Joseph Ryan Benjamin Jack George Harry Samuel Edward Peter".split(),
        S="Smith Johnson Williams Brown Jones Miller Davis Wilson Taylor Anderson Thomas Moore Clark Walker Wright Hall Wood Turner Hughes Cooper".split()),
    "Hispanic": dict(
        F="María Carmen Lucía Sofía Isabel Paula Elena Marta Rocío Pilar Guadalupe Ana Beatriz Alejandra Daniela Valentina Ximena Camila Fernanda Inés".split(),
        M="José Juan Javier Carlos Alejandro Miguel Pablo Sergio Jorge Diego Francisco Antonio Manuel Raúl Andrés Fernando Santiago Rodrigo Iván Gonzalo".split(),
        S="García Rodríguez Martínez Hernández López González Pérez Sánchez Ramírez Torres Flores Rivera Gómez Díaz Morales Jiménez Ruiz Álvarez Castillo Ortega".split()),
    "Arabic": dict(
        F="Fatima Aisha Mariam Layla Noor Huda Amira Salma Yasmin Rania Hana Zainab Leila Dalia Samira Nadia Reem Lina Malak Asma".split(),
        M="Mohammed Ahmed Omar Ali Hassan Youssef Khalid Ibrahim Mustafa Tariq Karim Hamza Bilal Faisal Samir Walid Nabil Rami Ziad Majid".split(),
        S="Al-Sayed Haddad Khalil Mansour Nasser Hamdan Saleh Aziz Farouk Abdullah Rahman Darwish Hakim Jaber Kassem Suleiman Othman Barakat Mahmoud Bakr".split()),
    "South Asian": dict(
        F="Priya Ananya Kavya Deepika Lakshmi Pooja Neha Shreya Divya Meera Aishwarya Sunita Anjali Radhika Swati Nisha Kavitha Gayatri Isha Ritu".split(),
        M="Rahul Arjun Vikram Rohan Amit Suresh Rajesh Anil Sanjay Karthik Aditya Vivek Pranav Siddharth Manoj Ravi Harish Nikhil Varun Abhishek".split(),
        S="Sharma Patel Gupta Singh Kumar Reddy Iyer Nair Mehta Joshi Rao Desai Chatterjee Banerjee Menon Pillai Agarwal Kapoor Malhotra Verma".split()),
    "Chinese": dict(
        F="Mei Jing Yan Hui Ling Fang Xiaoying Qian Yun Lan Ying Xue Shufen Lijuan Xiuying Yuting Jiayi Ruoxi Meiling Hong".split(),
        M="Wei Jun Hao Lei Qiang Tao Bo Gang Jian Ming Yong Chao Peng Zhiwei Haoran Yuxuan Zihao Jianguo Xiaoming Dong".split(),
        S="Wang Li Zhang Liu Chen Yang Huang Zhao Wu Zhou Xu Sun Ma Zhu Hu Guo He Lin Gao Luo".split()),
    "West African": dict(
        F="Ngozi Chiamaka Adaeze Folake Funmilayo Yetunde Abena Akosua Efua Amaka Chinwe Nneka Bisola Titilayo Adwoa Ifeoma Kemi Ebele Obiageli Yaa".split(),
        M="Chukwuemeka Babatunde Olumide Chinedu Emeka Obinna Ikenna Kwame Kofi Kwabena Yaw Kojo Tunde Segun Femi Nnamdi Chidi Ifeanyi Kayode Gbenga".split(),
        S="Okafor Adeyemi Okonkwo Mensah Owusu Boateng Eze Nwosu Balogun Oyelaran Asante Osei Danquah Obi Okeke Adeleke Ogunleye Nwachukwu Agyeman Afolabi".split()),
    "Slavic": dict(
        F="Agnieszka Katarzyna Małgorzata Natalia Olga Svetlana Tatiana Irina Ekaterina Yulia Oksana Magdalena Joanna Zofia Elżbieta Ludmila Daria Anastasia Ewa Milena".split(),
        M="Piotr Krzysztof Tomasz Paweł Andrzej Dmitri Sergei Alexei Ivan Mikhail Nikolai Vladimir Oleksandr Bogdan Jakub Wojciech Marek Igor Yuri Stanisław".split(),
        S="Nowak Mazur Wójcik Kowalczyk Krawczyk Dudek Wróbel Pawlak Sikora Baran Shevchenko Kovalenko Bondarenko Tkachenko Kravchenko Melnyk Horvat Novák Dvořák Marković".split()),
}

TEMPLATES = [
    "On 12 March, {N} reported a suspicious login to the security team.",
    "The phishing email was sent to {N}, who forwarded it to the SOC.",
    "{N} clicked the link and entered their credentials on the fake portal.",
    "According to {N}, the ransom note appeared on the file server at 09:14.",
    "The attacker impersonated {N} in messages to the finance department.",
    "Logs show that the account belonging to {N} was used to exfiltrate data.",
    "After the incident, {N} reset the passwords of all affected users.",
    "The analyst contacted {N} to confirm the unusual transfer.",
]
FIRST_ONLY_TEMPLATES = TEMPLATES[:4]


def covered(spans, s, e):
    """Number of characters of [s, e) covered by the given spans."""
    hit = set()
    for a, b in spans:
        hit.update(range(max(a, s), min(b, e)))
    return len(hit)


def run():
    sys.path.insert(0, str(REPO)); os.chdir(REPO)
    from en_anonymizer import EnglishAnonymizer
    a = EnglishAnonymizer()
    if FIRST:
        sys.path.insert(0, str(HERE))
        import run_siesta
        run_siesta.patch_general_first(a)
    # SIESTA_DEVICE=0 runs the models on the GPU (same output as CPU).
    if int(os.environ.get("SIESTA_DEVICE", "-1")) >= 0:
        import torch, custom_entities
        for pipe in list(a.pipelines.values()) + ([a.cisco_pipe] if a.cisco_pipe else []):
            pipe.model.to("cuda"); pipe.device = torch.device("cuda")
        if custom_entities._load() is not None:
            custom_entities._model = custom_entities._model.to("cuda")
    general = a.pipelines["general"]
    rows = []
    t0 = time.time()
    for origin, d in NAMES.items():
        for g in ("F", "M"):
            for i, first in enumerate(d[g]):
                for form, name, temps in (("full", f"{first} {d['S'][i]}", TEMPLATES),
                                          ("first", first, FIRST_ONLY_TEMPLATES)):
                    for ti, t in enumerate(temps):
                        text = t.replace("{N}", name)
                        s = text.index(name); e = s + len(name)
                        _, ents, _ = a.anonymize(text, 0)
                        red = [(x["start"], x["end"]) for x in ents if not x.get("preserved")]
                        per = [(x["start"], x["end"]) for x in ents if x.get("readable_type") == "PER"]
                        types = sorted({x.get("readable_type") for x in ents if covered([(x["start"], x["end"])], s, e)})
                        gen = [(x["start"], x["end"]) for x in general(text) if x["entity_group"] == "PER"]
                        n = e - s
                        rows.append(dict(origin=origin, gender=g, form=form, name=name, template=ti,
                                         redacted=covered(red, s, e) == n,
                                         leaked=1 - covered(red, s, e) / n,
                                         per=covered(per, s, e) == n,
                                         general_per=covered(gen, s, e) == n,
                                         types=types))
            print(origin, g, f"{time.time() - t0:.0f}s", flush=True)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return rows


def boot(rows, key, n=2000, seed=0):
    """Mean of `key` and a 95% CI from resampling names."""
    by = {}
    for r in rows:
        by.setdefault(r["name"], []).append(float(r[key]))
    means = [sum(v) / len(v) for v in by.values()]
    rng = random.Random(seed)
    bs = sorted(sum(rng.choice(means) for _ in means) / len(means) for _ in range(n))
    return sum(means) / len(means), bs[int(0.025 * n)], bs[int(0.975 * n) - 1]


def fmt(m, lo, hi):
    return f"{m:.2f} ({lo:.2f}–{hi:.2f})"


def report(rows):
    lines = ["# Name-detection bias test", "",
             f"{len(rows)} sentences; 95% bootstrap CIs over names. See template_name_bias.py for the design.", ""]
    keys = [("redacted", "Fully redacted"), ("per", "Typed as person"), ("general_per", "General model finds person")]
    for form, title in (("full", "Full names (8 templates)"), ("first", "Given name only (4 templates)")):
        sub = [r for r in rows if r["form"] == form]
        lines += [f"## {title}", "", "| Origin | Gender | " + " | ".join(k[1] for k in keys) + " | Mean leaked chars |",
                  "|---|---|" + "---|" * (len(keys) + 1)]
        for origin in NAMES:
            for g in ("F", "M", None):
                cell = [r for r in sub if r["origin"] == origin and (g is None or r["gender"] == g)]
                lines.append(f"| {origin} | {g or 'all'} | " + " | ".join(fmt(*boot(cell, k)) for k, _ in keys)
                             + f" | {boot(cell, 'leaked')[0]:.1%} |")
        for g in ("F", "M"):
            cell = [r for r in sub if r["gender"] == g]
            lines.append(f"| **All** | {g} | " + " | ".join(fmt(*boot(cell, k)) for k, _ in keys)
                         + f" | {boot(cell, 'leaked')[0]:.1%} |")
        lines.append("")
        # what non-person types names were given
        mis = {}
        for r in sub:
            if not r["per"]:
                for t in r["types"] or ["(none)"]:
                    mis.setdefault(r["origin"], {}).setdefault(t, 0)
                    mis[r["origin"]][t] += 1
        lines += ["Types given to names not typed as person:", ""]
        for origin, c in mis.items():
            lines.append(f"- {origin}: " + ", ".join(f"{t} {n}" for t, n in sorted(c.items(), key=lambda x: -x[1])))
        lines.append("")
    # worst names
    full = [r for r in rows if r["form"] == "full"]
    by = {}
    for r in full:
        by.setdefault((r["origin"], r["gender"], r["name"]), []).append(r["redacted"])
    worst = sorted(by.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))[:25]
    lines += ["## Full names least often fully redacted", "", "| Name | Origin | Gender | Redacted |", "|---|---|---|---|"]
    lines += [f"| {k[2]} | {k[0]} | {k[1]} | {sum(v)}/{len(v)} |" for k, v in worst]
    (HERE / f"TEMPLATE_NAME_BIAS{SUFFIX}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    if "--report" in sys.argv and OUT.exists():
        report([json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines()])
    else:
        report(run())
