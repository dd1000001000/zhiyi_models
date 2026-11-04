"""Word n-grams of four input methods on the neutral word-choice test sets v2 and v3, each n-gram alone, CPU only.
Each scores a row's whole candidates (those that spell the whole input); its best one is its pick. Scores go to
scores/<n-gram>_neutral-<set>.jsonl ({"id", "scores"}, null for candidates not compared), accuracies to results.json.

  python run.py --v2 neutral-v2.jsonl --v3 neutral-v3.jsonl --zhiyi zh.ngram --youmo lm.arpa --cyime t9_ngram-v3-ls-tri-c3.lm
                --wanxiang-gram wanxiang-lts-zh-hans.gram --wanxiang-dict <dicts.zip unpacked>/dicts
                --libime lm_sc.arpa --opengram lm_sc.3gm.std.arpa (opengram_arpa.py) --libpinyin <model20.text unpacked>
                --rime-gram zh-hans-t-essay-bgw.gram --rime-essay essay.txt --rime-opencc <OpenCC data/dictionary> --qingjian-lm <qj-score exe>,lm.qj
                --ximegboard-gram XimeGboard.gram --ximegboard-dict <its APK's assets/rime/cn_dicts>
XimeGboard (Murchey/Xime-Gboardlike): its octagram with the dictionary of its default schema rime_ice (the tables
rime_ice.dict.yaml imports), librime-octagram's default parameters (the schema sets none), contextual_suggestions on.
Score files already in scores/ are reused; remove one to score it again.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

GROUPS = {"all": lambda r: True, "full": lambda r: r["style"] == "full", "initials": lambda r: r["style"] == "initials",
          "partial/mixed": lambda r: r["style"] in ("partial", "mixed"), "short": lambda r: r["kind"] == "short",
          "two": lambda r: r["kind"] == "two", "long": lambda r: r["kind"] == "long"}
XIMEGBOARD_TABLES = ["8105", "base", "ext", "tencent", "others"]


def scorers(a):
    """name -> (load function) giving score(context, candidates) -> list of scores."""
    def zhiyi():
        from ngram_lm import NgramLM
        return NgramLM(a.zhiyi).score

    def youmo():
        from ngram_lm import NgramLM
        return NgramLM(a.youmo).score

    def cyime():
        from cyime_ngram import CyimeNgram
        return CyimeNgram(a.cyime).score

    def wanxiang(ctx):
        def load():
            from wanxiang_gram import WanxiangGram
            wg = WanxiangGram(a.wanxiang_gram, a.wanxiang_dict)
            return lambda c, cands: wg.score(c, cands, ctx=ctx)
        return load
    def arpa(path):
        def load():
            from ngram_lm import NgramLM
            return NgramLM(path).score
        return load

    def libpinyin():
        from libpinyin_lm import LibpinyinLM
        return LibpinyinLM(a.libpinyin).score

    def rime():
        from wanxiang_gram import RIME_CFG, WanxiangGram
        wg = WanxiangGram(a.rime_gram, essay=a.rime_essay, cfg=RIME_CFG, opencc=a.rime_opencc)
        return lambda c, cands: wg.score(c, cands, ctx=True)

    def ximegboard():
        from wanxiang_gram import RIME_CFG, WanxiangGram
        wg = WanxiangGram(a.ximegboard_gram, a.ximegboard_dict, cfg=RIME_CFG, tables=XIMEGBOARD_TABLES)
        return lambda c, cands: wg.score(c, cands, ctx=True)

    def qingjian():
        import json
        import subprocess
        exe, lm = a.qingjian_lm.split(",")

        def score(context, cands):
            row = json.dumps({"id": 0, "context": context, "code": "", "cands": cands}, ensure_ascii=False) + chr(10)
            res = subprocess.run([exe, "lm", lm], input=row, capture_output=True, text=True, encoding="utf-8", check=True)
            return [-1e9 if x is None else x for x in json.loads(res.stdout)["scores"]]
        return score
    out = {"zhiyi-ng1017": zhiyi, "youmo-lm": youmo, "cyime-t9-v3": cyime, "wanxiang-ctx": wanxiang(True),
           "wanxiang-noctx": wanxiang(False)}
    for name, ok, fn in (("libime-lm-sc", a.libime, arpa(a.libime)), ("opengram-sunpinyin", a.opengram, arpa(a.opengram)),
                         ("libpinyin-model20", a.libpinyin, libpinyin), ("rime-octagram-hans", a.rime_gram, rime),
                         ("qingjian-lm", a.qingjian_lm, qingjian), ("ximegboard-octagram", a.ximegboard_gram, ximegboard)):
        if ok:
            out[name] = fn
    return out


def score_file(name, load, v, rows):
    path = os.path.join(HERE, "scores", f"{name}_neutral-{v}.jsonl")
    if not os.path.exists(path):
        fn, t0 = load(), time.time()
        with open(path + ".part", "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                sc = [None] * len(r["candidates"])
                for j, x in zip(r["whole"], fn(r["context"], [r["candidates"][j] for j in r["whole"]])):
                    sc[j] = float(x)
                f.write(json.dumps({"id": r["id"], "scores": sc}) + "\n")
        os.replace(path + ".part", path)
        print(f"  {name} {v}: {time.time() - t0:.0f} s", flush=True)
    return {p["id"]: p["scores"] for p in map(json.loads, open(path, encoding="utf-8"))}


def hits(rows, sc):
    return [int(max(r["whole"], key=lambda j: sc[r["id"]][j]) == r["best"]) for r in rows]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    for k in ("v2", "v3", "zhiyi", "youmo", "cyime", "wanxiang-gram", "wanxiang-dict", "libime", "opengram", "libpinyin",
              "rime-gram", "rime-essay", "rime-opencc", "qingjian-lm", "ximegboard-gram", "ximegboard-dict"):
        p.add_argument("--" + k)
    a = p.parse_args()
    os.makedirs(os.path.join(HERE, "scores"), exist_ok=True)
    out = {}
    for v, data in (("v2", a.v2), ("v3", a.v3)):
        rows = [r for r in map(json.loads, open(data, encoding="utf-8")) if r["scored"]]
        out[v] = {}
        for name, load in scorers(a).items():
            sc = score_file(name, load, v, rows)
            out[v][name] = {}
            for g, test in GROUPS.items():
                h = hits([r for r in rows if test(r)], sc)
                out[v][name][g] = {"n": len(h), "acc": 100 * sum(h) / len(h)}
            print(f"{v} {name:<15}" + " ".join(f"{g} {x['acc']:.1f}" for g, x in out[v][name].items()), flush=True)
    path = os.path.join(HERE, "results.json")
    old = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    json.dump({v: old.get(v, {}) | out.get(v, {}) for v in old.keys() | out.keys()}, open(path, "w", encoding="utf-8"),
              indent=1)


if __name__ == "__main__":
    main()
