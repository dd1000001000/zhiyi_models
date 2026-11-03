"""Sentence composition: the whole pinyin of a clause in, the clause out, each model alone, on the neutral set v1's
12,000 long clauses (4-16 characters; typed in full, with the last syllable cut to its first letter, mixed, or as
initials; no context, a short one, or up to 128 characters). P@1: the first output is the clause; P@5: it is among the
first five.

- Zhiyi: one pass of the composition head over all the slots at once; the n-best of the independent slots (top 5
  characters per slot) as its list. The input method's own composition (more passes, the n-gram) is not used.
- PinyinGPT-Concat / GPT2-zh-21k: constrained beam search with 16 beams, as the paper's benchmark (pinyingpt.py).

  python compose.py --v1 test.jsonl --zhiyi zhiyi-pumpkin-1026=<unpacked int8 zip> ...      (onnxruntime-directml)
  python compose.py --v1 test.jsonl --pinyingpt <folder> --gpt2-zh-21k <folder>             (PyTorch, CUDA)
Outputs compose/<model>.jsonl ({"id", "top": [five texts]}, reused when present) and results.json's "compose".
"""
import argparse
import heapq
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

GROUPS = {"all": lambda r: True, **{f"style={s}": (lambda s: lambda r: r["style"] == s)(s)
                                     for s in ("full", "partial", "mixed", "initials")},
          **{f"context={b}": (lambda b: lambda r: r["bucket"] == b)(b) for b in ("none", "short", "long")}}


def units(r):
    from zhiyi import slot_units
    return slot_units(r["syllables"], r["code"])


def nbest(slots, n):
    beam = [(0.0, "")]
    for opts in slots:
        beam = heapq.nlargest(n, ((s + p, t + c) for s, t in beam for c, p in (opts or [("?", 0.0)])), key=lambda x: x[0])
    return beam


def systems(a):
    out = {}
    for spec in a.zhiyi or []:
        name, folder = spec.split("=", 1)

        def zhiyi(rows, folder=folder):
            from zhiyi import Zhiyi
            m = Zhiyi(folder, None if a.cpu else 0)
            return [[t for _, t in nbest(slots, 5)] for slots in m.compose([(r["context"], units(r)) for r in rows])]
        out[name] = zhiyi
    for key, concat, typed, name in (("pinyingpt", True, False, "pinyingpt-concat"),
                                     ("pinyingpt_typed", True, True, "pinyingpt-concat-typed"),
                                     ("gpt2_zh_21k", False, False, "gpt2-zh-21k")):
        path = getattr(a, key)
        if path:
            def pgpt(rows, path=path, concat=concat, typed=typed):
                from pinyingpt import PinyinGPT
                m = PinyinGPT(path, concat, typed=typed)
                todo, keys = [], []
                for i, r in enumerate(rows):
                    k = [s.replace("ü", "v") if kind == "f" else s[0] for kind, s in units(r)]
                    if all(x in m.key_index and x in m.py_id for x in k):  # a syllable it has no token for: no output
                        todo.append(i)
                        keys.append(k)
                got = m.generate([(rows[i]["context"], k) for i, k in zip(todo, keys)])
                res = [[] for _ in rows]
                for i, g in zip(todo, got):
                    res[i] = [t for _, t in g]
                return res
            out[name] = pgpt
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--v1", required=True)
    p.add_argument("--zhiyi", action="append")
    p.add_argument("--pinyingpt")
    p.add_argument("--pinyingpt-typed")
    p.add_argument("--gpt2-zh-21k")
    p.add_argument("--cpu", action="store_true")
    a = p.parse_args()
    rows = [json.loads(l) for l in open(a.v1, encoding="utf-8")]
    d = os.path.join(HERE, "compose")
    os.makedirs(d, exist_ok=True)
    for name, fn in systems(a).items():
        path = os.path.join(d, name + ".jsonl")
        if not os.path.exists(path):
            t0 = time.time()
            top = fn(rows)
            with open(path + ".part", "w", encoding="utf-8", newline="\n") as f:
                for r, t in zip(rows, top):
                    f.write(json.dumps({"id": r["id"], "top": t}, ensure_ascii=False) + "\n")
            os.replace(path + ".part", path)
            print(f"  {name}: {time.time() - t0:.0f} s", flush=True)
    res = {}
    for f in sorted(os.listdir(d)):
        if not f.endswith(".jsonl"):
            continue
        top = {q["id"]: q["top"] for q in map(json.loads, open(os.path.join(d, f), encoding="utf-8"))}
        name = f[:-6]
        res[name] = {}
        for g, test in GROUPS.items():
            rs = [r for r in rows if test(r)]
            p1 = sum(bool(top[r["id"]]) and top[r["id"]][0] == r["target"] for r in rs)
            p5 = sum(r["target"] in top[r["id"]][:5] for r in rs)
            res[name][g] = {"n": len(rs), "p1": 100 * p1 / len(rs), "p5": 100 * p5 / len(rs)}
        print(f"{name:<20}" + " ".join(f"{g} {x['p1']:.1f}/{x['p5']:.1f}" for g, x in res[name].items()), flush=True)
    path = os.path.join(HERE, "results.json")
    old = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    old["compose"] = old.get("compose", {}) | res
    json.dump(old, open(path, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
