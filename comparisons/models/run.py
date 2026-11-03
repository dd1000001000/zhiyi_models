"""Word-choice models of several input methods on the neutral word-choice test sets v2 and v3, each model alone.
Every model gets the same rows and the same candidates; the pick is the best-scoring of the row's whole candidates
(those that spell the whole input), right when it is the row's best. Scores go to scores/<model>_neutral-<set>.jsonl
({"id", "scores"} as the sets' score.py reads, null for candidates not scored), accuracies to results.json.

  python run.py --v2 neutral-v2.jsonl --v3 neutral-v3.jsonl
                --zhiyi zhiyi-pumpkin-1026=<unpacked laya-ml-pumpkin-1026-int8g.zip> --zhiyi zhiyi-flash-1031=<...>
                --metasequoia <rerank-v1 folder> --youmo sentence-model-large.safetensors
                --chaoinput <ChaoInput-v3.3.16-win64/data>,<ChaoInput-src-v3.3.16/src/ChaoInput.LmTrain.Gpu/model.py>
                --qwen <Qwen3.5-2B-Base folder> --pinyingpt <pinyingpt-concat folder> --gpt2-zh-21k <gpt2-zh-21k folder>
                [--pinyingpt-typed <pinyingpt-concat folder>]  (PinyinGPT fed the pinyin as typed: a side check)
Give only the models to run; score files already in scores/ are reused (remove one to score it again), and every
score file there is counted in results.json. Zhiyi's graphs run on DirectML (onnxruntime-directml, --cpu for CPU);
the PyTorch models on CUDA; MetasequoiaIME on CPU.
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


def whole_only(rows, score_rows, item):
    """score_rows(items for the rows' whole candidates) -> scores; null for the other candidates."""
    todo = [r for r in rows if r["scored"]]
    got = score_rows([item(r) for r in todo])
    out = {r["id"]: [None] * len(r["candidates"]) for r in rows}
    for r, sc in zip(todo, got):
        for j, x in zip(r["whole"], sc):
            out[r["id"]][j] = x
    return out


def scorers(a):
    """name -> function(all rows of a set) -> {id: scores}."""
    def cands(r):
        return [r["candidates"][j] for j in r["whole"]]

    out = {}
    for spec in a.zhiyi or []:
        name, folder = spec.split("=", 1)
        if a.context_chars:
            name += f"-ctx{a.context_chars}"

        def zhiyi(rows, folder=folder):
            from zhiyi import Zhiyi
            m = Zhiyi(folder, None if a.cpu else 0, context_chars=a.context_chars)
            return {r["id"]: sc for r, sc in zip(rows, m.score(rows))}  # the whole page, as the input method
        out[name] = zhiyi
    if a.metasequoia:
        for size in ("8x448", "6x256"):
            def meta(rows, size=size):
                from metasequoia import Metasequoia
                m = Metasequoia(a.metasequoia, "rerank-" + size)
                return whole_only(rows, m.score_rows, lambda r: (r["context"], cands(r)))
            out["metasequoia-rerank-" + size] = meta
    if a.youmo:
        def youmo(rows):
            from youmo import Youmo
            m = Youmo(a.youmo)
            return whole_only(rows, m.score_rows, lambda r: (r["context"], cands(r)))
        out["youmo-25m"] = youmo
    if a.chaoinput:
        def chao(rows):
            from chaoinput import Chao
            m = Chao(*a.chaoinput.split(","))
            return whole_only(rows, m.score_rows, lambda r: (r["context"], cands(r), [
                "".join(s for s, _ in r["readings"][j]) for j in r["whole"]]))
        out["chaoinput"] = chao
    if a.qwen:
        def qwen(rows):
            from qwen import Qwen
            m = Qwen(a.qwen)
            return whole_only(rows, m.score_rows, lambda r: (r["context"], cands(r)))
        out["qwen3.5-2b-base"] = qwen
    for key, concat, typed, name in (("pinyingpt", True, False, "pinyingpt-concat"),
                                     ("pinyingpt_typed", True, True, "pinyingpt-concat-typed"),
                                     ("gpt2_zh_21k", False, False, "gpt2-zh-21k")):
        path = getattr(a, key)
        if path:
            def pgpt(rows, path=path, concat=concat, typed=typed):
                from pinyingpt import PinyinGPT
                m = PinyinGPT(path, concat, typed=typed)
                return whole_only(rows, m.score_rows, lambda r: (r["context"], [
                    (r["candidates"][j], r["readings"][j]) for j in r["whole"]]))
            out[name] = pgpt
    return out


def write(path, rows, sc):
    with open(path + ".part", "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps({"id": r["id"], "scores": [None if x is None else round(x, 4) for x in sc[r["id"]]]}) + "\n")
    os.replace(path + ".part", path)


def hits(rows, sc):
    return [int(max(r["whole"], key=lambda j: sc[r["id"]][j] if sc[r["id"]][j] is not None else -1e9) == r["best"])
            for r in rows]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--v2")
    p.add_argument("--v3")
    p.add_argument("--zhiyi", action="append", help="name=folder of an unpacked int8 release zip (repeatable)")
    for k in ("metasequoia", "youmo", "chaoinput", "qwen", "pinyingpt", "pinyingpt-typed", "gpt2-zh-21k"):
        p.add_argument("--" + k)
    p.add_argument("--cpu", action="store_true", help="Zhiyi's graphs on the CPU instead of DirectML")
    p.add_argument("--context_chars", type=int, help="Zhiyi: the context's last N characters only (named <name>-ctxN)")
    a = p.parse_args()
    os.makedirs(os.path.join(HERE, "scores"), exist_ok=True)
    out = {}
    for v, data in (("v2", a.v2), ("v3", a.v3)):
        if not data:
            continue
        rows = [json.loads(l) for l in open(data, encoding="utf-8")]
        for name, fn in scorers(a).items():
            path = os.path.join(HERE, "scores", f"{name}_neutral-{v}.jsonl")
            if not os.path.exists(path):
                t0 = time.time()
                write(path, rows, fn(rows))
                print(f"  {name} {v}: {time.time() - t0:.0f} s", flush=True)
        scored = [r for r in rows if r["scored"]]
        out[v] = {}
        for f in sorted(os.listdir(os.path.join(HERE, "scores"))):
            if not f.endswith(f"_neutral-{v}.jsonl"):
                continue
            name = f[:-len(f"_neutral-{v}.jsonl")]
            sc = {q["id"]: q["scores"] for q in map(json.loads, open(os.path.join(HERE, "scores", f), encoding="utf-8"))}
            out[v][name] = {}
            for g, test in GROUPS.items():
                h = hits([r for r in scored if test(r)], sc)
                out[v][name][g] = {"n": len(h), "acc": 100 * sum(h) / len(h)}
            print(f"{v} {name:<28}" + " ".join(f"{g} {x['acc']:.1f}" for g, x in out[v][name].items()), flush=True)
    old = json.load(open(os.path.join(HERE, "results.json"), encoding="utf-8")) if os.path.exists(
        os.path.join(HERE, "results.json")) else {}
    json.dump(old | out, open(os.path.join(HERE, "results.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
