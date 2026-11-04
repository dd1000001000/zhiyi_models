"""Word-choice models of several input methods on the neutral word-choice test sets v2 and v3 (web), v4 (chat) and v5
(the hard set), each model alone.
Every model gets the same rows and the same candidates; the pick is the best-scoring of the row's whole candidates
(those that spell the whole input), right when it is the row's best. Scores go to scores/<model>_neutral-<set>.jsonl
({"id", "scores"} as the sets' score.py reads, null for candidates not scored), accuracies to results.json.

  python run.py --v2 neutral-v2.jsonl --v3 neutral-v3.jsonl [--v4 neutral-v4.jsonl --v5 neutral-v5.jsonl]
                --zhiyi zhiyi-pumpkin-1026=<unpacked laya-ml-pumpkin-1026-int8g.zip> --zhiyi zhiyi-flash-1031=<...>
                --metasequoia <rerank-v1 folder> --youmo sentence-model-large.safetensors
                --chaoinput <ChaoInput-v3.3.16-win64/data>,<ChaoInput-src-v3.3.16/src/ChaoInput.LmTrain.Gpu/model.py>
                --qwen <Qwen3.5-2B-Base folder> --pinyingpt <pinyingpt-concat folder> --gpt2-zh-21k <gpt2-zh-21k folder>
                [--pinyingpt-typed <pinyingpt-concat folder>]  (PinyinGPT fed the pinyin as typed: a side check)
                --metaseq-v1-4m sentence-model.safetensors --metaseq-v1-25m sentence-model-desktop.safetensors
                --rime-ai-ime <Qwen3.5-0.8B>,<adapter_v1> --fujik0 <Qwen3.5-0.8B> --lime <Qwen3-0.6B>
                --cnmbert <CNMBert folder> --qingjian <qj-score exe>,<tongbian .qjm>,<zhiwei .qjm>
                --zhizi <shunshou-zhizi-30m folder>
Give only the models to run; score files already in scores/ are reused (remove one to score it again), and every
score file there is counted in results.json. Zhiyi's graphs run on DirectML (onnxruntime-directml, --cpu for CPU);
the PyTorch models on CUDA; MetasequoiaIME and Zhizi on CPU.
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
          "two": lambda r: r["kind"] == "two", "long": lambda r: r["kind"] == "long",
          "no context": lambda r: not r["context"], "context 1-32": lambda r: 0 < len(r["context"]) <= 32,
          "context 33+": lambda r: len(r["context"]) > 32}


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
    if a.zhizi:
        def zhizi(rows):
            from zhizi30m import Zhizi
            m = Zhizi(a.zhizi)
            return whole_only(rows, m.score_rows, lambda r: (r["context"], cands(r)))
        out["zhizi-30m"] = zhizi
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
    for key, name in (("metaseq_v1_4m", "metasequoia-v1-4m"), ("metaseq_v1_25m", "metasequoia-v1-25m")):
        path = getattr(a, key)
        if path:
            def ms1(rows, path=path):
                from metaseq_v1 import MetaseqSentence
                m = MetaseqSentence(path)
                return whole_only(rows, m.score_rows, lambda r: (r["context"], cands(r)))
            out[name] = ms1
    for key, name, style in (("rime_ai_ime", "rime-ai-ime-0.8b-lora", "rime-ai-ime"),
                             ("fujik0", "fujik0-qwen3.5-0.8b", "fujik0")):
        spec = getattr(a, key)
        if spec:
            def judge(rows, spec=spec, style=style):
                from chat_judge import ChatJudge
                base, _, adapter = spec.partition(",")
                m = ChatJudge(base, style, adapter or None)
                return whole_only(rows, m.score_rows, lambda r: (r["context"], r["code"], cands(r)))
            out[name] = judge
    if a.lime:
        def lime(rows):
            from qwen import LIME_PREFIX, Qwen
            m = Qwen(a.lime, prefix=LIME_PREFIX, start=None)
            return whole_only(rows, m.score_rows, lambda r: (r["context"], cands(r)))
        out["lime-qwen3-0.6b"] = lime
    if a.cnmbert:
        def cnm(rows):
            from cnmbert import CNMBert
            m = CNMBert(a.cnmbert)
            ini = [r for r in rows if r["style"] == "initials"]
            got = whole_only(ini, m.score_rows, lambda r: (r["context"], [
                (r["candidates"][j], "".join(s[0] for s, _ in r["readings"][j])) for j in r["whole"]]))
            return {r["id"]: got.get(r["id"], [None] * len(r["candidates"])) for r in rows}
        out["cnmbert"] = cnm
    if a.qingjian:
        exe, tongbian, zhiwei = a.qingjian.split(",")
        for name, mode, model in (("qingjian-tongbian", "p2c", tongbian), ("qingjian-zhiwei", "char", zhiwei)):
            def qj(rows, mode=mode, model=model):
                import qingjian
                return whole_only(rows, lambda items: qingjian.score_rows(exe, mode, model, items),
                                  lambda r: (r["context"], r["code"], cands(r)))
            out[name] = qj
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
    p.add_argument("--v4", help="neutral-v4.jsonl (chat)")
    p.add_argument("--v5", help="neutral-v5.jsonl (the hard set)")
    p.add_argument("--zhiyi", action="append", help="name=folder of an unpacked int8 release zip (repeatable)")
    for k in ("metasequoia", "youmo", "chaoinput", "qwen", "pinyingpt", "pinyingpt-typed", "gpt2-zh-21k",
              "metaseq-v1-4m", "metaseq-v1-25m", "lime", "cnmbert", "zhizi"):
        p.add_argument("--" + k)
    p.add_argument("--rime-ai-ime", help="<Qwen3.5-0.8B folder>,<adapter_v1 folder>")
    p.add_argument("--fujik0", help="<Qwen3.5-0.8B folder>")
    p.add_argument("--qingjian", help="<qj-score exe>,<hanzhang-tongbian-small.qjm>,<hanzhang-zhiwei-small.qjm>")
    p.add_argument("--cpu", action="store_true", help="Zhiyi's graphs on the CPU instead of DirectML")
    p.add_argument("--context_chars", type=int, help="Zhiyi: the context's last N characters only (named <name>-ctxN)")
    a = p.parse_args()
    os.makedirs(os.path.join(HERE, "scores"), exist_ok=True)
    out = {}
    for v, data in (("v2", a.v2), ("v3", a.v3), ("v4", a.v4), ("v5", a.v5)):
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
                if h:
                    out[v][name][g] = {"n": len(h), "acc": 100 * sum(h) / len(h)}
            print(f"{v} {name:<28}" + " ".join(f"{g} {x['acc']:.1f}" for g, x in out[v][name].items()), flush=True)
    old = json.load(open(os.path.join(HERE, "results.json"), encoding="utf-8")) if os.path.exists(
        os.path.join(HERE, "results.json")) else {}
    json.dump(old | out, open(os.path.join(HERE, "results.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
