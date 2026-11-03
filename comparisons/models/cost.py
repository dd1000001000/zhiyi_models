"""Size, parameters and theoretical compute of each model, for the chart (relative to Zhiyi ML-Flash-1031 = 1.0).

- File size: the model file as released (Zhiyi: laya.int8g.onnx; MetasequoiaIME: <name>.ort; Youmo: the
  safetensors; ChaoInput: lmt.bin (its dict.db not counted); PinyinGPT / GPT2-zh-21k: the released weights;
  Qwen: the safetensors, vision tower included).
- Parameters: the tensors in that file (Zhiyi: the ONNX initializers, scalars left out; the ones read from configs
  where the format is not readable: MetasequoiaIME); Qwen: the language model only.
- Theoretical compute: multiply-adds to score one row's candidates from scratch, averaged over neutral-v3's scored
  rows: the context once (causal models keep its keys and values for every candidate), every token through the
  linear layers, attention over what each token sees, and the output layer where the model computes it: a softmax
  over its vocabulary for every character scored (MetasequoiaIME 6x256 is self-normalized: its candidate's logit
  alone). Zhiyi: one pass over the whole page (context, slots, every candidate), encoder + choice head; its
  composition head is not counted.

  python cost.py --v3 neutral-v3.jsonl --zhiyi name=<folder> ... --metasequoia <rerank-v1 folder> --youmo <file>
                 --chaoinput <data dir>,<model.py> --qwen <folder> --pinyingpt <folder> --gpt2-zh-21k <folder>
-> cost.json
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


# the released graphs carry no encoder config: the attention kinds of the kept layers (mmBERT-small pruned), by depth
LAYER_TYPES = {12: "FSSFSSFSSSSF", 8: "FSFSFSSF"}
D, FF, HEAD_FF, WINDOW = 384, 1152, 1536, 128


def causal(ctx, cands, W, d, layers, out):
    """ctx tokens once, then per candidate (new tokens, characters scored): linear + attention + output."""
    macs = W * ctx + layers * 2 * d * sum(range(1, ctx + 1))
    for new, scored in cands:
        macs += W * new + layers * 2 * d * sum(range(ctx + 1, ctx + new + 1)) + out * scored
    return macs


def onnx_params(path):
    import onnx
    m = onnx.load(path)
    total = 0
    for i in m.graph.initializer:
        n = 1
        for x in i.dims:
            n *= x
        total += n if n > 1 else 0
    return total


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--v3", required=True)
    p.add_argument("--zhiyi", action="append")
    for k in ("metasequoia", "youmo", "chaoinput", "qwen", "pinyingpt", "gpt2-zh-21k"):
        p.add_argument("--" + k)
    a = p.parse_args()
    rows = [r for r in map(json.loads, open(a.v3, encoding="utf-8")) if r["scored"]]
    whole = [[r["candidates"][j] for j in r["whole"]] for r in rows]
    out = {}

    def add(name, size, params, per_row):
        out[name] = {"file_mb": size / 1e6, "params": params, "macs_per_row": sum(per_row) / len(per_row)}
        print(f"{name:<28} {size / 1e6:8.1f} MB {params / 1e6:8.1f} M params {out[name]['macs_per_row'] / 1e6:10.1f} M MACs/row",
              flush=True)

    for spec in a.zhiyi or []:
        name, folder = spec.split("=", 1)
        import onnx
        from zhiyi import Zhiyi, units_for
        z = Zhiyi(folder, None)
        g = os.path.join(folder, "laya.int8g.onnx")
        depth = sum(1 for i in onnx.load(g, load_external_data=False).graph.initializer if list(i.dims) == [D, 2 * FF])
        kinds = LAYER_TYPES[depth]
        d = D
        enc = 3 * d * d + d * d + d * 2 * FF + FF * d  # Wqkv, Wo, Wi (gated), Wo
        head = 2 * (3 * d * d + d * d + 2 * d * HEAD_FF)  # the choice head: two layers, attention + feed-forward
        per = []
        for r in rows:
            code = "".join(c for c in r["code"] if "a" <= c <= "z")
            L = len(z.sequence(r["context"], r["candidates"], units_for(r["candidates"][0], code) or [])[0])
            att = sum(2 * L * (L if k == "F" else min(L, WINDOW + 1)) * d for k in kinds)
            per.append(L * (depth * enc + head) + att + 2 * 2 * L * L * d)
        add(name, os.path.getsize(g), onnx_params(g), per)

    if a.metasequoia:
        for size in ("8x448", "6x256"):
            c = json.load(open(os.path.join(a.metasequoia, f"rerank-{size}.model.json"), encoding="utf-8"))
            cc = c["config"]
            d, n, V = cc["n_embd"], cc["n_layer"], cc["vocab"]
            W = n * 12 * d * d
            params = V * d + cc["max_positions"] * d + n * (12 * d * d + 13 * d)
            per = []
            for r, cands in zip(rows, whole):
                ctx = 1 + min(len(r["context"]), c["context_chars"])
                nodes = {c_[:t] for c_ in cands for t in range(1, len(c_))}  # its runtime shares the candidates' prefixes
                per.append(causal(ctx, [(len(nodes), 0)], W, d, n, 0)
                           + sum(map(len, cands)) * (d if cc["self_normalized"] else V * d))
            add(f"metasequoia-rerank-{size}", os.path.getsize(os.path.join(a.metasequoia, f"rerank-{size}.ort")), params, per)

    if a.youmo:
        from youmo import ROOM, clean, load_safetensors
        meta, t = load_safetensors(a.youmo)
        cc = json.loads(meta["config"])
        d, n, V = cc["n_embd"], cc["n_layer"], cc["vocab"]
        W = n * 12 * d * d
        per = [causal(1 + min(len(clean(r["context"])), cc["context"] - 1 - ROOM), [(len(c) - 1, len(c)) for c in cands],
                      W, d, n, V * d) for r, cands in zip(rows, whole)]
        add("youmo-25m", os.path.getsize(a.youmo), sum(v.numel() for k, v in t.items() if not k.endswith(".scale")), per)

    if a.chaoinput:
        import chaoinput as C
        data, _ = a.chaoinput.split(",")
        W = C.L * (4 * C.H * C.H + 2 * C.H * C.F)
        per = []
        for r, cands in zip(rows, whole):
            ctx = 1 + min(len(r["context"]), C.K_CTX_MAX)
            per.append(causal(ctx, [], W, C.H, C.L, 0) + C.H * sum(map(len, cands)))
        params = C.V * C.H * 2 + C.MAX_SEQ * C.H + C.L * (4 * C.H * C.H + 2 * C.H * C.F)
        add("chaoinput", os.path.getsize(os.path.join(data, "lmt.bin")), params, per)

    for key, name, concat in (("pinyingpt", "pinyingpt-concat", True), ("gpt2_zh_21k", "gpt2-zh-21k", False)):
        path = getattr(a, key)
        if not path:
            continue
        from transformers import BertTokenizer
        tok = BertTokenizer.from_pretrained(path)
        f = os.path.join(path, "pytorch_model.bin" if concat else "model.safetensors")
        V = 21571 if concat else 21128
        d, n = 768, 12
        W = n * 12 * d * d
        from pinyingpt import CONTEXT_CHARS
        per = []
        for r, cands in zip(rows, whole):
            ctx = 1 + len(tok.encode(r["context"][-CONTEXT_CHARS:], add_special_tokens=False)) + (1 if concat else 0)
            per.append(causal(ctx, [((2 * len(c)) if concat else (len(c) - 1), len(c)) for c in cands], W, d, n, V * d))
        add(name, os.path.getsize(f), V * d + 1024 * d + n * (12 * d * d + 13 * d) + 2 * d, per)

    if a.qwen:
        from safetensors import safe_open
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.qwen)
        files = [os.path.join(a.qwen, x) for x in os.listdir(a.qwen) if x.endswith(".safetensors")]
        lm = emb = 0
        for fn in files:
            with safe_open(fn, "pt") as f:
                for k in f.keys():
                    if "visual" in k or "mtp" in k:
                        continue
                    c = 1
                    for x in f.get_slice(k).get_shape():
                        c *= x
                    lm += c
                    if "embed_tokens" in k:
                        emb += c
        cfg = json.load(open(os.path.join(a.qwen, "config.json"), encoding="utf-8"))
        tc = cfg.get("text_config", cfg)
        d, V = tc["hidden_size"], tc["vocab_size"]
        full = sum(1 for k in tc["layer_types"] if k == "full_attention")
        qd = tc["num_attention_heads"] * tc["head_dim"]
        W = lm - emb
        per = []
        for r, cands in zip(rows, whole):
            base = tok(r["context"], add_special_tokens=False).input_ids
            ctx = 1 + len(base)
            news = []
            for c in cands:
                fullc = tok(r["context"] + c, add_special_tokens=False).input_ids
                k = next((i for i, (x, y) in enumerate(zip(base, fullc)) if x != y), min(len(base), len(fullc)))
                news.append((len(fullc) - k, len(fullc) - k))
            per.append(causal(ctx, news, W, qd, full, V * d))
        add("qwen3.5-2b-base", sum(os.path.getsize(f) for f in files), lm, per)

    path = os.path.join(HERE, "cost.json")
    old = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    json.dump(old | out, open(path, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
