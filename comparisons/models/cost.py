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
         --metaseq-v1-4m <file> --metaseq-v1-25m <file> --rime-ai-ime <Qwen3.5-0.8B>,<qwen35-08b-ime-v1-Q8_0.gguf>
                 --fujik0 <Qwen3.5-0.8B> --lime <Qwen3-0.6B>,367804096 --cnmbert <folder> --qingjian <tongbian>,<zhiwei>
                 --zhizi <shunshou-zhizi-30m folder>
-> cost.json
New models' file sizes are what each project runs: Rime-AI-IME its Q8_0 GGUF, FuJiK0 the bf16 Qwen3.5-0.8B (its MLX
bf16 default), LIME the IQ4_XS GGUF of Qwen3-0.6B (Hugging Face unsloth/Qwen3-0.6B-GGUF), CNMBert the fp32
safetensors, Qingjian the f16 .qjm, Zhizi its fp32 ONNX (Windows; macOS runs a Core ML fp16 copy). Rime-AI-IME / FuJiK0: the prompt (with the candidate list) once, then each answer;
CNMBert (initials rows only): one forward per character of each candidate, as its left-to-right fill.
"""
import argparse
import json
import math
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
    for k in ("metasequoia", "youmo", "chaoinput", "qwen", "pinyingpt", "gpt2-zh-21k", "metaseq-v1-4m", "metaseq-v1-25m",
              "cnmbert", "zhizi"):
        p.add_argument("--" + k)
    p.add_argument("--rime-ai-ime", help="<Qwen3.5-0.8B folder>,<its released GGUF, for the file size>")
    p.add_argument("--fujik0", help="<Qwen3.5-0.8B folder>")
    p.add_argument("--lime", help="<Qwen3-0.6B folder>[,<size in bytes of the IQ4_XS GGUF LIME runs>]")
    p.add_argument("--qingjian", help="<hanzhang-tongbian-small.qjm>,<hanzhang-zhiwei-small.qjm>")
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

    if a.zhizi:
        from zhizi30m import KEEP, clean
        d, n, V, P = 512, 8, 7745, 128  # its ONNX keeps a second copy of the tied output matrix: counted once
        W = n * 12 * d * d
        per = []
        for r, cands in zip(rows, whole):
            ctx = 1 + min(len(clean(r["context"])), KEEP, 39 - max(map(len, cands)))
            per.append(causal(ctx, [(len(c) - 1, len(c)) for c in cands], W, d, n, V * d))
        add("zhizi-30m", os.path.getsize(os.path.join(a.zhizi, "zhizi30m_v2.onnx")),
            V * d + P * d + n * (12 * d * d + 13 * d) + 2 * d, per)

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

    for key, name in (("metaseq_v1_4m", "metasequoia-v1-4m"), ("metaseq_v1_25m", "metasequoia-v1-25m")):
        path = getattr(a, key)
        if not path:
            continue
        from youmo import load_safetensors
        meta, t = load_safetensors(path)
        cc = json.loads(meta["config"])
        d, n, V = cc["n_embd"], cc["n_layer"], cc["vocab"]
        W = n * 12 * d * d
        per = []
        for r, cands in zip(rows, whole):
            room = max(0, cc["context"] - (max(map(len, cands)) + 1))
            per.append(causal(1 + min(len(r["context"]), room), [(len(c) - 1, len(c)) for c in cands], W, d, n, V * d))
        add(name, os.path.getsize(path), sum(v.numel() for k, v in t.items() if not k.endswith(".scale")), per)

    def lm_params(folder):
        from safetensors import safe_open
        lm = emb = 0
        for fn in [os.path.join(folder, x) for x in os.listdir(folder) if x.endswith(".safetensors")]:
            with safe_open(fn, "pt") as f:
                for k in f.keys():
                    if "visual" in k or "mtp" in k or "lm_head" in k:  # lm_head: tied to embed_tokens (Qwen3 stores a copy)
                        continue
                    c = math.prod(f.get_slice(k).get_shape())
                    lm += c
                    emb += c if "embed_tokens" in k else 0
        cfg = json.load(open(os.path.join(folder, "config.json"), encoding="utf-8"))
        return lm, emb, cfg.get("text_config", cfg)

    for key, name, style in (("rime_ai_ime", "rime-ai-ime-0.8b-lora", "rime-ai-ime"), ("fujik0", "fujik0-qwen3.5-0.8b", "fujik0")):
        spec = getattr(a, key)
        if not spec:
            continue
        base, _, size_file = spec.partition(",")
        from transformers import AutoTokenizer
        from chat_judge import PROMPTS, CONTEXT_CHARS
        tok = AutoTokenizer.from_pretrained(base)
        lm, emb, tc = lm_params(base)
        d, V = tc["hidden_size"], tc["vocab_size"]
        full = sum(1 for k in tc["layer_types"] if k == "full_attention")
        qd = tc["num_attention_heads"] * tc["head_dim"]
        p = PROMPTS[style]
        per = []
        for r, cands in zip(rows, whole):
            ctx = r["context"][-CONTEXT_CHARS:]
            user = (p["with"] if ctx else p["without"]).format(ctx=ctx, py=r["code"], cands="、".join(cands))
            head = len(tok.apply_chat_template([{"role": "system", "content": p["system"]}, {"role": "user", "content": user}],
                                               tokenize=True, add_generation_prompt=True, enable_thinking=False))
            ans = [len(tok(p["answer"].format(w=c), add_special_tokens=False).input_ids) for c in cands]
            per.append(causal(head, [(k, k) for k in ans], lm - emb, qd, full, V * d))
        add(name, os.path.getsize(size_file or os.path.join(base, os.listdir(base)[0])), lm, per)

    if a.lime:
        folder, _, gguf_size = a.lime.partition(",")
        from transformers import AutoTokenizer
        from qwen import LIME_PREFIX
        tok = AutoTokenizer.from_pretrained(folder)
        lm, emb, tc = lm_params(folder)
        d, V = tc["hidden_size"], tc["vocab_size"]
        qd = tc["num_attention_heads"] * tc["head_dim"]
        per = []
        for r, cands in zip(rows, whole):
            base = tok(LIME_PREFIX + r["context"], add_special_tokens=False).input_ids
            news = []
            for c in cands:
                fullc = tok(LIME_PREFIX + r["context"] + c, add_special_tokens=False).input_ids
                k = next((i for i, (x, y) in enumerate(zip(base, fullc)) if x != y), min(len(base), len(fullc)))
                news.append((len(fullc) - k, len(fullc) - k))
            per.append(causal(len(base), news, lm - emb, qd, tc["num_hidden_layers"], V * d))
        add("lime-qwen3-0.6b", int(gguf_size) if gguf_size else os.path.getsize(os.path.join(folder, "model.safetensors")),
            lm, per)

    if a.cnmbert:
        from safetensors import safe_open
        from transformers import BertTokenizer
        f = os.path.join(a.cnmbert, "model.safetensors")
        with safe_open(f, "pt") as st:
            shapes = {k: st.get_slice(k).get_shape() for k in st.keys()}
        params = sum(math.prod(s) for s in shapes.values())
        cfg = json.load(open(os.path.join(a.cnmbert, "config.json"), encoding="utf-8"))
        d, n, V = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["vocab_size"]
        W = n * (4 * d * d + 2 * d * cfg["intermediate_size"])
        tok = BertTokenizer.from_pretrained(a.cnmbert)
        per = []
        for r, cands in zip(rows, whole):
            if r["style"] != "initials":
                continue
            m = max(map(len, cands))
            L = 2 + m + min(len(tok.encode(r["context"], add_special_tokens=False)), 62 - m)
            per.append(sum(len(c) * (W * L + n * 2 * L * L * d + d * d + V * d) for c in cands))
        add("cnmbert", os.path.getsize(f), params, per)

    if a.qingjian:
        tongbian, zhiwei = a.qingjian.split(",")
        for name, path, p2c in (("qingjian-tongbian", tongbian, True), ("qingjian-zhiwei", zhiwei, False)):
            raw = open(path, "rb").read()
            i = raw.index(b'{"tokens"')
            V = len(json.JSONDecoder().raw_decode(raw[i:].decode("utf-8", "ignore"))[0]["tokens"])
            d, n, ctxlen = 448, 8, 128
            W = n * 12 * d * d
            per = []
            for r, cands in zip(rows, whole):
                if p2c:
                    per.append(causal(1 + len(r["code"]), [(len(c), len(c)) for c in cands], W, d, n, V * d))
                else:
                    head = min(1 + len(r["context"]), ctxlen - max(map(len, cands)) - 1)
                    per.append(causal(head, [(len(c), len(c)) for c in cands], W, d, n, V * d))
            add(name, len(raw), V * d + ctxlen * d + n * (12 * d * d + 13 * d) + 2 * d, per)

    path = os.path.join(HERE, "cost.json")
    old = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    json.dump(old | out, open(path, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
