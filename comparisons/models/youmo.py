"""Youmo IME's sentence model (sentence-model-large.safetensors, 25M, chinese-ime-lm format) in PyTorch, scored as its
SentenceModel.Scorer does: <bos> + the context's last (window - 1 - 24) characters, then the candidate's log P summed
over its characters. Only the context's last line is given, without [URL]: line breaks and [URL] are not in its
vocabulary, and its input method passes the text before the cursor on the same line.
"""
import json
import struct

import torch
import torch.nn.functional as F

PAD, UNK, BOS = 0, 1, 2
ROOM = 24


def clean(context):
    return context.replace("[URL]", "").split("\n")[-1]


def load_safetensors(path):
    raw = open(path, "rb").read()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + n])
    meta = header.pop("__metadata__")
    base = 8 + n
    dt = {"F32": torch.float32, "F16": torch.float16, "I8": torch.int8}
    tensors = {}
    for name, info in header.items():
        a, b = info["data_offsets"]
        tensors[name] = torch.frombuffer(bytearray(raw[base + a:base + b]), dtype=dt[info["dtype"]]).reshape(info["shape"])
    return meta, tensors


class Youmo:
    def __init__(self, path, device="cuda"):
        meta, t = load_safetensors(path)
        self.cfg = json.loads(meta["config"])
        vocab = json.loads(meta["vocab"])
        self.index = {ch: i for i, ch in enumerate(vocab) if len(ch) == 1 and i > 2}
        self.device = device

        def w(name):
            x = t[name]
            if x.dtype == torch.int8:
                x = x.float() * t[name + ".scale"].float()[:, None]
            return x.float().to(device)

        def opt(name):
            return t[name].float().to(device) if name in t else None

        self.tok = w("tok.weight")
        self.pos = t["pos.weight"].float().to(device)
        self.blocks = []
        for i in range(self.cfg["n_layer"]):
            p = f"blocks.{i}."
            self.blocks.append({k: (w(p + k + ".weight"), opt(p + k + ".bias")) for k in ("qkv", "proj", "fc", "out")}
                               | {k: (opt(p + k + ".weight"), opt(p + k + ".bias")) for k in ("ln1", "ln2")})
        self.ln_f = (opt("ln_f.weight"), opt("ln_f.bias"))
        self.n = self.cfg["n_embd"]
        self.h = self.cfg["n_head"]

    def encode(self, text):
        return [self.index.get(ch, UNK) for ch in text]

    @torch.no_grad()
    def hidden(self, ids):
        B, T = ids.shape
        x = self.tok[ids] + self.pos[:T][None]
        for b in self.blocks:
            y = F.layer_norm(x, (self.n,), *b["ln1"], eps=1e-5)
            q, k, v = F.linear(y, *b["qkv"]).split(self.n, dim=-1)
            q, k, v = (z.view(B, T, self.h, -1).transpose(1, 2) for z in (q, k, v))
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + F.linear(y.transpose(1, 2).reshape(B, T, self.n), *b["proj"])
            y = F.layer_norm(x, (self.n,), *b["ln2"], eps=1e-5)
            x = x + F.linear(F.gelu(F.linear(y, *b["fc"])), *b["out"])
        return F.layer_norm(x, (self.n,), *self.ln_f, eps=1e-5)

    @torch.no_grad()
    def score_rows(self, rows, bs=256):
        """rows: (context, [candidates]) -> summed log P per candidate."""
        window = self.cfg["context"]
        max_ctx = window - 1 - ROOM
        seqs = []  # (row, cand, ids fed, first predicting position, targets)
        for r, (ctx, cands) in enumerate(rows):
            c = self.encode(clean(ctx))
            prefix = [BOS] + c[max(0, len(c) - max_ctx):]
            for j, cand in enumerate(cands):
                t = self.encode(cand)[:window - len(prefix) + 1]
                seqs.append((r, j, prefix + t[:-1], len(prefix) - 1, t))
        out = [[float("-inf")] * len(c) for _, c in rows]
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][2]))
        for s in range(0, len(order), bs):
            chunk = [seqs[i] for i in order[s:s + bs]]
            T = -(-max(len(x[2]) for x in chunk) // 16) * 16  # few shapes: Windows' CUDA cache otherwise spills
            if (s // bs) % 50 == 49:
                torch.cuda.empty_cache()
            ids = torch.tensor([x[2] + [PAD] * (T - len(x[2])) for x in chunk], device=self.device)
            h = self.hidden(ids)
            bi, pi, tg = [], [], []
            for b, (_, _, _, p0, t) in enumerate(chunk):
                for k, tok in enumerate(t):
                    bi.append(b)
                    pi.append(p0 + k)
                    tg.append(tok)
            if not tg:
                continue
            logits = h[torch.tensor(bi, device=self.device), torch.tensor(pi, device=self.device)] @ self.tok.T
            tgt = torch.tensor(tg, device=self.device)
            lp = (logits.gather(1, tgt[:, None])[:, 0] - torch.logsumexp(logits, -1)).cpu()
            sums = torch.zeros(len(chunk)).index_add_(0, torch.tensor(bi), lp).tolist()
            for b, (r, j, _, _, t) in enumerate(chunk):
                if t:
                    out[r][j] = sums[b]
        return out
