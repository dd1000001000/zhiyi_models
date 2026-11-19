"""WowType (Gloridust/WowType) 整句增强模型 nnlm.bin, read from the file as its Rust engine runs it (core/src/nnlm.rs at
bb2a04f); nothing of WowType is run. WTNN v1: a 3-layer GPT-2, width 256, 4 heads, feed-forward 1024, 512 positions,
pre-LN, tanh GELU, the output tied to the character embeddings, float16 weights (computed here in float32 on the CPU).
Context = <s> + the last <= 24 non-space characters of the context's last line (context_tokens); a candidate's score =
the sum of ln P per character (score_tokens; a character outside its 6,998 gets <unk>). In WowType it reranks the
beam's top 8 sentences with weight 1.5 (with no context only for inputs of 5+ syllables); here it is scored alone.
  m = WowtypeNnlm("nnlm.bin"); m.score(context, cands)
"""
import struct

import numpy as np
import torch

D, HEADS, FF, LAYERS, MAX_POS = 256, 4, 1024, 3, 512
UNK, BOS = 1, 2
MAX_CONTEXT = 24


class WowtypeNnlm:
    def __init__(self, path, threads=4):
        torch.set_num_threads(threads)
        d = open(path, "rb").read()
        assert d[:4] == b"WTNN" and struct.unpack_from("<I", d, 4)[0] == 1
        ml = struct.unpack_from("<I", d, 8)[0]
        self.meta = d[12:12 + ml].decode("utf-8")
        at = 12 + ml
        v, n = struct.unpack_from("<II", d, at)
        at += 8
        pairs = np.frombuffer(d, "<u4", 2 * n, at).reshape(n, 2)
        at += 8 * n
        self.vocab = {chr(cp): int(i) for cp, i in pairs}
        self.v = v

        def take(*shape):
            nonlocal at
            k = int(np.prod(shape))
            x = np.frombuffer(d, "<f2", k, at).astype(np.float32).reshape(shape)
            at += 2 * k
            return torch.from_numpy(x.copy())
        self.wte, self.wpe = take(v, D), take(MAX_POS, D)
        self.layers = []
        for _ in range(LAYERS):
            self.layers.append(dict(ln1_g=take(D), ln1_b=take(D), attn_w=take(3 * D, D), attn_b=take(3 * D),
                                    proj_w=take(D, D), proj_b=take(D), ln2_g=take(D), ln2_b=take(D),
                                    fc_w=take(FF, D), fc_b=take(FF), out_w=take(D, FF), out_b=take(D)))
        self.lnf_g, self.lnf_b = take(D), take(D)
        assert at == len(d), (at, len(d))

    def token(self, ch):
        return self.vocab.get(ch, UNK)

    def context_tokens(self, context):
        tail = (context or "").split("\n")[-1].lstrip()
        chars = [c for c in tail if not c.isspace()]
        return [BOS] + [self.token(c) for c in chars[-MAX_CONTEXT:]]

    @torch.no_grad()
    def logprobs(self, seqs):
        """seqs: equal-length token lists -> log-softmax [B, T, V]."""
        x = torch.tensor(seqs)
        b, t = x.shape
        h = self.wte[x] + self.wpe[:t]
        mask = torch.full((t, t), float("-inf")).triu(1)
        F = torch.nn.functional
        for L in self.layers:
            a = F.layer_norm(h, (D,), L["ln1_g"], L["ln1_b"], 1e-5)
            q, k, vv = (a @ L["attn_w"].T + L["attn_b"]).split(D, -1)
            q, k, vv = (z.view(b, t, HEADS, D // HEADS).transpose(1, 2) for z in (q, k, vv))
            att = torch.softmax(q @ k.transpose(-1, -2) * 0.125 + mask, -1)
            o = (att @ vv).transpose(1, 2).reshape(b, t, D)
            h = h + o @ L["proj_w"].T + L["proj_b"]
            a = F.layer_norm(h, (D,), L["ln2_g"], L["ln2_b"], 1e-5)
            h = h + F.gelu(a @ L["fc_w"].T + L["fc_b"], approximate="tanh") @ L["out_w"].T + L["out_b"]
        y = F.layer_norm(h, (D,), self.lnf_g, self.lnf_b, 1e-5)
        return torch.log_softmax(y @ self.wte.T, -1)

    def score(self, context, cands):
        pre = self.context_tokens(context)
        toks = [[self.token(c) for c in cand if not c.isspace()] for cand in cands]
        w = max(map(len, toks))
        seqs = [pre + t + [0] * (w - len(t)) for t in toks]
        lp = self.logprobs(seqs)
        out = []
        for i, t in enumerate(toks):
            out.append(float(sum(lp[i, len(pre) - 1 + j, tok] for j, tok in enumerate(t))) if t else 0.0)
        return out
