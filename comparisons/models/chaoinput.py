"""ChaoInput (吵吵拼音) CLMG v2 reranker (lmt.bin, 6 layers x 512, 28.7M) with its dictionary prior (dict.db),
reimplemented from its source (ChaoInput.LmTrain.Gpu/model.py, read, not run):
  h = the encoder's last state over [BOS] + the context's last 64 characters
  score = log(10 + weight of (full pinyin, text) in dict.db) + h . mean(head rows of the candidate's characters)
as its engine ranks a pinyin key's pool. A candidate outside its dictionary gets weight 0.
"""
import ast
import math
import os
import sqlite3
import struct

import numpy as np
import torch
import torch.nn.functional as Fn

L, H, HEADS, F, V, MAX_SEQ = 6, 512, 8, 2048, 9537, 96
K_BOS, K_LETTER_BASE, K_UNK, K_CHAR_BASE, K_CTX_MAX = 1, 5, 31, 32, 64


def vocab(model_py):
    """_ID2S_HEAD + _ID2S_CHARS from its model.py, read as literals (the file is not executed)."""
    tree = ast.parse(open(model_py, encoding="utf-8").read())
    vals = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in (
                "_ID2S_HEAD", "_ID2S_CHARS"):
            vals[node.targets[0].id] = ast.literal_eval(node.value)
    ids = vals["_ID2S_HEAD"] + list(vals["_ID2S_CHARS"])
    assert len(ids) == V
    return {s: i for i, s in enumerate(ids)}


class Chao:
    def __init__(self, data_dir, model_py, device="cuda"):
        """data_dir: the win64 package's data folder (lmt.bin, dict.db); model_py: its source's model.py."""
        raw = open(os.path.join(data_dir, "lmt.bin"), "rb").read()
        hdr = struct.unpack("<4s7I5Q", raw[:72])
        assert hdr[:8] == (b"CLMG", 2, L, H, HEADS, F, V, MAX_SEQ) and hdr[11] * 4 == len(raw) - 72, hdr
        flat = torch.from_numpy(np.frombuffer(raw[72:], dtype=np.float32).copy())
        off = 0

        def take(*shape):
            nonlocal off
            n = math.prod(shape)
            t = flat[off:off + n].view(*shape).to(device)
            off += n
            return t
        self.emb, self.pos, self.lnF = take(V, H), take(MAX_SEQ, H), (take(H), take(H))
        self.layers = []
        for _ in range(L):
            lay = {}
            for k, shape in (("ln1_g", (H,)), ("ln1_b", (H,)), ("qw", (H, H)), ("qb", (H,)), ("kw", (H, H)), ("kb", (H,)),
                             ("vw", (H, H)), ("vb", (H,)), ("ow", (H, H)), ("ob", (H,)), ("ln2_g", (H,)), ("ln2_b", (H,)),
                             ("f1w", (H, F)), ("f1b", (F,)), ("f2w", (F, H)), ("f2b", (H,))):
                lay[k] = take(*shape)
            self.layers.append(lay)
        self.head = take(V, H)
        assert off == flat.numel()
        self.device = device
        self.s2id = vocab(model_py)
        db = sqlite3.connect("file:%s?mode=ro" % os.path.join(data_dir, "dict.db"), uri=True)
        self.prior = {}
        for py, text, w in db.execute("SELECT py, text, weight FROM dict"):  # the dict table only (not userlog)
            key = ("".join(c for c in py.lower() if "a" <= c <= "z"), text)
            self.prior.setdefault(key, math.log(10.0 + max(0.0, float(w or 0))))
        db.close()

    def encode_ctx(self, s):
        out = []
        for c in s or "":
            cp = ord(c)
            if 0x4E00 <= cp <= 0x9FFF:
                out.append(self.s2id.get(c, K_UNK))
            elif 0x61 <= cp <= 0x7A:
                out.append(K_LETTER_BASE + cp - 0x61)
            elif 0x41 <= cp <= 0x5A:
                out.append(K_LETTER_BASE + cp - 0x41)
            else:
                i = self.s2id.get(c)
                if i is not None and i >= K_CHAR_BASE:
                    out.append(i)
        return out[-K_CTX_MAX:]

    @torch.no_grad()
    def hidden(self, contexts):
        """[BOS] + context ids, right-padded, causal: each row's last real position."""
        seqs = [[K_BOS] + self.encode_ctx(c) for c in contexts]
        T = max(map(len, seqs))
        ids = torch.zeros(len(seqs), T, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
        ids = ids.to(self.device)
        B = ids.shape[0]
        hd = H // HEADS
        x = self.emb[ids] + self.pos[:T].unsqueeze(0)
        mask = torch.triu(torch.full((T, T), float("-inf"), device=self.device), 1)
        for lay in self.layers:
            y = Fn.layer_norm(x, (H,), lay["ln1_g"], lay["ln1_b"], 1e-5)
            q, k, v = ((y @ lay[w] + lay[b]).view(B, T, HEADS, hd).transpose(1, 2)
                       for w, b in (("qw", "qb"), ("kw", "kb"), ("vw", "vb")))
            p = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(hd) + mask, dim=-1)
            x = x + (p @ v).transpose(1, 2).reshape(B, T, H) @ lay["ow"] + lay["ob"]
            y = Fn.layer_norm(x, (H,), lay["ln2_g"], lay["ln2_b"], 1e-5)
            x = x + Fn.gelu(y @ lay["f1w"] + lay["f1b"], approximate="tanh") @ lay["f2w"] + lay["f2b"]
        x = Fn.layer_norm(x, (H,), *self.lnF, 1e-5)
        last = torch.tensor([len(s) - 1 for s in seqs], device=self.device)
        return x[torch.arange(B, device=self.device), last]

    def residual(self, h, text):
        ids = [self.s2id[c] for c in text if self.s2id.get(c, -1) >= K_CHAR_BASE]
        if not ids:
            return 0.0
        return float(h @ self.head[torch.tensor(ids, device=self.device)].mean(0))

    def score_rows(self, rows):
        """rows: (context, [candidates], [full pinyin per candidate]) -> prior + model per candidate."""
        out = []
        for s in range(0, len(rows), 256):
            chunk = rows[s:s + 256]
            for (_, cands, pys), h in zip(chunk, self.hidden([c for c, _, _ in chunk])):
                out.append([self.prior.get((py, c), math.log(10.0)) + self.residual(h, c) for c, py in zip(cands, pys)])
        return out
