"""MetasequoiaIME rerank-v1 (rerank-8x448 / rerank-6x256: <name>.ort, <name>.model.json, <name>.vocab.json) through
ONNX Runtime, scored as its runtime does (export_step.write_golden): <bos> + the context's last 64 characters into an
empty cache, then pages of the candidates cut to 1, 2, ... characters with the prefixes cached across them, as keys are
typed; the last page's scores (mean log P per character). It is a character language model: no pinyin goes in.
CPU only (its .ort graph fails on DirectML).
"""
import json
import os

import numpy as np


def keystroke_feed(n_ctx, bodies, cached):
    """Its export_step.keystroke_feed / model.step_inputs."""
    index = {p: n_ctx + i for i, p in enumerate(cached)}
    fresh = []
    for b in bodies:
        for t in range(1, len(b)):
            p = tuple(b[:t])
            if p not in index:
                index[p] = n_ctx + len(cached) + len(fresh)
                fresh.append(p)
    every = list(cached) + fresh
    w = max(map(len, bodies))
    cands = np.zeros((len(bodies), w), np.int64)
    for i, b in enumerate(bodies):
        cands[i, :len(b)] = b
    vis = np.array([[True] * n_ctx + [len(q) <= len(p) and p[:len(q)] == q for q in every] for p in fresh],
                   bool).reshape(len(fresh), n_ctx + len(every))
    steps = [[(n_ctx - 1 if t == 0 else index[tuple(b[:t])]) if t < len(b) else 0 for t in range(w)] for b in bodies]
    return {"nodes": np.array([p[-1] for p in fresh], np.int64),
            "positions": np.array([n_ctx + len(p) - 1 for p in fresh], np.int64), "visible": vis,
            "candidates": cands, "steps": np.array(steps, np.int64),
            "lengths": np.array([len(b) for b in bodies], np.int64)}, every


class Metasequoia:
    def __init__(self, folder, name):
        import onnxruntime as ort
        self.cfg = json.load(open(os.path.join(folder, name + ".model.json"), encoding="utf-8"))
        tokens = json.load(open(os.path.join(folder, name + ".vocab.json"), encoding="utf-8"))["tokens"]
        self.ids = {t: i for i, t in enumerate(tokens) if len(t) == 1}
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        so.inter_op_num_threads = 1
        self.s = ort.InferenceSession(os.path.join(folder, name + ".ort"), so, providers=["CPUExecutionProvider"])
        c = self.cfg["config"]
        d = c["n_embd"] // c["n_head"]
        self.empty = {"cache_k": np.zeros((c["n_layer"], c["n_head"], 0, d), np.float32),
                      "cache_v": np.zeros((c["n_layer"], c["n_head"], 0, d), np.float32),
                      "cache_h": np.zeros((0, c["n_embd"]), np.float32), "cache_z": np.zeros((0,), np.float32)}

    def encode(self, text):
        return [self.ids.get(ch, self.cfg["unk"]) for ch in text]

    def context(self, text):
        ctx = [self.cfg["bos"], *self.encode(text[-self.cfg["context_chars"]:] if text else "")]
        n = len(ctx)
        feed = self.empty | {"nodes": np.array(ctx, np.int64), "positions": np.arange(n, dtype=np.int64),
                             "visible": np.tril(np.ones((n, n), bool)), "candidates": np.array([[ctx[0]]], np.int64),
                             "steps": np.zeros((1, 1), np.int64), "lengths": np.ones(1, np.int64)}
        out = self.s.run(None, feed)
        return n, dict(zip(["cache_k", "cache_v", "cache_h", "cache_z"], out[1:]))

    def score(self, context, cands):
        n, cache = self.context(context)
        bodies = [self.encode(c) for c in cands]
        cached = []
        for k in range(1, max(map(len, bodies)) + 1):
            page = []
            for b in bodies:
                if b[:k] not in page:
                    page.append(b[:k])
            feed, cached = keystroke_feed(n, page, cached)
            out = self.s.run(None, cache | feed)
            cache = dict(zip(["cache_k", "cache_v", "cache_h", "cache_z"], out[1:]))
        last = {tuple(b): s for b, s in zip(page, out[0].tolist())}
        return [last[tuple(b)] for b in bodies]

    def score_rows(self, rows):
        return [self.score(c, cands) for c, cands in rows]
