"""A word trigram ARPA model (Youmo's lm.arpa and our mixes of it) as a candidate scorer: ln P(candidate | context),
the candidate split into the model's words as the model likes best (Viterbi over the last two words, a beam).

  lm = NgramLM("path/lm.arpa")          # parsed once into path.npz beside it
  lm.score(context, ["候选一", "候选二"])  # -> [ln P, ...]

Backoff as ARPA has it; no <s> (Youmo's model has none: a text starts from no history). The context's last
CTX characters lead in: score = best(context + candidate) - best(context).
"""
import math
import os

import numpy as np

LN10 = math.log(10)
CTX = 12
MAX_WORD = 8
BEAM = 8
SHIFT = 20  # word ids below 2^20


class NgramLM:
    def __init__(self, path):
        if path.endswith(".ngram"):
            z = self._packed(path)
        else:
            cache = path + ".npz"
            if not os.path.exists(cache) or os.path.getmtime(cache) < os.path.getmtime(path):
                self._parse(path, cache)
            z = np.load(cache, allow_pickle=False)
        self.words = list(z["words"])
        self.ids = {w: i for i, w in enumerate(self.words)}
        self.p1, self.b1 = z["p1"], z["b1"]
        self.k2, self.p2, self.b2 = z["k2"], z["p2"], z["b2"]
        self.k3, self.p3 = z["k3"], z["p3"]
        self.unk = self.ids.get("<unk>", 0)
        self.longest = min(MAX_WORD, max(len(w) for w in self.words))
        self._c2, self._c3 = {}, {}

    @staticmethod
    def _packed(path):
        """The engine's .ngram (ngram_prune.py writes it) as _parse's arrays, its int16 values and all."""
        buf = open(path, "rb").read()
        assert buf[:8] == b"ZYNGRAM1"
        n1, n2, n3, _ = np.frombuffer(buf, "<u4", 4, 8)
        pos = 24

        def take(dtype, n):
            nonlocal pos
            pos += -pos % 8
            a = np.frombuffer(buf, dtype, int(n), pos)
            pos += a.nbytes
            return a
        off = take("<u4", n1 + 1)
        blob = take("u1", off[-1]).tobytes()
        words = np.array([blob[off[i]:off[i + 1]].decode("utf-8") for i in range(n1)])
        q = lambda a: a.astype(np.float32) / 2048.0  # noqa: E731
        p1, b1 = q(take("<i2", n1)), q(take("<i2", n1))
        bs = take("<u4", n1 + 1)
        bw, bp, bb = take("<u4", n2), q(take("<i2", n2)), q(take("<i2", n2))
        ts = take("<u4", n2 + 1)
        tw, tp = take("<u4", n3), q(take("<i2", n3))
        bv = np.repeat(np.arange(n1, dtype=np.int64), np.diff(bs.astype(np.int64)))
        k2 = (bv << SHIFT) | bw.astype(np.int64)
        tc = np.repeat(np.arange(n2, dtype=np.int64), np.diff(ts.astype(np.int64)))
        k3 = (k2[tc] << SHIFT) | tw.astype(np.int64)
        return {"words": words, "p1": p1, "b1": b1, "k2": k2, "p2": bp, "b2": bb, "k3": k3, "p3": tp}

    @staticmethod
    def _parse(path, cache):
        words, p1, b1 = [], [], []
        k2, p2, b2, k3, p3 = [], [], [], [], []
        ids = {}
        order = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("\\"):
                    order = int(line[1]) if line[1].isdigit() else 0
                    continue
                if not order or not line.strip():
                    continue
                parts = line.rstrip("\n").split("\t")
                if order == 1:
                    ids[parts[1]] = len(words)
                    words.append(parts[1])
                    p1.append(float(parts[0]))
                    b1.append(float(parts[2]) if len(parts) > 2 else 0.0)
                elif order == 2:
                    v, w = parts[1].split(" ")
                    k2.append((ids[v] << SHIFT) | ids[w])
                    p2.append(float(parts[0]))
                    b2.append(float(parts[2]) if len(parts) > 2 else 0.0)
                else:
                    u, v, w = parts[1].split(" ")
                    k3.append((ids[u] << (2 * SHIFT)) | (ids[v] << SHIFT) | ids[w])
                    p3.append(float(parts[0]))
        assert len(words) < 1 << SHIFT
        k2, k3 = np.array(k2, np.int64), np.array(k3, np.int64)
        o2, o3 = np.argsort(k2), np.argsort(k3)
        np.savez(cache[:-4], words=np.array(words), p1=np.array(p1, np.float32), b1=np.array(b1, np.float32),
                 k2=k2[o2], p2=np.array(p2, np.float32)[o2], b2=np.array(b2, np.float32)[o2],
                 k3=k3[o3], p3=np.array(p3, np.float32)[o3])

    @staticmethod
    def _find(keys, k):
        i = int(np.searchsorted(keys, k))
        return i if i < len(keys) and keys[i] == k else -1

    def _bigram(self, v, w):
        k = (v << SHIFT) | w
        r = self._c2.get(k)
        if r is None:
            i = self._find(self.k2, k)
            # a context with no bigram of its own backs off by nothing (weight 1)
            r = self._c2[k] = (float(self.p2[i]), float(self.b2[i])) if i >= 0 else (None, 0.0)
        return r

    def lp2(self, v, w):
        if v < 0:
            return float(self.p1[w])
        p, _ = self._bigram(v, w)
        return p if p is not None else float(self.b1[v]) + float(self.p1[w])

    def lp3(self, u, v, w):
        if u < 0:
            return self.lp2(v, w)
        k = (u << (2 * SHIFT)) | (v << SHIFT) | w
        r = self._c3.get(k)
        if r is None:
            i = self._find(self.k3, k)
            r = self._c3[k] = float(self.p3[i]) if i >= 0 else math.nan
        if r == r:
            return r
        return self._bigram(u, v)[1] + self.lp2(v, w)

    def _edges(self, s):
        """Per start position, (end, word id) of each word of the model there; a char of none is <unk>."""
        out = []
        for i in range(len(s)):
            e = [(j, self.ids[s[i:j]]) for j in range(i + 1, min(len(s), i + self.longest) + 1) if s[i:j] in self.ids]
            if not any(j == i + 1 for j, _ in e):
                e.append((i + 1, self.unk))
            out.append(e)
        return out

    def _run(self, s, start, states):
        """states {(u, v): log10} at position start of s -> at its end (beam per position)."""
        edges = self._edges(s)
        at = [dict() for _ in range(len(s) + 1)]
        at[start] = states
        for i in range(start, len(s)):
            if not at[i]:
                continue
            best = sorted(at[i].items(), key=lambda kv: -kv[1])[:BEAM]
            for (u, v), sc in best:
                for j, w in edges[i]:
                    t = sc + (self.lp3(u, v, w) if v >= 0 else float(self.p1[w]))
                    key = (v, w)
                    if t > at[j].get(key, -math.inf):
                        at[j][key] = t
        return at[len(s)]

    def score(self, context, cands):
        """ln P(cand | context's last CTX chars) per candidate."""
        ctx = context[-CTX:]
        s0 = self._run(ctx, 0, {(-1, -1): 0.0}) if ctx else {(-1, -1): 0.0}
        top = dict(sorted(s0.items(), key=lambda kv: -kv[1])[:BEAM])
        base = max(top.values())
        out = []
        for c in cands:
            end = self._run(ctx + c, len(ctx), top)
            out.append(LN10 * (max(end.values()) - base) if end else -math.inf)
        if len(self._c3) > 4_000_000:
            self._c2.clear()
            self._c3.clear()
        return out
