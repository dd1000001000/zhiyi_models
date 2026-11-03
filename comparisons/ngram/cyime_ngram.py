"""CyIME's nine-key word trigram (github.com/Cyletix/CyIME, release `models` t9_ngram-v3-ls-tri-c3.lm, user 2026-11-03:
compare the n-grams) as a candidate scorer with ngram_lm.NgramLM's interface: ln P(candidate | context).
The compact v3 file is read as librime-t9's t9_ngram_format.h / t9_ngram_model.cc define it (FNV-1a hash table, 24-bit
ids, int16 natural logs / 1024, trigram -> bigram -> lower backoff); nothing of CyIME is run. Words are any substring
(up to 8 chars) the hash table holds; a single char it lacks is unseen (unseen_lower / unseen_ml); a longer unseen
substring is never a word (as its Decompose). Viterbi over (u, v) states with a beam, as NgramLM: the context's last
CTX characters lead in, score = best(context + candidate) - best(context); its first word scored by ln_ml (no history,
as T9NgramModel::Start without sentence start), from <s> when the context is empty (its ngram_sentence_start).
  lm = CyimeNgram("t9_ngram-v3-ls-tri-c3.lm"); lm.score(ctx, ["候选"])
"""
import math
import struct

import numpy as np

CTX, MAX_WORD, BEAM = 12, 8, 8  # as ngram_lm
SCALE = 1024.0
KEY_MASK = ~0xFFFFFF & 0xFFFFFFFFFFFFFFFF
UNSEEN = -1


def fnv(b):
    h = 14695981039346656037
    for c in b:
        h = ((h ^ c) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h or 1


class CyimeNgram:
    def __init__(self, path):
        m = np.memmap(path, np.uint8, "r")
        assert bytes(m[:8]) == b"CYT9NGR3"
        (self.version, words, self.eos, self.bos, bigrams, trigrams, cap, self.unseen_lower, self.unseen_ml,
         self.shift, _) = struct.unpack_from("<IIIIQQQffII", m, 8)
        offs = struct.unpack_from("<10Q", m, 8 + struct.calcsize("<IIIIQQQffII"))
        assert self.version == 3
        view = lambda off, dt, n: np.frombuffer(m, dt, int(n), off)  # noqa: E731
        self.slots = view(offs[0], "<u8", cap)
        self.words3 = view(offs[1], "<i2", words * 3).astype(np.float64) / SCALE
        self.begin = view(offs[2], "<u4", words + 1)
        nxt = view(offs[3], "u1", bigrams * 3).reshape(-1, 3).astype(np.uint32)
        self.next = nxt[:, 0] | (nxt[:, 1] << 8) | (nxt[:, 2] << 16)
        self.logp = view(offs[4], "<i2", bigrams)
        self.bback = view(offs[5], "<i2", bigrams)
        self.tri_base = view(offs[6], "<u4", (bigrams >> self.shift) + 1)
        self.tri_lo = view(offs[7], "<u2", bigrams + 1)
        tn = view(offs[8], "u1", trigrams * 3).reshape(-1, 3).astype(np.uint32)
        self.tri_next = tn[:, 0] | (tn[:, 1] << 8) | (tn[:, 2] << 16)
        self.tri_logp = view(offs[9], "<i2", trigrams)
        self.mask = cap - 1
        assert int(self.begin[words]) == bigrams and self.tri_begin(bigrams) == trigrams
        self.words, self.bigrams, self.trigrams = words, bigrams, trigrams
        self._ids, self._c2, self._c3 = {}, {}, {}

    def tri_begin(self, b):
        return int(self.tri_base[b >> self.shift]) + int(self.tri_lo[b])

    def id(self, text):
        r = self._ids.get(text)
        if r is None:
            key = fnv(text.encode("utf-8"))
            high, i, r = key & KEY_MASK, key & self.mask, UNSEEN
            while True:
                slot = int(self.slots[i])
                if not slot:
                    break
                if slot & KEY_MASK == high:
                    r = (slot & 0xFFFFFF) - 1
                    break
                i = (i + 1) & self.mask
            self._ids[text] = r
        return r

    def lower(self, w):
        return self.unseen_lower if w == UNSEEN else self.words3[w * 3]

    def ml(self, w):
        return self.unseen_ml if w == UNSEEN else self.words3[w * 3 + 1]

    @staticmethod
    def _find(arr, first, last, w):
        i = first + int(np.searchsorted(arr[first:last], w))
        return i if i < last and int(arr[i]) == w else -1

    def _bigram(self, v, w):
        k = (v, w)
        r = self._c2.get(k)
        if r is None:
            r = self._c2[k] = self._find(self.next, int(self.begin[v]), int(self.begin[v + 1]), w) if w != UNSEEN else -1
        return r

    def lp2(self, v, w):
        if v == UNSEEN:
            return self.lower(w)
        i = self._bigram(v, w)
        return self.logp[i] / SCALE if i >= 0 else self.words3[v * 3 + 2] + self.lower(w)

    def lp3(self, u, v, w):
        if u == UNSEEN or v == UNSEEN:
            return self.lp2(v, w)
        k = (u, v, w)
        r = self._c3.get(k)
        if r is None:
            b = self._bigram(u, v)
            if b < 0:
                r = self.lp2(v, w)
            else:
                t = self._find(self.tri_next, self.tri_begin(b), self.tri_begin(b + 1), w) if w != UNSEEN else -1
                r = self.tri_logp[t] / SCALE if t >= 0 else self.bback[b] / SCALE + self.lp2(v, w)
            self._c3[k] = r
        return r

    def _edges(self, s):
        out = []
        for i in range(len(s)):
            e = []
            for j in range(i + 1, min(len(s), i + MAX_WORD) + 1):
                w = self.id(s[i:j])
                if w != UNSEEN or j == i + 1:
                    e.append((j, w))
            out.append(e)
        return out

    def _run(self, s, start, states):
        """states {(u, v): ln} at position start of s -> at its end; v None = no history yet."""
        edges = self._edges(s)
        at = [dict() for _ in range(len(s) + 1)]
        at[start] = states
        for i in range(start, len(s)):
            if not at[i]:
                continue
            for (u, v), sc in sorted(at[i].items(), key=lambda kv: -kv[1])[:BEAM]:
                for j, w in edges[i]:
                    t = sc + (self.ml(w) if v is None else self.lp3(u, v, w))
                    key = (v if v is not None else UNSEEN, w)
                    if t > at[j].get(key, -math.inf):
                        at[j][key] = t
        return at[len(s)]

    def score(self, context, cands, han_run=True):
        """han_run (default): its corpus is Han runs, one per line with <s> (ngram_build.cc), so the context is its
        trailing Han run, from <s> when the whole run fits in CTX; False: the context's last CTX characters as they
        are (punctuation then reads as an unseen word)."""
        if han_run:
            k = len(context)
            while k > 0 and "一" <= context[k - 1] <= "鿿":
                k -= 1
            run = context[k:]
            ctx = run[-CTX:]
            start = {(UNSEEN, self.bos): 0.0} if len(run) <= CTX else {(UNSEEN, None): 0.0}
            s0 = self._run(ctx, 0, start) if ctx else start
        else:
            ctx = context[-CTX:]
            s0 = self._run(ctx, 0, {(UNSEEN, None): 0.0}) if ctx else {(UNSEEN, self.bos): 0.0}
        top = dict(sorted(s0.items(), key=lambda kv: -kv[1])[:BEAM])
        base = max(top.values())
        out = []
        for c in cands:
            end = self._run(ctx + c, len(ctx), top)
            out.append(max(end.values()) - base if end else -math.inf)
        if len(self._c3) > 4_000_000:
            self._c2.clear()
            self._c3.clear()
        return out
