"""WowType (Gloridust/WowType) word model words.lm with its rime-ice lexicon.tsv, as its Rust engine scores a path
(core/src/wordlm.rs, beam.rs, lexicon.rs at bb2a04f); read from the files, nothing of WowType is run.
- words.lm (WTLM v1): per word pair ln P(cur | prev) - ln P(cur) (capped at 6), a pair not stored ln lambda(prev) (the
  backoff), a word the model lacks max(backoff, -1.5) (OOV_FLOOR), a previous word the model lacks 0.
- A path's score: per word -cost + 0.85 x lm(prev, word) (lm_weight); cost = -ln(max(freq, 0.5) / the total freq of every
  lexicon entry); a character no entry has costs 18 (unknown_cost) and gets no lm term. Words up to 6 characters
  (max_word_syllables); a word's freq is its largest over readings (the typed code is not matched to a reading).
- Decoding: the best segmentation, exact (dynamic programming over the last word) instead of its beam of 20.
- Context: its beam starts at <s> (WowType gives the word model no text before the caret), so the default ignores the
  row's context; ctx=True starts from the context's last lexicon word instead (a side check).
  ww = WowtypeWords("words.lm", "lexicon.tsv"); ww.score(context, cands)
"""
import math
import struct

import numpy as np


class WowtypeWords:
    OOV_FLOOR, LM_WEIGHT, UNKNOWN_COST, MAX_WORD = -1.5, 0.85, 18.0, 6

    def __init__(self, lm_path, lexicon_path, ctx=False):
        self.ctx = ctx
        d = open(lm_path, "rb").read()
        assert d[:4] == b"WTLM" and struct.unpack_from("<I", d, 4)[0] == 1
        n, m = struct.unpack_from("<II", d, 8)
        at = 16
        words = []
        for _ in range(n):
            ln = d[at]
            words.append(d[at + 1:at + 1 + ln].decode("utf-8"))
            at += 1 + ln
        self.id = {}
        for i, w in enumerate(words):
            if i == 1 and w.startswith("wowtype-data"):  # its data version marker, not a word
                continue
            self.id[w] = i  # a word listed twice keeps its later id, as its table does
        self.backoff = np.frombuffer(d, "<f4", n, at)
        at += 4 * n
        self.starts = np.frombuffer(d, "<u4", n + 1, at)
        at += 4 * (n + 1)
        self.curs = np.frombuffer(d, "<u4", m, at)
        at += 4 * m
        self.scores = np.frombuffer(d, "<f4", m, at)
        at += 4 * m
        assert at == len(d)
        self.n_words, self.n_bigrams = n, m
        freq, total, self.n_entries = {}, 0.0, 0
        for line in open(lexicon_path, encoding="utf-8"):
            if line.startswith("#") or not line.strip():
                continue
            _, word, f = line.rstrip("\n").split("\t")[:3]
            f = float(f)
            total += f
            self.n_entries += 1
            if f > freq.get(word, -1.0):
                freq[word] = f
        self.cost = {w: -math.log(max(f, 0.5) / max(total, 1.0)) for w, f in freq.items()}

    def lm(self, prev, cur):
        p = self.id.get(prev if prev is not None else "<s>")
        if p is None:
            return 0.0
        c = self.id.get(cur)
        if c is None:
            return max(float(self.backoff[p]), self.OOV_FLOOR)
        a, b = int(self.starts[p]), int(self.starts[p + 1])
        k = int(np.searchsorted(self.curs[a:b], c))
        if k < b - a and self.curs[a + k] == c:
            return float(self.scores[a + k])
        return float(self.backoff[p])

    def last_word(self, context):
        tail = (context or "").split("\n")[-1].rstrip()
        for k in range(min(self.MAX_WORD, len(tail)), 0, -1):
            if tail[-k:] in self.cost:
                return tail[-k:]
        return None

    def path(self, text, prev):
        """Best score of text after prev: DP over (end, last word); an unknown character is its own state."""
        best = [dict() for _ in range(len(text) + 1)]
        best[0][prev] = 0.0
        for i in range(len(text)):
            for last, s in best[i].items():
                for j in range(i + 1, min(len(text), i + self.MAX_WORD) + 1):
                    w = text[i:j]
                    c = self.cost.get(w)
                    if c is not None:
                        t, key = s - c + self.LM_WEIGHT * self.lm(last, w), w
                    elif j == i + 1:
                        t, key = s - self.UNKNOWN_COST, ("?", w)
                    else:
                        continue
                    if t > best[j].get(key, -math.inf):
                        best[j][key] = t
        return max(best[-1].values())

    def score(self, context, cands):
        prev = self.last_word(context) if self.ctx and context else None
        return [self.path(c, prev) for c in cands]
