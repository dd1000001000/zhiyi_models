"""libpinyin's interpolation model (model20.text.tar.gz: interpolation2.text + the phrase tables) as a candidate scorer,
read as its phonetic_lookup.h scores a step:
  P(w | v) = lambda * c(v, w) / sum_x c(v, x) + (1 - lambda) * u(w)      lambda = 0.312699 (data/table.conf.in)
  u(w) = (c(w) + 1) / sum (c + 1)     unigram counts of \\1-gram plus gen_unigram's +1 for every phrase of the tables
The words are the phrases of its default tables (gb_char, gbk_char, opengram, merged) and of the model; a sentence
starts from <start>. Its pronunciation probability (how often a phrase is read this way) is left out: only the
language model is compared. Viterbi over the last word with a beam, as ngram_lm: the context's last 12 characters
lead in from <start>, score = best(context + candidate) - best(context).
"""
import math
import os

LAMBDA = 0.312699
TABLES = ["gb_char.table", "gbk_char.table", "opengram.table", "merged.table"]
CTX, MAX_WORD, BEAM = 12, 8, 8
START = "<start>"


class LibpinyinLM:
    def __init__(self, folder):
        uni, bi, total_v = {}, {}, {}
        section = 0
        for line in open(os.path.join(folder, "interpolation2.text"), encoding="utf-8"):
            if line.startswith("\\1-gram"):
                section = 1
            elif line.startswith("\\2-gram"):
                section = 2
            elif line.startswith("\\item"):
                f = line.split()
                if section == 1:  # \item token word count n
                    uni[f[2]] = uni.get(f[2], 0) + int(f[4])
                elif section == 2:  # \item token1 word1 token2 word2 count n
                    k = (f[2], f[4])
                    bi[k] = bi.get(k, 0) + int(f[6])
                    total_v[f[2]] = total_v.get(f[2], 0) + int(f[6])
        words = set(uni)
        for t in TABLES:
            for line in open(os.path.join(folder, t), encoding="utf-8"):
                f = line.split("\t")
                if len(f) >= 2:
                    words.add(f[1])
        words.discard(START)
        n = sum(uni.get(w, 0) + 1 for w in words)
        self.u = {w: (uni.get(w, 0) + 1) / n for w in words}
        self.bi, self.total_v = bi, total_v
        self.longest = min(MAX_WORD, max(map(len, self.u)))
        self.unknown = math.log(0.5 / n)

    def lp(self, v, w):
        u = self.u.get(w)
        if u is None:
            return self.unknown if len(w) == 1 else None
        t = self.total_v.get(v)
        b = self.bi.get((v, w), 0) / t if t else 0.0
        return math.log(LAMBDA * b + (1 - LAMBDA) * u)

    def _run(self, s, start, states):
        at = [dict() for _ in range(len(s) + 1)]
        at[start] = states
        for i in range(start, len(s)):
            if not at[i]:
                continue
            best = sorted(at[i].items(), key=lambda kv: -kv[1])[:BEAM]
            for j in range(i + 1, min(len(s), i + self.longest) + 1):
                w = s[i:j]
                for v, sc in best:
                    lp = self.lp(v, w)
                    if lp is not None and sc + lp > at[j].get(w, -math.inf):
                        at[j][w] = sc + lp
        return at[len(s)]

    def score(self, context, cands):
        """ln P(cand | context's last CTX chars) per candidate."""
        ctx = context[-CTX:]
        s0 = self._run(ctx, 0, {START: 0.0}) if ctx else {START: 0.0}
        top = dict(sorted(s0.items(), key=lambda kv: -kv[1])[:BEAM])
        base = max(top.values())
        out = []
        for c in cands:
            end = self._run(ctx + c, len(ctx), top)
            out.append(max(end.values()) - base if end else -math.inf)
        return out
