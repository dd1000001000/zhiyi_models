"""Wanxiang (amzxyz RIME-LMDG, user 2026-11-03: compare the n-grams) as Rime scores a sentence with it: the sum over
the candidate's words of the dictionary weight, ln(weight) - ln(1e8) (librime dict_compiler + dictionary.cc), plus
Octagram::Query(preceding text, word, is_rear) with Wanxiang's schema values. Not a probability: only the order of a
row's candidates means anything. Read from the files as librime-octagram defines them (gram_db.cc Metadata,
darts-clone units, grammar::encode); nothing of Rime is run.
- Words: the tables wanxiang.dict.yaml imports by default (zi jichu cuoyin duoyin diming mingren yiren), a word's
  weight = its largest over readings (the row's typed code is not matched to a reading). A character no table has
  scores ln(1) - ln(1e8) plus UNKNOWN.
- The candidate's best segmentation (Viterbi on the full score, whole preceding text as Rime's Line::context()).
- ctx=True gives the row's context as preceding text (contextual_suggestions on; Wanxiang's default is off: False).
  wg = WanxiangGram(gram_path, dict_dir); wg.score(context, cands, ctx=True)

Also Rime's own grammar (lotem/rime-octagram-data, branch hans: zh-hans-t-essay-bgw.gram, the "hans" model of its
grammar.yaml, contextual_suggestions on): WanxiangGram(gram, essay="essay.txt", opencc=<dir>, cfg=RIME_CFG), word
weights from rime-essay's essay.txt (the weights Rime's luna_pinyin dictionaries take), librime-octagram's default
parameters. essay.txt is Traditional (luna_pinyin_simp converts its output with OpenCC), so its words are turned
Simplified with OpenCC's TSPhrases.txt / TSCharacters.txt (longest phrase first, else each character's first value);
a Simplified word several Traditional ones turn into keeps the largest weight.
"""
import math
import os
import re
import struct

import numpy as np

CFG = dict(max_length=6, min_length=2, collocation=-14.0, non=-6.0, weak=-100.0, rear=-18.0)  # wanxiang.schema.yaml
RIME_CFG = dict(max_length=4, min_length=3, collocation=-12.0, non=-12.0, weak=-24.0, rear=-18.0)  # octagram.cc defaults
TABLES = ["zi", "jichu", "cuoyin", "duoyin", "diming", "mingren", "yiren"]
KS = math.log(1e8)
UNKNOWN = -20.0
MAX_WORD = 12
VALUE_SCALE = 10000.0


def encode(text):
    out = bytearray()
    for ch in text:
        u = ord(ch)
        if u < 0x80:
            out.append(0xE0 if u == 0 else u)
        elif 0x4000 <= u < 0xA000:
            if u & 0xFF == 0:
                out += bytes([0xE1, (u >> 8) + 0x40])
            else:
                out += bytes([(u >> 8) + 0x40, u & 0xFF])
        else:
            bits = 32
            while bits > 0 and (u & 0xFE000000) == 0:
                bits -= 7
                u = (u << 7) & 0xFFFFFFFF
            n = (bits + 6) // 7
            out.append(0xE0 | n)
            while n > 0:
                n -= 1
                out.append(((u >> 25) & 0x7F) | 0x80)
                u = (u << 7) & 0xFFFFFFFF
    return bytes(out)


class Darts:
    """darts-clone DoubleArray over 32-bit units."""

    def __init__(self, units):
        self.a = units

    def _offset(self, u):
        return (u >> 10) << ((u & (1 << 9)) >> 6)

    def traverse(self, key):
        """Node after the whole key, or None."""
        nid = 0
        u = int(self.a[0])
        for c in key:
            nid ^= self._offset(u) ^ c
            u = int(self.a[nid])
            if (u & ((1 << 31) | 0xFF)) != c:
                return None
        return nid

    def prefixes(self, key, node, limit=8):
        """(value, length) of every prefix of key that is a whole entry below node, at most limit."""
        out = []
        nid = node
        u = int(self.a[nid])
        nid ^= self._offset(u)
        for i, c in enumerate(key):
            nid ^= c
            u = int(self.a[nid])
            if (u & ((1 << 31) | 0xFF)) != c:
                break
            nid ^= self._offset(u)
            if (u >> 8) & 1:
                if len(out) < limit:
                    out.append((int(self.a[nid]) & 0x7FFFFFFF, i + 1))
        return out


def unicode_len(enc, length):
    n, p = 0, 0
    while p < length:
        b = enc[p]
        p += 1 if b & 0x80 == 0 else ((b & 0x0F) + 1 if b & 0xF0 == 0xE0 else 2)
        n += 1
    return n


class T2S:
    """OpenCC Traditional -> Simplified: the longest TSPhrases.txt match at each position, else TSCharacters.txt."""

    def __init__(self, folder):
        def table(name):
            out = {}
            for line in open(os.path.join(folder, name), encoding="utf-8"):
                if line.startswith("#") or "\t" not in line:
                    continue
                k, v = line.rstrip("\n").split("\t", 1)
                out[k] = v.split(" ")[0]
            return out
        self.phrases, self.chars = table("TSPhrases.txt"), table("TSCharacters.txt")
        self.longest = max(map(len, self.phrases))

    def __call__(self, text):
        out, i = [], 0
        while i < len(text):
            for n in range(min(self.longest, len(text) - i), 1, -1):
                if text[i:i + n] in self.phrases:
                    out.append(self.phrases[text[i:i + n]])
                    i += n
                    break
            else:
                out.append(self.chars.get(text[i], text[i]))
                i += 1
        return "".join(out)


class WanxiangGram:
    def __init__(self, gram, dict_dir=None, essay=None, cfg=CFG, opencc=None, tables=None):
        self.cfg = cfg
        m = np.memmap(gram, np.uint8, "r")
        assert bytes(m[:14]) == b"Rime::Grammar/"
        size, off = struct.unpack_from("<Ii", m, 36)
        self.trie = Darts(np.frombuffer(m, "<u4", size, 40 + off))
        self.weights = {}
        t2s = T2S(opencc) if opencc else (lambda w: w)
        for line in open(essay, encoding="utf-8") if essay else []:
            f = line.rstrip("\n").split("\t")
            if len(f) >= 2 and f[0]:
                f[0] = t2s(f[0])
                w = float(f[1])
                self.weights[f[0]] = max(self.weights.get(f[0], -math.inf), math.log(w) if w > 0 else math.log(2.220446049250313e-16))
        for t in (tables or TABLES) if dict_dir else []:
            body, cols, at = False, [], 2
            for line in open(os.path.join(dict_dir, t + ".dict.yaml"), encoding="utf-8"):
                if not body:
                    body = line.startswith("...")
                    if line.startswith("  - "):  # a header's columns (rime-ice's tencent: text, weight)
                        cols.append(line[4:].split("#")[0].strip())
                    at = cols.index("weight") if "weight" in cols else 2
                    continue
                f = line.rstrip("\n").split("\t")
                if len(f) < 2 or not f[0] or f[0].startswith("#"):
                    continue
                num = re.match(r"[0-9.]+", f[at]) if len(f) > at else None  # "6N": std::stod reads the leading 6
                w = float(num.group()) if num else 0.0
                lw = math.log(w) if w > 0 else math.log(2.220446049250313e-16)
                if lw > self.weights.get(f[0], -math.inf):
                    self.weights[f[0]] = lw
        self._q = {}

    def query(self, context, word, rear):
        k = (context[-(self.cfg["max_length"] - 1):], word, rear)
        r = self._q.get(k)
        if r is not None:
            return r
        n = min(8, self.cfg["max_length"] - 1)
        result = self.cfg["non"]
        if context:
            ctx = context[-n:]
            wq = encode(word[:n])
            for s in range(len(ctx)):
                cq = encode(ctx[s:])
                node = self.trie.traverse(cq)
                if node is None:
                    continue
                for value, length in self.trie.prefixes(wq, node):
                    mlen = unicode_len(wq, length)
                    whole = s == 0 and length == len(wq)
                    pen = self.cfg["collocation"] if (len(ctx) - s) + mlen >= self.cfg["min_length"] or whole else self.cfg["weak"]
                    result = max(result, value / VALUE_SCALE + pen)
        if rear and len(word) <= n:
            node = self.trie.traverse(encode(word))
            if node is not None:
                hit = self.trie.prefixes(b"$", node)
                if hit:
                    result = max(result, hit[0][0] / VALUE_SCALE + self.cfg["rear"])
        self._q[k] = result
        if len(self._q) > 3_000_000:
            self._q.clear()
        return result

    def _best(self, text, preceding):
        n = len(text)
        best = [(-math.inf, "")] * (n + 1)  # (score, text so far as Rime's line context)
        best[0] = (0.0, preceding)
        for i in range(n):
            sc, ctx = best[i]
            if sc == -math.inf:
                continue
            for j in range(i + 1, min(n, i + MAX_WORD) + 1):
                w = text[i:j]
                lw = self.weights.get(w)
                if lw is None:
                    if j > i + 1:
                        continue
                    lw = UNKNOWN
                t = sc + lw - KS + self.query(ctx, w, j == n)
                if t > best[j][0]:
                    best[j] = (t, ctx + w)
        return best[n][0]

    def score(self, context, cands, ctx=True):
        pre = context.split("\n")[-1] if ctx else ""
        return [self._best(c, pre) for c in cands]
