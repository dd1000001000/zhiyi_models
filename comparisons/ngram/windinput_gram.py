"""WindInput (清风输入法, huanfeng/WindInput) pinyin sentence scoring with its grammar wanxiang-mini.gram (a subset of
Wanxiang's LTS grammar) and its dictionary rime_frost, as its Rust engine scores a sentence (wind-engine
src/pinyin/lattice.rs score_node, viterbi.rs decode_beam, octagram.rs OctagramGrammar; tag v0.130.1). Read from the
files; nothing of WindInput is run.
- Node score (score_node): ln(weight / 242,154,693) (weight <= 0: 0.5); a single character -3, or +2 when it is one of
  its function words read as such (no word penalty then); a longer word ending in 了的着过得地 -1, else
  +3 x sqrt(length) x clamp((ln p + 15) / 12, 0, 1); every word -3. A word's weight is its largest over readings and
  tables (the row's typed code is not matched to a reading); the function-word check takes that reading.
- Words: the tables rime_frost.dict.yaml imports (8105 41448 base ext others corrections); a character no table has
  scores as weight 0.
- Grammar (octagram.rs, its defaults; config.toml sets only weight 0.5 and the model): context = the path's last two
  words (empty at the sentence start), its last <= 3 characters shortened one by one, the word's first <= 3
  characters, best ln(count) of a hit, -12 when shorter than 2 characters and not a whole match, the rear query
  word$ -6, floor 0; added as weight x (hit - 8.34).
- Decoding: a beam of 7 lines per position, one per last word, as decode_beam. WindInput passes no text before the
  caret, so the row's context is not used.
  wg = WindGram(gram_path, dict_dir, weight=0.5); wg.score(context, cands)
"""
import math
import os
import re

import numpy as np

from wanxiang_gram import VALUE_SCALE, Darts, encode, unicode_len

TABLES = ["8105", "41448", "base", "ext", "others", "corrections"]
DICT_TOTAL = 242_154_693.0
WORD_PENALTY = 3.0
BASELINE = 8.34
BEAM = 7
MAX_WORD = 12
FUNCTION_WORDS = {
    "了": ["le"], "的": ["de"], "地": ["de"], "得": ["de", "dei"], "着": ["zhe"], "过": ["guo"],
    "我": ["wo"], "你": ["ni"], "他": ["ta"], "她": ["ta"], "它": ["ta"], "们": ["men"], "这": ["zhe", "zhei"],
    "那": ["na", "nei"], "和": ["he"], "与": ["yu"], "在": ["zai"], "把": ["ba"], "被": ["bei"], "让": ["rang"],
    "从": ["cong"], "到": ["dao"], "对": ["dui"], "向": ["xiang"], "跟": ["gen"], "不": ["bu"], "没": ["mei"],
    "也": ["ye"], "都": ["dou"], "就": ["jiu"], "才": ["cai"], "还": ["hai"], "又": ["you"], "再": ["zai"],
    "很": ["hen"], "太": ["tai"], "最": ["zui"], "是": ["shi"], "有": ["you"], "会": ["hui"], "能": ["neng"],
    "要": ["yao"], "可": ["ke"], "去": ["qu"], "来": ["lai"], "做": ["zuo"], "说": ["shuo"], "看": ["kan"],
    "想": ["xiang"]}


def score_node(word, code, weight):
    lp = math.log(weight / DICT_TOTAL) if weight > 0 else math.log(0.5 / DICT_TOTAL)
    function = len(word) == 1 and code in FUNCTION_WORDS.get(word, ())
    if len(word) == 1:
        lp += 2.0 if function else -3.0
    elif word[-1] in "了的着过得地":
        lp -= 1.0
    elif weight > 0:
        lp += 3.0 * math.sqrt(len(word)) * min(1.0, max(0.0, (lp + 15.0) / 12.0))
    return lp if function else lp - WORD_PENALTY


class WindGram:
    def __init__(self, gram, dict_dir, weight=0.5, tables=TABLES):
        self.weight = weight
        m = np.memmap(gram, np.uint8, "r")
        assert bytes(m[:14]) == b"Rime::Grammar/"
        import struct
        size, off = struct.unpack_from("<Ii", m, 36)
        self.trie = Darts(np.frombuffer(m, "<u4", size, 40 + off))
        self.words = {}  # text -> (weight, reading)
        for t in tables:
            body = False
            for line in open(os.path.join(dict_dir, t + ".dict.yaml"), encoding="utf-8"):
                if not body:
                    body = line.startswith("...")
                    continue
                f = line.rstrip("\n").split("\t")
                if len(f) < 2 or not f[0] or f[0].startswith("#"):
                    continue
                num = re.match(r"-?[0-9.]+", f[2]) if len(f) > 2 else None
                w = int(float(num.group())) if num else 0
                old = self.words.get(f[0])
                if old is None or w > old[0]:
                    self.words[f[0]] = (w, f[1].replace(" ", ""))
        self._q = {}

    def hit(self, context, word, rear):
        """octagram.rs best_collocation: ln(count) of the best hit, 0 when none."""
        if not context:
            return 0.0
        k = (context[-3:], word[:4], rear)
        r = self._q.get(k)
        if r is not None:
            return r
        wq = encode(word[:3])
        full = len(word) <= 3
        ctx = context[-3:]
        best = 0.0
        for s in range(len(ctx)):
            node = self.trie.traverse(encode(ctx[s:]))
            if node is None:
                continue
            for value, length in self.trie.prefixes(wq, node):
                whole = s == 0 and length == len(wq)
                pen = 0.0 if (len(ctx) - s) + unicode_len(wq, length) >= 2 or whole else -12.0
                best = max(best, value / VALUE_SCALE + pen)
        if rear and full:
            node = self.trie.traverse(wq)
            if node is not None:
                found = self.trie.prefixes(b"$", node)
                if found:
                    best = max(best, found[0][0] / VALUE_SCALE - 6.0)
        self._q[k] = best
        if len(self._q) > 3_000_000:
            self._q.clear()
        return best

    def query(self, context, word, rear):
        return 0.0 if self.weight == 0 else self.weight * (self.hit(context, word, rear) - BASELINE)

    def _best(self, text):
        n = len(text)
        states = [[] for _ in range(n + 1)]  # per position: [score, last word, word before], best first
        states[0] = [[0.0, "", ""]]
        for end in range(1, n + 1):
            for start in range(max(0, end - MAX_WORD), end):
                if not states[start]:
                    continue
                w = text[start:end]
                entry = self.words.get(w)
                if entry is None and end - start > 1:
                    continue
                weight, code = entry or (0, "")
                node = score_node(w, code, weight)
                for sc, last, prev in states[start]:
                    ctx = "" if not last else (prev + last if prev else last)
                    self._push(states[end], sc + node + self.query(ctx, w, end == n), w, last)
        return states[n][0][0]

    @staticmethod
    def _push(here, total, word, prev):
        """viterbi.rs push_state: one line per last word, higher score wins (ties keep the first), at most 7."""
        for i, s in enumerate(here):
            if s[1] == word:
                if total > s[0]:
                    here.pop(i)
                    break
                return
        if len(here) >= BEAM and total <= here[-1][0]:
            return
        at = len(here)
        while at > 0 and here[at - 1][0] < total:
            at -= 1
        here.insert(at, [total, word, prev])
        del here[BEAM:]

    def score(self, context, cands):
        return [self._best(c) for c in cands]
