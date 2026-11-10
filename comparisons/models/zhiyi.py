"""Zhiyi word-choice models (the int8 graph laya.int8g.onnx of a release zip, with its tokenizer.json and
unified_vocab.json; zhen-mini-next-preview: the single 4-bit file laya.mini.w4.onnx, context then key mode) as the
input method feeds them: the whole page of candidates, the choice head's logit per candidate.

Sequence (choice_format "ctx-first"):
  <bos> context(last 128 tokens) <eos> <mask> per syllable slot <eos> <mask> cand1(16 tokens) <mask> cand2 ... <eos>
The slots: the leading candidate's reading aligned with the letters typed, each syllable typed in full (f:zhong) or as
a prefix (p:zh); none when that reading does not spell the code. Each slot's <mask> carries its unit vector (unit_ids);
each candidate's <mask> is its marker, and the graph returns one logit per marker.
The compact sequence (unified_vocab.json "choice_compact": zh-lite-preview): no <mask> before a candidate, each
candidate tokenized with tokenizer.candidates.json (no leading "▁") and its first token as the marker:
  <bos> context <eos> <mask> per syllable slot <eos> cand1(16 tokens) cand2 ... <eos>
"""
import itertools
import json
import os

import numpy as np

CONTEXT_TOKENS = 128
CAND_TOKENS = 16


def slot_units(syl, code):
    """Each syllable as typed in `code`: in full ("f"), or as a prefix ("p": an initial, or the unfinished last
    syllable); None when the code does not spell the syllables that way."""
    def walk(i, pos):
        if i == len(syl):
            return [] if pos == len(code) else None
        s = syl[i]
        options = []
        if code.startswith(s, pos):
            options.append(("f", s, len(s)))
        for k in range(len(s) - 1, 0, -1):
            if code.startswith(s[:k], pos) and (k == 1 or i == len(syl) - 1):
                options.append(("p", s[:k], k))
        for kind, u, n in options:
            rest = walk(i + 1, pos + n)
            if rest is not None:
                return [(kind, u)] + rest
        return None
    return walk(0, 0)


def units_for(cand, code):
    """The candidate's syllables (a reading per character, pypinyin) aligned with the typed code."""
    from pypinyin import Style, lazy_pinyin, pinyin
    syl = lazy_pinyin(cand)
    u = slot_units(syl, code) if len(syl) == len(cand) else None
    if u is not None:
        return u
    readings = [pinyin(c, style=Style.NORMAL, heteronym=True)[0] for c in cand]
    for combo in itertools.islice(itertools.product(*readings), 64):
        u = slot_units(list(combo), code)
        if u is not None:
            return u
    return None


GRAPHS = ("laya.int8g.onnx", "laya.mini.w4.onnx")  # a release zip's word-choice graph, the first one present


class Zhiyi:
    def __init__(self, folder, dml=0, graph=None, context_chars=None):
        """folder: an unpacked int8 release zip (or zhen-mini-next-preview's w4 zip: the single 4-bit file, run on the
        CPU in its two word-choice passes, see score_single). dml: DirectML device id (None = CPU). context_chars: cut
        the context to its last N characters first (a check against models that see less)."""
        graph = graph or next(g for g in GRAPHS if os.path.exists(os.path.join(folder, g)))
        self.graph_path = os.path.join(folder, graph)
        self.context_chars = context_chars
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(os.path.join(folder, "tokenizer.json"))
        self.bos, self.eos, self.mask, self.pad, self.unk = (self.tok.token_to_id(t)
                                                             for t in ("<bos>", "<eos>", "<mask>", "<pad>", "<unk>"))
        v = json.load(open(os.path.join(folder, "unified_vocab.json"), encoding="utf-8"))
        assert v["choice_format"] == "ctx-first"
        self.compact = v.get("choice_compact", False)
        if self.compact:
            self.ctok = Tokenizer.from_file(os.path.join(folder, "tokenizer.candidates.json"))
        self.unit_index = {u: i for i, u in enumerate(v["units"])}
        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        so.inter_op_num_threads = 1
        providers = ["CPUExecutionProvider"]
        if dml is not None:
            so.enable_mem_pattern = False
            providers = [("DmlExecutionProvider", {"device_id": dml}), "CPUExecutionProvider"]
        so.log_severity_level = 3  # the single file's inputs a mode leaves out are overridable defaults (a warning each)
        probe = ort.InferenceSession(self.graph_path, so, providers=["CPUExecutionProvider"])
        # is_step has a default, so ONNX Runtime lists it among the overridable initializers, not the inputs
        self.single = "is_step" in {i.name for i in probe.get_inputs() + probe.get_overridable_initializers()}
        if self.single:  # weight-only 4-bit operators: CPU
            providers = ["CPUExecutionProvider"]
        self.s = ort.InferenceSession(self.graph_path, so, providers=providers)
        self.inputs = {i.name for i in self.s.get_inputs()}
        self._cache = {}
        # composition: the input method's pinyin masks (the Simplified characters' readings) over char_logits' columns
        self.chars = v["chars"]
        self.context_prefix, self.max_context_tokens, self.max_slots = (v["context_prefix"], v["max_context_tokens"],
                                                                         v["max_slots"])
        self.by_syl = {}
        for c, rs in v["readings"].items():
            for s in rs:
                self.by_syl.setdefault(s, []).append(self.chars.index(c))
        self._masks = {}

    def _text(self, s, n=None):
        ids = self._cache.get(s)
        if ids is None:
            ids = self.tok.encode(s, add_special_tokens=False).ids
            if len(self._cache) < 200_000:
                self._cache[s] = ids
        return ids if n is None else ids[:n]

    def sequence(self, context, cands, units):
        ctx = self.tok.encode(context.replace("<mask>", " "), add_special_tokens=False).ids[-CONTEXT_TOKENS:]
        ids = [self.bos] + ctx + [self.eos]
        unit_ids = [0] * len(ids)
        for kind, s in units:
            ids.append(self.mask)
            unit_ids.append(self.unit_index.get(f"{kind}:{s}", 0))
        ids.append(self.eos)
        unit_ids.append(0)
        markers = []
        for c in cands:
            markers.append(len(ids))
            if self.compact:
                ids += self.ctok.encode(c.replace("<mask>", " "), add_special_tokens=False).ids[:CAND_TOKENS] or [self.unk]
            else:
                ids += [self.mask] + self._text(" " + c.replace("<mask>", " "), CAND_TOKENS)
        ids.append(self.eos)
        unit_ids += [0] * (len(ids) - len(unit_ids))
        return ids, unit_ids, markers

    def score(self, rows, bs=64):
        """rows: test-set rows (context, code, candidates). A logit for every candidate of each row."""
        items = []
        for r in rows:
            code = "".join(c for c in r["code"] if "a" <= c <= "z")
            context = r["context"][-self.context_chars:] if self.context_chars else r["context"]
            items.append(self.sequence(context, r["candidates"], units_for(r["candidates"][0], code) or []))
        if self.single:
            return [self.score_single(*x)[:len(r["candidates"])] for x, r in zip(items, rows)]
        order = sorted(range(len(rows)), key=lambda i: len(rows[i]["context"]) + 4 * len(rows[i]["candidates"]))
        out = [None] * len(rows)
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            L = max(len(items[i][0]) for i in idx)
            K = max(len(items[i][2]) for i in idx)
            B = len(idx)
            ids = np.full((B, L), self.pad, np.int64)
            att = np.zeros((B, L), np.int64)
            units = np.zeros((B, L), np.int64)
            pos = np.zeros((B, K), np.int64)
            mask = np.zeros((B, K), bool)
            for b, i in enumerate(idx):
                x, u, m = items[i]
                ids[b, :len(x)] = x
                att[b, :len(x)] = 1
                units[b, :len(x)] = u
                pos[b, :len(m)] = m
                mask[b, :len(m)] = True
            feed = {"input_ids": ids, "attention_mask": att, "unit_ids": units, "marker_pos": pos, "marker_mask": mask,
                    "slot_pos": np.zeros((B, 1), np.int64)}
            lg = self.s.run(["logits"], {k: x for k, x in feed.items() if k in self.inputs})[0]
            for b, i in enumerate(idx):
                out[i] = lg[b, :len(rows[i]["candidates"])].astype(float).tolist()
        return out

    def score_single(self, ids, unit_ids, markers):
        """The single file (is_step false): the leading block (up to the first <eos>) in context mode, its keys and
        values (present_0..3) as the past of the rest in key mode, as an input method caches the context."""
        n = ids.index(self.eos) + 1
        x, u = np.array([ids], np.int64), np.array([unit_ids], np.int64)
        c = self.s.run(["present_0", "present_1", "present_2", "present_3"], {"input_ids": x[:, :n], "unit_ids": u[:, :n]})
        feed = dict(zip(("past_enc_k", "past_enc_v", "past_choice_k", "past_choice_v"), c))
        feed["past_mask"] = np.ones((1, n), bool)
        lg = self.s.run(["logits"], {"input_ids": x[:, n:], "unit_ids": u[:, n:], **feed,
                                     "marker_pos": np.array([[m - n for m in markers]], np.int64)})[0]
        return lg[0].astype(float).tolist()

    def allowed(self, kind, s):
        """The characters a unit may spell: f:zhong those read zhong, p:zh those with a reading starting zh."""
        key = (kind, s)
        if key not in self._masks:
            m = np.zeros(len(self.chars), bool)
            for syl, cs in self.by_syl.items():
                if (syl == s) if kind == "f" else syl.startswith(s):
                    m[cs] = True
            self._masks[key] = m
        return self._masks[key]

    def compose_sequence(self, context, units):
        """<bos> "已输入的上文: " + context (160 tokens at most, the caret's end kept) <eos> <mask> per slot <eos>."""
        ctx = self.tok.encode(self.context_prefix + context, add_special_tokens=False).ids
        head = self.tok.encode(self.context_prefix, add_special_tokens=False).ids
        if len(ctx) > self.max_context_tokens:
            ctx = head + ctx[len(ctx) - (self.max_context_tokens - len(head)):]
        ids = [self.bos] + ctx + [self.eos]
        unit_ids = [0] * len(ids)
        slots = []
        for kind, s in units[:self.max_slots]:
            slots.append(len(ids))
            ids.append(self.mask)
            unit_ids.append(self.unit_index.get(f"{kind}:{s}", 0))
        ids.append(self.eos)
        unit_ids.append(0)
        return ids, unit_ids, slots

    def compose(self, rows, k=5, bs=64):
        """rows: (context, units). One pass of the composition head: per slot, the top k [(character, log P)] over
        the characters the unit may spell."""
        items = [self.compose_sequence(c, u) for c, u in rows]
        order = sorted(range(len(rows)), key=lambda i: len(items[i][0]))
        out = [None] * len(rows)
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            L = max(len(items[i][0]) for i in idx)
            S = max(len(items[i][2]) for i in idx)
            B = len(idx)
            ids = np.full((B, L), self.pad, np.int64)
            att = np.zeros((B, L), np.int64)
            units = np.zeros((B, L), np.int64)
            pos = np.zeros((B, S), np.int64)
            for b, i in enumerate(idx):
                x, u, sl = items[i]
                ids[b, :len(x)] = x
                att[b, :len(x)] = 1
                units[b, :len(x)] = u
                pos[b, :len(sl)] = sl
            lg = self.s.run(["char_logits"], {"input_ids": ids, "attention_mask": att, "unit_ids": units,
                                              "marker_pos": np.zeros((B, 1), np.int64), "marker_mask": np.ones((B, 1), bool),
                                              "slot_pos": pos})[0]
            for b, i in enumerate(idx):
                slots = []
                for j, (kind, sy) in enumerate(rows[i][1][:self.max_slots]):
                    m = self.allowed(kind, sy)
                    x = np.where(m, lg[b, j].astype(np.float64), -np.inf)
                    x = x - np.logaddexp.reduce(x[m]) if m.any() else x
                    top = np.argsort(-x)[:k]
                    slots.append([(self.chars[t], float(x[t])) for t in top if np.isfinite(x[t])])
                out[i] = slots
        return out
