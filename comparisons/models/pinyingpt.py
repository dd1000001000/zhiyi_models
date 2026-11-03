"""PinyinGPT-Concat and the frozen GPT2-zh-21k it was built on (Tan et al., ACL 2022, Tencent AI Lab), as the paper's
code (VisualJoyce/Transformers4IME 8f1573e) feeds them, scoring given candidates instead of generating.

PinyinGPT-Concat (data/benchmark/pinyingpt_concat.py, concat_mode "segmented", position_mode "aligned"):
  [CLS] context [SEP] [py1] ... [pyn] [SEP] c1 ... c(n-1)
  positions: 0 .. s for [CLS] context [SEP] (s = the [SEP]'s index), s+1 .. s+n+1 for the pinyin tokens and the second
  [SEP], and the characters reuse the pinyin tokens' positions (ck at s+k). The second [SEP] predicts c1, ck predicts
  c(k+1).
GPT2-zh-21k (data/benchmark/pinyingpt_compatible.py): [CLS] context c1 ... c(n-1); the pinyin only constrains.
Either way a character's log P is taken over the characters its pinyin token allows (pinyin2char.json, as the
paper's constrained decoding), and a candidate's score is the sum over its characters.

Pinyin: the released PinyinGPT-Concat was trained with every syllable abbreviated to its first letter (the repository's
pretrain_pinyingpt.sh: --abbr_mode full, "AbbrOnly"), so its input tokens are always the first letters ([z] for zhong);
typed=True feeds what was typed instead ([zhong] for a full syllable), which it never saw in training. The constraint
follows what was typed either way: a syllable typed in full allows its own characters ([zhong]'s), a prefix its first
letter's ([z]'s). The syllables come from the row's readings of that candidate. A character the pinyin table does not
list under that token (a reading it lacks) is added to the allowed set for that position. A candidate the model cannot spell (a character
outside its vocabulary, Latin letters) scores -1e4: these are never the answer on the neutral sets.
"""
import json
import os
import re

import torch

HAN = re.compile(r"[㐀-䶿一-鿿]+")
CONTEXT_CHARS = 100  # the paper trains on up to 128 tokens: the context's last 100 characters leave room for the rest


class PinyinGPT:
    def __init__(self, path, concat, device="cuda", typed=False):
        from safetensors.torch import load_file
        from transformers import BertTokenizer, GPT2Config, GPT2LMHeadModel
        self.concat, self.device, self.typed = concat, device, typed
        self.tok = BertTokenizer.from_pretrained(path)
        n = len(self.tok)
        pys = json.load(open(os.path.join(path, "additional_special_tokens.json"), encoding="utf-8"))
        self.py_id = {t[1:-1]: n + i for i, t in enumerate(pys)}  # the model's rows after BERT's 21128 (wte 21571)
        st = os.path.join(path, "model.safetensors")
        sd = load_file(st) if os.path.exists(st) else torch.load(os.path.join(path, "pytorch_model.bin"),
                                                                 map_location="cpu", weights_only=True)
        cfg = GPT2Config.from_pretrained(path)
        cfg.vocab_size = sd["transformer.wte.weight"].shape[0]
        self.model = GPT2LMHeadModel(cfg)
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        assert not [k for k in missing if not k.endswith("attn.bias") and k != "lm_head.weight"], missing
        self.model.tie_weights()
        self.model.to(device).eval()
        self.V = cfg.vocab_size
        vocab = self.tok.get_vocab()
        p2c = json.load(open(os.path.join(path, "pinyin2char.json"), encoding="utf-8"))
        keys = sorted(p2c)
        self.key_index = {k: i for i, k in enumerate(keys)}
        self.allowed = torch.zeros(len(keys), self.V, dtype=torch.bool)
        for k in keys:
            ids = [vocab[c] for c in p2c[k] if c in vocab]
            self.allowed[self.key_index[k], ids] = True
        self.allowed = self.allowed.to(device)
        self.vocab = vocab

    def logits_at(self, seqs, poss, rows, at, step):
        """seqs / poss: token and position ids per sequence; (rows, at): the positions whose next-token logits are
        wanted. The output layer only there (the full one over every position filled the card and spilled into
        shared memory on Windows); lengths padded to 16, the cache emptied now and then."""
        L = -(-max(map(len, seqs)) // 16) * 16
        ids = torch.zeros((len(seqs), L), dtype=torch.long)
        pos = torch.zeros((len(seqs), L), dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for r, (x, p) in enumerate(zip(seqs, poss)):
            ids[r, :len(x)] = torch.tensor(x)
            pos[r, :len(x)] = torch.tensor(p if p is not None else range(len(x)))
            att[r, :len(x)] = 1
        if step % 50 == 49:
            torch.cuda.empty_cache()
        h = self.model.transformer(input_ids=ids.to(self.device), attention_mask=att.to(self.device),
                                   position_ids=pos.to(self.device)).last_hidden_state
        return self.model.lm_head(h[torch.tensor(rows, device=self.device), torch.tensor(at, device=self.device)]).float()

    def fed(self, key):
        """The input token of a typed pinyin key: its first letter, as the model was trained (or as typed)."""
        return key if self.typed else key[0]

    def encode(self, cand, readings):
        """(character ids, pinyin keys) or None when the model cannot spell the candidate."""
        if not HAN.fullmatch(cand) or len(readings) != len(cand):
            return None
        ids = [self.vocab.get(c) for c in cand]
        keys = [s.replace("ü", "v") if kind == "full" else s[0] for s, kind in readings]
        if None in ids or any(k not in self.key_index or k not in self.py_id for k in keys):
            return None
        return ids, keys

    def sequence(self, ctx_ids, ids, keys):
        """(input ids, position ids or None, index of each character's predicting position)."""
        cls, sep = self.tok.cls_token_id, self.tok.sep_token_id
        if self.concat:
            s = 1 + len(ctx_ids)
            n = len(ids)
            seq = [cls] + ctx_ids + [sep] + [self.py_id[self.fed(k)] for k in keys] + [sep] + ids[:-1]
            pos = list(range(s + 1)) + list(range(s + 1, s + n + 2)) + list(range(s + 1, s + n))
            first = s + n + 1
        else:
            seq = [cls] + ctx_ids + ids[:-1]
            pos = None
            first = len(ctx_ids)
        return seq, pos, list(range(first, first + len(ids)))

    @torch.no_grad()
    def generate(self, rows, beam=16, keep=5, group=16):
        """Constrained beam search as the paper's benchmark (num_beams 16): rows (context, pinyin keys) -> per row the
        top `keep` [(sum log P, text)]. Each step's log P is taken over the characters the step's pinyin token allows."""
        cls, sep = self.tok.cls_token_id, self.tok.sep_token_id
        inv = {i: t for t, i in self.vocab.items()}
        order = sorted(range(len(rows)), key=lambda i: len(rows[i][1]))
        out = [None] * len(rows)
        for g0 in range(0, len(order), group):
            idx = order[g0:g0 + group]
            heads = []  # per row: (prefix ids, prefix positions, first character position, keys)
            for i in idx:
                context, keys = rows[i]
                ctx = self.tok.encode(context[-CONTEXT_CHARS:], add_special_tokens=False)
                if self.concat:
                    s, n = 1 + len(ctx), len(keys)
                    seq = [cls] + ctx + [sep] + [self.py_id[self.fed(k)] for k in keys] + [sep]
                    heads.append((seq, list(range(s + 1)) + list(range(s + 1, s + n + 2)), s + 1, keys))
                else:
                    heads.append(([cls] + ctx, list(range(1 + len(ctx))), 1 + len(ctx), keys))
            beams = [[(0.0, [])] for _ in idx]
            for t in range(max(len(h[3]) for h in heads)):
                live = [(b, k) for b, h in enumerate(heads) if t < len(h[3]) for k in range(len(beams[b]))]
                if not live:
                    break
                seqs = [heads[b][0] + beams[b][k][1] for b, k in live]
                poss = [heads[b][1] + list(range(heads[b][2], heads[b][2] + t)) for b, k in live]
                lg = self.logits_at(seqs, poss, list(range(len(seqs))), [len(x) - 1 for x in seqs], g0 // group * 16 + t)
                keys = torch.tensor([self.key_index[heads[b][3][t]] for b, _ in live], device=self.device)
                lg = lg.masked_fill(~self.allowed[keys], float("-inf"))
                lp = lg - lg.logsumexp(1, keepdim=True)
                top_v, top_i = lp.topk(beam, dim=1)
                top_v, top_i = top_v.cpu().tolist(), top_i.cpu().tolist()
                new = {b: [] for b, _ in live}
                for r, (b, k) in enumerate(live):
                    s0, chars = beams[b][k]
                    for v, c in zip(top_v[r], top_i[r]):
                        if v > float("-inf"):
                            new[b].append((s0 + v, chars + [c]))
                for b, cand in new.items():
                    beams[b] = sorted(cand, key=lambda x: -x[0])[:beam]
            for b, i in enumerate(idx):
                out[i] = [(s, "".join(inv[c] for c in chars)) for s, chars in beams[b][:keep]]
        return out

    @torch.no_grad()
    def score_rows(self, rows, bs=96):
        """rows: (context, [(candidate, readings)]) -> lists of scores."""
        seqs = []  # (row, cand, seq, pos, predict positions, ids, keys)
        out = [[-1e4] * len(c) for _, c in rows]
        for i, (context, cands) in enumerate(rows):
            ctx_ids = self.tok.encode(context[-CONTEXT_CHARS:], add_special_tokens=False)
            for j, (cand, readings) in enumerate(cands):
                e = self.encode(cand, readings)
                if e is None:
                    continue
                seq, pos, at = self.sequence(ctx_ids, *e)
                seqs.append((i, j, seq, pos, at, *e))
        order = sorted(range(len(seqs)), key=lambda k: len(seqs[k][2]))
        for b in range(0, len(order), bs):
            chunk = [seqs[k] for k in order[b:b + bs]]
            lg = self.logits_at([c[2] for c in chunk], [c[3] for c in chunk],
                                [r for r, c in enumerate(chunk) for _ in c[4]], [p for c in chunk for p in c[4]], b // bs)
            want = torch.tensor([x for c in chunk for x in c[5]], device=self.device)
            keys = torch.tensor([self.key_index[k] for c in chunk for k in c[6]], device=self.device)
            mask = self.allowed[keys].clone()
            mask[torch.arange(len(want), device=self.device), want] = True
            lp = lg.gather(1, want[:, None])[:, 0] - lg.masked_fill(~mask, float("-inf")).logsumexp(1)
            k = 0
            for c in chunk:
                n = len(c[4])
                out[c[0]][c[1]] = float(lp[k:k + n].sum())
                k += n
        return out
