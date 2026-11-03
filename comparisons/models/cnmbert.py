"""CNMBert (Midsummra/CNMBert, the default 131M model; IgarashiAkatuki/zh-CN-Multi-Mask-Bert): a BERT that turns pinyin
initials into characters (ys -> 原神), scored as its predict() fills them, teacher-forced on each candidate:
  [CLS] context <letter mask> ... [SEP]   (each initial is its own mask token: id = letter - 96, 64 tokens at most)
the masks filled left to right, each step's character taken from a softmax over the whole vocabulary (no pinyin
constraint, as its beam search); score = the sum of the log probabilities (its beam keeps their product).
It reads first letters only, so it is scored on the rows typed as initials alone (null elsewhere).
"""
import re

import torch

HAN = re.compile(r"[㐀-䶿一-鿿]+")
MAX_LEN = 64


class CNMBert:
    def __init__(self, path, device="cuda"):
        from transformers import BertForMaskedLM, BertTokenizer
        self.tok = BertTokenizer.from_pretrained(path)
        self.model = BertForMaskedLM.from_pretrained(path).to(device).eval()
        self.device = device

    @torch.no_grad()
    def score_rows(self, rows, bs=128):
        """rows: (context, [(candidate, initials)]) -> sum of log P per candidate (None when it cannot spell it)."""
        cls, sep = self.tok.cls_token_id, self.tok.sep_token_id
        seqs = []  # (row, cand, ids, position, target)
        out = [[None] * len(c) for _, c in rows]
        for n, (ctx, cands) in enumerate(rows):
            for j, (cand, letters) in enumerate(cands):
                ids = self.tok.convert_tokens_to_ids(list(cand))
                if (not HAN.fullmatch(cand) or len(letters) != len(cand) or self.tok.unk_token_id in ids
                        or not all("a" <= x <= "z" for x in letters)):
                    continue
                masks = [ord(x) - 96 for x in letters]
                c = self.tok.encode(ctx, add_special_tokens=False)
                c = c[max(0, len(c) - (MAX_LEN - 2 - len(masks))):]
                base = [cls] + c + masks + [sep]
                p0 = 1 + len(c)
                for k in range(len(cand)):
                    filled = base[:p0] + ids[:k] + masks[k:] + [sep]
                    seqs.append((n, j, filled, p0 + k, ids[k]))
                out[n][j] = 0.0
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][2]))
        for s in range(0, len(order), bs):
            chunk = [seqs[i] for i in order[s:s + bs]]
            T = max(len(x[2]) for x in chunk)
            ids = torch.tensor([x[2] + [0] * (T - len(x[2])) for x in chunk], device=self.device)
            att = torch.tensor([[1] * len(x[2]) + [0] * (T - len(x[2])) for x in chunk], device=self.device)
            lg = self.model(input_ids=ids, attention_mask=att).logits
            rows_ = torch.arange(len(chunk), device=self.device)
            pos = torch.tensor([x[3] for x in chunk], device=self.device)
            tgt = torch.tensor([x[4] for x in chunk], device=self.device)
            lp = lg[rows_, pos].float().log_softmax(-1).gather(1, tgt[:, None])[:, 0].tolist()
            for x, v in zip(chunk, lp):
                out[x[0]][x[1]] += v
        return out
