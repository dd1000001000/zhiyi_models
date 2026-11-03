"""MetasequoiaIME's sentence rerankers, release model-v1 of metasequoiaime/chinese-ime-lm (sentence-model.safetensors,
the 4M one; sentence-model-desktop.safetensors, 25M; also on Hugging Face as pinyin-ime-reranker-4M / -25M), scored
as the format's reference SentenceModel::score: <bos> + the context's end, cut to leave room for the longest
candidate, then the candidate's mean log P per character. Same chinese-ime-lm format as Youmo's (youmo.py).
"""
import torch

from youmo import BOS, PAD, Youmo


class MetaseqSentence(Youmo):
    @torch.no_grad()
    def score_rows(self, rows, bs=256):
        """rows: (context, [candidates]) -> mean log P per character of each candidate."""
        window = self.cfg["context"]
        seqs = []
        for r, (ctx, cands) in enumerate(rows):
            tails = [self.encode(c) for c in cands]
            room = max(0, window - (max(map(len, tails)) + 1))
            c = self.encode(ctx)
            prefix = [BOS] + c[max(0, len(c) - room):]
            for j, t in enumerate(tails):
                seqs.append((r, j, prefix + t[:-1], len(prefix) - 1, t))
        out = [[float("-inf")] * len(c) for _, c in rows]
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][2]))
        for s in range(0, len(order), bs):
            chunk = [seqs[i] for i in order[s:s + bs]]
            T = -(-max(len(x[2]) for x in chunk) // 16) * 16
            if (s // bs) % 50 == 49:
                torch.cuda.empty_cache()
            ids = torch.tensor([x[2] + [PAD] * (T - len(x[2])) for x in chunk], device=self.device)
            h = self.hidden(ids)
            bi, pi, tg = [], [], []
            for b, (_, _, _, p0, t) in enumerate(chunk):
                for k, tok in enumerate(t):
                    bi.append(b)
                    pi.append(p0 + k)
                    tg.append(tok)
            logits = h[torch.tensor(bi, device=self.device), torch.tensor(pi, device=self.device)] @ self.tok.T
            tgt = torch.tensor(tg, device=self.device)
            lp = (logits.gather(1, tgt[:, None])[:, 0] - torch.logsumexp(logits, -1)).cpu()
            sums = torch.zeros(len(chunk)).index_add_(0, torch.tensor(bi), lp).tolist()
            for b, (r, j, _, _, t) in enumerate(chunk):
                if t:
                    out[r][j] = sums[b] / len(t)
        return out
