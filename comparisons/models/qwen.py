"""Qwen3.5-2B-Base as a reference, zero-shot (not an input-method model): log P(<|endoftext|> + context + candidate)
- log P(<|endoftext|> + context), each string tokenized whole. The candidate often merges with the context's last
characters into one token (必要 + 的 -> 必要的), so the candidate tokenized apart is a split the model almost never saw;
only the tokens after the two tokenizations part are scored, and the context's own tokens from that point subtracted.
"""
import torch


class Qwen:
    def __init__(self, path, device="cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(device).eval()
        self.start = self.tok.convert_tokens_to_ids("<|endoftext|>")
        self.device = device

    @torch.no_grad()
    def score_rows(self, rows, bs=48):
        """rows: (context, [candidates]) -> a score per candidate."""
        tok, start, dev = self.tok, self.start, self.device
        seqs = []  # (row, cand or -1 for the context, ids, first scored token)
        for n, (ctx, cands) in enumerate(rows):
            base = [start] + tok(ctx, add_special_tokens=False).input_ids
            firsts = []
            for j, c in enumerate(cands):
                full = [start] + tok(ctx + c, add_special_tokens=False).input_ids
                d = next((k for k, (a, b) in enumerate(zip(base, full)) if a != b), min(len(base), len(full)))
                firsts.append(d)
                seqs.append((n, j, full, d))
            seqs.append((n, -1, base, min(firsts)))
        tok_lp = {}  # (row, cand) -> log P of each token from its first scored one
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][2]))
        for s in range(0, len(order), bs):
            chunk = [seqs[i] for i in order[s:s + bs]]
            T = max(len(x[2]) for x in chunk)
            ids = torch.tensor([x[2] + [start] * (T - len(x[2])) for x in chunk], device=dev)
            att = torch.tensor([[1] * len(x[2]) + [0] * (T - len(x[2])) for x in chunk], device=dev)
            hid = self.model.model(input_ids=ids, attention_mask=att).last_hidden_state
            bi = [b for b, x in enumerate(chunk) for _ in range(x[3], len(x[2]))]
            pi = [k - 1 for x in chunk for k in range(x[3], len(x[2]))]
            tg = torch.tensor([x[2][k] for x in chunk for k in range(x[3], len(x[2]))], device=dev)
            lp = []
            for c0 in range(0, len(bi), 2048):  # the 248k-token projection in slices
                lg = self.model.lm_head(hid[torch.tensor(bi[c0:c0 + 2048], device=dev),
                                            torch.tensor(pi[c0:c0 + 2048], device=dev)]).float()
                lp += (lg.gather(1, tg[c0:c0 + 2048, None])[:, 0] - torch.logsumexp(lg, -1)).tolist()
            at = 0
            for x in chunk:
                k = len(x[2]) - x[3]
                tok_lp[(x[0], x[1])] = lp[at:at + k]
                at += k
        out = [[None] * len(c) for _, c in rows]
        d_of = {(x[0], x[1]): x[3] for x in seqs}
        for (n, j), v in tok_lp.items():
            if j < 0:
                continue
            ctx_tail = tok_lp[(n, -1)][d_of[(n, j)] - d_of[(n, -1)]:]
            out[n][j] = sum(v) - sum(ctx_tail)
        return out
