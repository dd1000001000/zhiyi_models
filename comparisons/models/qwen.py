"""Qwen3.5-2B-Base as a reference, zero-shot (not an input-method model): log P(<|endoftext|> + context + candidate)
- log P(<|endoftext|> + context), each string tokenized whole. The candidate often merges with the context's last
characters into one token (必要 + 的 -> 必要的), so the candidate tokenized apart is a split the model almost never saw;
only the tokens after the two tokenizations part are scored, and the context's own tokens from that point subtracted.

Also LIME (xushengfeng/lime, lime-v1): Qwen3-0.6B fed its preamble "下面的内容主题多样" + the text typed so far, no
start token (its init_ctx); Qwen(path, prefix=LIME_PREFIX, start=None). LIME runs an IQ4_XS GGUF of the same model and
keeps only tokens whose pinyin matches the keys; here the bf16 weights, and the candidate's plain log P.
"""
import torch


LIME_PREFIX = "下面的内容主题多样"


class Qwen:
    def __init__(self, path, device="cuda", prefix="", start="<|endoftext|>"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(device).eval()
        self.head = [self.tok.convert_tokens_to_ids(start)] if start else []
        self.pad = self.head[0] if self.head else (self.tok.pad_token_id or 0)
        self.prefix = prefix
        self.device = device

    @torch.no_grad()
    def score_rows(self, rows, bs=48):
        """rows: (context, [candidates]) -> a score per candidate."""
        tok, head, dev = self.tok, self.head, self.device
        seqs = []  # (row, cand or -1 for the context, ids, first scored token)
        for n, (ctx, cands) in enumerate(rows):
            ctx = self.prefix + ctx
            base = head + tok(ctx, add_special_tokens=False).input_ids
            firsts = []
            for j, c in enumerate(cands):
                full = head + tok(ctx + c, add_special_tokens=False).input_ids
                d = next((k for k, (a, b) in enumerate(zip(base, full)) if a != b), min(len(base), len(full)))
                firsts.append(d)
                seqs.append((n, j, full, d))
            seqs.append((n, -1, base, min(firsts)))
        tok_lp = {}  # (row, cand) -> log P of each token from its first scored one
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][2]))
        for s in range(0, len(order), bs):
            chunk = [seqs[i] for i in order[s:s + bs]]
            T = max(len(x[2]) for x in chunk)
            ids = torch.tensor([x[2] + [self.pad] * (T - len(x[2])) for x in chunk], device=dev)
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
