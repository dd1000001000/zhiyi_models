"""zhanghaozhecn/rime-llm-rerank (the Weasel plugin; same scoring in rime-llm-ime) with its default model
Qwen3-0.6B-Base-Q4_K_M.gguf (ModelScope zhanghaozhecn/Qwen3-0.6B-Base-GGUF), scored as its rime_llm.cpp score_batch
(commit 73c7282): the context with all whitespace removed (llm_processor.lua), tokenized, its last
max_tokens = 20 tokens (llm_filter.lua default since 2026-11-19); each candidate tokenized alone, only its first three
tokens count: score = -(1.0 CE1 + 1.10 CE2 + 0.60 CE3) (kCeW, fewer terms for shorter candidates). With no context the
plugin does not rerank: those rows keep the engine order (scores null). Model alone: no word-frequency fusion
(beta log1p(eff) + log(0.5 p1)), every whole candidate scored, not only its window of the first 6.
The Q4_K_M weights are dequantized by transformers' GGUF loader and run in float32 on CUDA (llama.cpp would quantize
the activations to 8 bits on the CPU; the weights are the same).
"""
import torch

W = (1.0, 1.10, 0.60)
MAX_TOKENS = 20


class LlmRerank:
    def __init__(self, folder, gguf, device="cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(folder, gguf_file=gguf)
        self.m = AutoModelForCausalLM.from_pretrained(folder, gguf_file=gguf, torch_dtype=torch.float32).to(device).eval()
        self.dev = device

    def ids(self, text):
        return self.tok(text, add_special_tokens=True)["input_ids"]

    @torch.no_grad()
    def score(self, context, cands):
        ctx = self.ids("".join((context or "").split()))[-MAX_TOKENS:]
        if not ctx:
            return [None] * len(cands)
        toks = [self.ids(c)[:3] for c in cands]
        w = max(len(t) for t in toks)
        seqs = [ctx + t[:2] + [0] * (2 - len(t[:2])) for t in toks]  # positions ctx-1 .. ctx+1 predict t0 .. t2
        x = torch.tensor(seqs, device=self.dev)
        lp = torch.log_softmax(self.m(input_ids=x).logits[:, len(ctx) - 1:len(ctx) - 1 + w].float(), -1)
        out = []
        for i, t in enumerate(toks):
            out.append(-sum(W[j] * -lp[i, j, tok].item() for j, tok in enumerate(t)) if t else None)
        return out
