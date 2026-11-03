"""Rerankers that ask a chat LLM to pick the best candidate, as Rime plug-ins do, scored per candidate:
log P(the model answers with that candidate) after the plug-in's own system and user prompts.

- rime-ai-ime: oc160/Rime-AI-IME (release v0.1.0-model): Qwen3.5-0.8B + its LoRA adapter_v1 (the same weights as its
  qwen35-08b-ime-v1 GGUF, here in bf16), prompts from its tools/sidecar.py; the answer is the word alone, so the score
  is log P(word + <|im_end|>). Thinking off, as its requests ask.
- fujik0: FuJiK0/rime-llm-reranker: Qwen3.5-0.8B as released (no fine-tuning), prompts from its rime.lua; it answers a
  ranked list joined by 、, so the score is log P(word + 、) as the list's first item.
Both plug-ins give the text before the caret (their context_length: the last 60 characters), the pinyin as typed and
the candidates. They show Rime's first 5 candidates; here every row's whole candidates are listed, in the engine's
order, so all models compare the same candidates.
"""
import torch

PROMPTS = {
    "rime-ai-ime": {
        "system": "你是输入法候选词排序助手。根据对话历史和当前输入，从候选词列表中选出最符合语境的词。只输出这1个词本身，不要拼音、数字序号或任何解释。",
        "with": "对话历史：「{ctx}」\n当前输入拼音：「{py}」\n候选词：{cands}\n只输出最符合对话语境的那1个候选词：",
        "without": "输入拼音：「{py}」\n候选词：{cands}\n只输出最可能的那1个候选词：",
        "answer": "{w}<|im_end|>",
    },
    "fujik0": {
        "system": "你是输入法候选词排序助手。根据对话历史和当前输入，从候选词列表中选出最符合语境的词。按优先级从高到低列出最多5个候选词，用顿号分隔，不要任何解释。",
        "with": "对话历史：「{ctx}」\n当前输入拼音：「{py}」\n候选词：{cands}\n按最符合对话语境的顺序列出前5个候选词（顿号分隔）：",
        "without": "输入拼音：「{py}」\n候选词：{cands}\n按最可能的顺序列出前5个候选词（顿号分隔）：",
        "answer": "{w}、",
    },
}
CONTEXT_CHARS = 60


class ChatJudge:
    def __init__(self, base, style, adapter=None, device="cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(base)
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        self.model = model.to(device).eval()
        self.p, self.device = PROMPTS[style], device

    def prompt(self, ctx, py, cands):
        ctx = ctx[-CONTEXT_CHARS:]
        user = (self.p["with"] if ctx else self.p["without"]).format(ctx=ctx, py=py, cands="、".join(cands))
        msgs = [{"role": "system", "content": self.p["system"]}, {"role": "user", "content": user}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    @torch.no_grad()
    def score_rows(self, rows, bs=32):
        """rows: (context, pinyin typed, [candidates]) -> log P of answering each candidate."""
        seqs = []  # (row, cand, ids, first answer token)
        for n, (ctx, py, cands) in enumerate(rows):
            head = self.tok(self.prompt(ctx, py, cands), add_special_tokens=False).input_ids
            for j, c in enumerate(cands):
                full = self.tok(self.prompt(ctx, py, cands) + self.p["answer"].format(w=c), add_special_tokens=False).input_ids
                assert full[:len(head)] == head
                seqs.append((n, j, full, len(head)))
        out = [[None] * len(c) for _, _, c in rows]
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][2]))
        pad = self.tok.pad_token_id or 0
        for s in range(0, len(order), bs):
            chunk = [seqs[i] for i in order[s:s + bs]]
            T = max(len(x[2]) for x in chunk)
            ids = torch.tensor([x[2] + [pad] * (T - len(x[2])) for x in chunk], device=self.device)
            att = torch.tensor([[1] * len(x[2]) + [0] * (T - len(x[2])) for x in chunk], device=self.device)
            hid = self.model.model(input_ids=ids, attention_mask=att).last_hidden_state
            bi = [b for b, x in enumerate(chunk) for _ in range(x[3], len(x[2]))]
            pi = [k - 1 for x in chunk for k in range(x[3], len(x[2]))]
            tg = torch.tensor([x[2][k] for x in chunk for k in range(x[3], len(x[2]))], device=self.device)
            lg = self.model.lm_head(hid[torch.tensor(bi, device=self.device), torch.tensor(pi, device=self.device)]).float()
            lp = (lg.gather(1, tg[:, None])[:, 0] - torch.logsumexp(lg, -1)).tolist()
            at = 0
            for x in chunk:
                k = len(x[2]) - x[3]
                out[x[0]][x[1]] = sum(lp[at:at + k])
                at += k
            if (s // bs) % 100 == 99:
                torch.cuda.empty_cache()
        return out
