"""Zhizi IME's context model Shunshou-Zhizi 30M (zhizi30m_v2.onnx, fixed tokens Int32[8,40] -> logprobs Float32[8,39],
the log P of each next token), scored as its ContextRankingScorer does: <bos> + the context's last (39 - longest
candidate) characters, then each candidate; a candidate's score is its characters' log P summed. Characters not in
itos.json are <unk>. Its input method keeps the last 32 characters committed in the same app, so the context is cut to
32 as well, and only the context's last line is given, without [URL], as for Youmo. On the CPU (6 threads).
"""
import json
import os

import numpy as np
import onnxruntime as ort

BOS, UNK = 1, 2
KEEP = 32


def clean(context):
    return context.replace("[URL]", "").split("\n")[-1]


class Zhizi:
    def __init__(self, folder, threads=6):
        itos = json.load(open(os.path.join(folder, "itos.json"), encoding="utf-8"))
        assert itos[:3] == ["<pad>", "<bos>", "<unk>"]
        self.index = {ch: i for i, ch in enumerate(itos)}
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(os.path.join(folder, "zhizi30m_v2.onnx"), so, providers=["CPUExecutionProvider"])

    def encode(self, text):
        return [self.index.get(ch, UNK) for ch in text]

    def score(self, context, cands):
        longest = max(len(c) for c in cands)
        ctx = self.encode(clean(context)[-KEEP:])
        prefix = [BOS] + ctx[max(0, len(ctx) - (39 - longest)):]
        start = len(prefix) - 1
        out = []
        for s in range(0, len(cands), 8):
            batch = cands[s:s + 8]
            tokens = np.zeros((8, 40), np.int32)
            for i, c in enumerate(batch):
                seq = prefix + self.encode(c)
                tokens[i, :len(seq)] = seq
            lp = self.sess.run(["logprobs"], {"tokens": tokens})[0].astype(np.float64)
            out += [float(lp[i, start:start + len(c)].sum()) for i, c in enumerate(batch)]
        return out

    def score_rows(self, rows):
        """rows: (context, [candidates]) -> summed log P per candidate."""
        return [self.score(ctx, cands) for ctx, cands in rows]
