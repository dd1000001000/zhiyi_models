"""Qingjian's 含章 models (qingjian-team/qingjian) through its own Rust code: qingjian/qj-score/ (build it inside a
Qingjian checkout, see qingjian/README.md) reads rows and calls qingjian-neural's CharScorer.
- 含章·通变 (hanzhang-tongbian-small.qjm, release data-v9): P2C, log P(candidate | keys typed), the text before the caret
  not used (its P2cScorer: <eos> keys <sep> characters).
- 含章·知微 (hanzhang-zhiwei-small.qjm, release data-v7; taken offline in 0.3.2): log P(candidate | text before the caret),
  the keys not used (its CharScorer: <eos> context candidate).
A row where some candidate has a character outside the model's vocabulary is not scored (Qingjian keeps its static
order then): its candidates get -10000, -10001, ... in the engine's order, so the pick is the first one.
"""
import json
import subprocess


def score_rows(exe, mode, model, rows):
    """rows: (context, code typed, [candidates]) -> scores (None where not scored)."""
    text = "".join(json.dumps({"id": i, "context": c, "code": k, "cands": cands}, ensure_ascii=False) + "\n"
                   for i, (c, k, cands) in enumerate(rows))
    res = subprocess.run([exe, mode, model], input=text, capture_output=True, text=True, encoding="utf-8", check=True)
    got = {q["id"]: q["scores"] for q in map(json.loads, res.stdout.splitlines())}
    return [got[i] if any(x is not None for x in got[i]) else [-10000.0 - j for j in range(len(got[i]))]
            for i in range(len(rows))]
