"""Size, entries and lookups of each n-gram, for the chart (relative to Zhiyi ng1017 = 1.0).

- File size: the file as released (Youmo: lm.arpa unpacked; Wanxiang: the grammar file plus the 7 dictionary tables
  it imports, as text).
- Entries (an n-gram's "parameters"): unigrams + bigrams + trigrams; Wanxiang: the grammar's keys plus the tables'
  entries.
- Lookups (its "compute"): probability queries per row while scoring neutral-v3's whole candidates, averaged over
  the scored rows: trigram queries (backoff included in one); Wanxiang: grammar queries plus dictionary weights.

  python cost.py --v3 neutral-v3.jsonl --zhiyi zh.ngram --youmo lm.arpa --cyime t9_ngram-v3-ls-tri-c3.lm
                 --wanxiang-gram wanxiang-lts-zh-hans.gram --wanxiang-dict <dicts.zip unpacked>/dicts
                 --libime lm_sc.arpa,sc.lm --opengram lm_sc.3gm.std.arpa,lm_sc.3gm.arpa --libpinyin <model20.text unpacked>
                 --rime-gram zh-hans-t-essay-bgw.gram --rime-essay essay.txt --rime-opencc <dir> --qingjian-lm <qj-score>,lm.qj
                 --ximegboard-gram XimeGboard.gram --ximegboard-dict <its APK's assets/rime/cn_dicts>
-> cost.json (merged into the existing one: give only the n-grams to count)
File sizes of the new ones: libime its built sc.lm (kenlm binary, the file fcitx5 installs; chinese-ime-lm's release
language-model-libime-20260629 carries the same build), open-gram its lm_sc.3gm.arpa text, libpinyin interpolation2.text,
Rime the grammar plus essay.txt, Qingjian lm.qj, XimeGboard the grammar plus the 5 rime_ice tables (text).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


class CountingDict(dict):
    n = 0

    def get(self, k, default=None):
        CountingDict.n += 1
        return super().get(k, default)


def lookups(rows, score, counter):
    before = counter()
    for r in rows:
        score(r["context"], [r["candidates"][j] for j in r["whole"]])
    return (counter() - before) / len(rows)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    for k in ("v3", "zhiyi", "youmo", "cyime", "wanxiang-gram", "wanxiang-dict", "libime", "opengram", "libpinyin",
              "rime-gram", "rime-essay", "rime-opencc", "qingjian-lm", "ximegboard-gram", "ximegboard-dict"):
        p.add_argument("--" + k)
    a = p.parse_args()
    rows = [r for r in map(json.loads, open(a.v3, encoding="utf-8")) if r["scored"]]
    out = {}

    def add(name, size, entries, per_row):
        out[name] = {"file_mb": size / 1e6, "entries": entries, "lookups_per_row": per_row}
        print(f"{name:<16} {size / 1e6:8.1f} MB {entries / 1e6:8.2f} M entries {per_row:8.1f} lookups/row", flush=True)

    from ngram_lm import NgramLM
    for name, path in (("zhiyi-ng1017", a.zhiyi), ("youmo-lm", a.youmo)):
        if not path:
            continue
        lm = NgramLM(path)
        calls = [0]
        lp3 = lm.lp3

        def counted(u, v, w, lp3=lp3, calls=calls):
            calls[0] += 1
            return lp3(u, v, w)
        lm.lp3 = counted
        add(name, os.path.getsize(path), len(lm.words) + len(lm.k2) + len(lm.k3),
            lookups(rows, lm.score, lambda: calls[0]))

    if a.cyime:
        from cyime_ngram import CyimeNgram
        cy = CyimeNgram(a.cyime)
        calls = [0]
        lp3 = cy.lp3

        def counted(u, v, w):
            calls[0] += 1
            return lp3(u, v, w)
        cy.lp3 = counted
        add("cyime-t9-v3", os.path.getsize(a.cyime), cy.words + cy.bigrams + cy.trigrams,
            lookups(rows, cy.score, lambda: calls[0]))

    if a.wanxiang_gram:
        from wanxiang_gram import TABLES, WanxiangGram
        wg = WanxiangGram(a.wanxiang_gram, a.wanxiang_dict)
        keys = int(((wg.trie.a >> 31) & 1).sum())
        entries = 0
        for t in TABLES:
            body = False
            for line in open(os.path.join(a.wanxiang_dict, t + ".dict.yaml"), encoding="utf-8"):
                if not body:
                    body = line.startswith("...")
                    continue
                f = line.split("\t")
                entries += len(f) >= 2 and bool(f[0]) and not f[0].startswith("#")
        size = os.path.getsize(a.wanxiang_gram) + sum(os.path.getsize(os.path.join(a.wanxiang_dict, t + ".dict.yaml"))
                                                      for t in TABLES)
        wg.weights = CountingDict(wg.weights)
        q = [0]
        query = wg.query

        def counted_q(context, word, rear):
            q[0] += 1
            return query(context, word, rear)
        wg.query = counted_q
        for name, ctx in (("wanxiang-ctx", True), ("wanxiang-noctx", False)):
            add(name, size, keys + entries, lookups(rows, lambda c, cands: wg.score(c, cands, ctx=ctx),
                                                    lambda: q[0] + CountingDict.n))

    for name, spec in (("libime-lm-sc", a.libime), ("opengram-sunpinyin", a.opengram)):
        if not spec:
            continue
        arpa, _, shipped = spec.partition(",")  # shipped: the file the project installs, when not the ARPA itself
        lm = NgramLM(arpa)
        calls = [0]
        lp3 = lm.lp3

        def counted(u, v, w, lp3=lp3, calls=calls):
            calls[0] += 1
            return lp3(u, v, w)
        lm.lp3 = counted
        add(name, os.path.getsize(shipped or arpa), len(lm.words) + len(lm.k2) + len(lm.k3),
            lookups(rows, lm.score, lambda calls=calls: calls[0]))

    if a.libpinyin:
        from libpinyin_lm import LibpinyinLM
        lp = LibpinyinLM(a.libpinyin)
        calls = [0]
        f = lp.lp

        def counted_lp(v, w):
            calls[0] += 1
            return f(v, w)
        lp.lp = counted_lp
        add("libpinyin-model20", os.path.getsize(os.path.join(a.libpinyin, "interpolation2.text")),
            len(lp.u) + len(lp.bi), lookups(rows, lp.score, lambda: calls[0]))

    if a.rime_gram:
        from wanxiang_gram import RIME_CFG, WanxiangGram
        rg = WanxiangGram(a.rime_gram, essay=a.rime_essay, cfg=RIME_CFG, opencc=a.rime_opencc)
        keys = int(((rg.trie.a >> 31) & 1).sum())
        CountingDict.n = 0
        rg.weights = CountingDict(rg.weights)
        q = [0]
        query = rg.query

        def counted_rq(context, word, rear):
            q[0] += 1
            return query(context, word, rear)
        rg.query = counted_rq
        add("rime-octagram-hans", os.path.getsize(a.rime_gram) + os.path.getsize(a.rime_essay), keys + len(rg.weights),
            lookups(rows, lambda c, cands: rg.score(c, cands, ctx=True), lambda: q[0] + CountingDict.n))

    if a.ximegboard_gram:
        from run import XIMEGBOARD_TABLES
        from wanxiang_gram import RIME_CFG, WanxiangGram
        xg = WanxiangGram(a.ximegboard_gram, a.ximegboard_dict, cfg=RIME_CFG, tables=XIMEGBOARD_TABLES)
        keys = int(((xg.trie.a >> 31) & 1).sum())
        entries = 0
        for t in XIMEGBOARD_TABLES:
            body = False
            for line in open(os.path.join(a.ximegboard_dict, t + ".dict.yaml"), encoding="utf-8"):
                if not body:
                    body = line.startswith("...")
                    continue
                f = line.split("\t")
                entries += len(f) >= 2 and bool(f[0]) and not f[0].startswith("#")
        size = os.path.getsize(a.ximegboard_gram) + sum(
            os.path.getsize(os.path.join(a.ximegboard_dict, t + ".dict.yaml")) for t in XIMEGBOARD_TABLES)
        CountingDict.n = 0
        xg.weights = CountingDict(xg.weights)
        q = [0]
        query_x = xg.query

        def counted_xq(context, word, rear):
            q[0] += 1
            return query_x(context, word, rear)
        xg.query = counted_xq
        add("ximegboard-octagram", size, keys + entries,
            lookups(rows, lambda c, cands: xg.score(c, cands, ctx=True), lambda: q[0] + CountingDict.n))

    if a.qingjian_lm:
        import subprocess
        exe, lmqj = a.qingjian_lm.split(",")
        text = "".join(json.dumps({"id": i, "context": r["context"], "code": "", "cands": [r["candidates"][j] for j in r["whole"]]},
                                  ensure_ascii=False) + chr(10) for i, r in enumerate(rows))
        res = subprocess.run([exe, "lm", lmqj], input=text, capture_output=True, text=True, encoding="utf-8", check=True)
        info = dict(line.split(" ", 1) for line in res.stderr.strip().splitlines() if " " in line)
        words, bigrams = map(int, info["entries"].split())
        add("qingjian-lm", os.path.getsize(lmqj), words + bigrams, int(info["lookups"]) / len(rows))

    path = os.path.join(HERE, "cost.json")
    old = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    json.dump(old | out, open(path, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
