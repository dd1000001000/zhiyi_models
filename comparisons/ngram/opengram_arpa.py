"""Open-gram's lm_sc.3gm.arpa (SourceForge open-gram, lm_sc.3gm.arpa-20140820.tar.bz2: sunpinyin's trigram, the model
libime's lm_sc started from) to a standard ARPA file for ngram_lm.NgramLM. Sunpinyin's text format has sections
"\\N-gram\\count" and lines "w1 w2 ... prob [backoff]" with plain (not log10) probabilities and backoff weights;
<unknown> is its unknown word.

  python opengram_arpa.py lm_sc.3gm.arpa lm_sc.3gm.std.arpa
"""
import math
import sys


def main():
    src, dst = sys.argv[1], sys.argv[2]
    sections, cur = {}, None
    for line in open(src, encoding="utf-8"):
        if line.startswith("\\") and "-gram\\" in line:
            cur = int(line[1:line.index("-gram")])
            sections[cur] = []
            continue
        f = line.split()
        if cur and f:
            sections[cur].append(f)
    with open(dst, "w", encoding="utf-8", newline="\n") as out:
        out.write("\\data\\\n")
        for n in sorted(k for k in sections if k > 0):
            out.write(f"ngram {n}={len(sections[n])}\n")
        for n in sorted(k for k in sections if k > 0):
            out.write(f"\n\\{n}-grams:\n")
            for f in sections[n]:
                words = ["<unk>" if w == "<unknown>" else w for w in f[:n]]
                p = max(float(f[n]), 1e-30)
                bow = f"\t{math.log10(max(float(f[n + 1]), 1e-30)):.6f}" if len(f) > n + 1 else ""
                out.write(f"{math.log10(p):.6f}\t{' '.join(words)}{bow}\n")
        out.write("\n\\end\\\n")


if __name__ == "__main__":
    main()
