# 词语 n-gram 对比：知意、幽默、万象、CyIME

四个输入法的词语 n-gram 在[中立选词测试集 v2](../../benchmarks/neutral-v2/) 和 [v3](../../benchmarks/neutral-v3/) 上的首选准确率，只用 n-gram，不用任何神经模型。全部在 CPU 上单进程跑，每个 n-gram 跑一套题只要几秒。测试日期：2026-11-03。

> **说明**：测试结果仅供参考。各家的 n-gram 都是按其公开的文件格式自己实现读取和打分，没有运行各家的程序，难免与原作者的实际用法有出入；测试集、打分方式和题目来源的偏差见下文。如果发现任何问题（读取或打分有误、版本不对、对哪一家不公平等），请到 [Issues](https://github.com/dd1000001000/zhiyi_models/issues) 提出，我们会核对并更正。

![n-gram 对比](media/ngram_compare.zh-CN.svg)

![准确率与查询次数的帕累托前沿](media/ngram_pareto.zh-CN.svg)

## 测了哪些

| n-gram | 版本和文件 | 大小 | sha256 | 来源 | 许可 |
|---|---|---|---|---|---|
| 知意 ng1017 | `zh.ngram`（Release `ngram-zh-1017` 的 `zhiyi-ngram-zh-1017.zip` 里），词语三元 | 79 MB | `0a9e5c15…72cd` | 本仓库 | LGPL-2.1 |
| 幽默输入法 lm.arpa | `lm.tar.zst` 里的 `lm.arpa`，Release `engine-data-20261005-3`（2026-10-05，幽默 App 用的版本），词语三元（349,150 / 14,431,841 / 24,841,208 条），libime 的 lm_sc 与 CC-MAIN-2026-39、2026-34 网页按 0.6 混合 | 1.1 GB（压缩包 362 MB） | `2cd1fe0c…ceca`（压缩包 `be95e252…2b67`） | [Sraw/youmo-ime](https://github.com/Sraw/youmo-ime) | LGPL-2.1（仓库） |
| 万象 LMDG 语法 | `wanxiang-lts-zh-hans.gram`，Release `LTS`（2026-10-26 更新） | 418 MB | `e34500ab…e094` | [amzxyz/RIME-LMDG](https://github.com/amzxyz/RIME-LMDG) | CC BY 4.0 |
| 万象词库 | `dicts.zip`，Release `dict-nightly`（2026-11-02），用万象默认导入的 7 个：zi、jichu、cuoyin、duoyin、diming、mingren、yiren | 30 MB（zip） | `c83544a0…e243` | 同上 | CC BY 4.0 |
| CyIME 九键词组模型 | `t9_ngram-v3-ls-tri-c3.lm`，Release `models`（默认的完整版，2026-10-13 上传），词语三元，修正 Kneser-Ney，语料 LCCC + OpenSubtitles | 213 MB | `11cee2cb…c867` | [Cyletix/CyIME](https://github.com/Cyletix/CyIME) | GPL-3.0（仓库） |

测试集：`neutral-v2.jsonl`（sha256 `1d624a68…6a06ff`，计分 5,211 行）、`neutral-v3.jsonl`（`5d9dc858…e4b`，计分 10,388 行），都在本仓库的同名 Release 里。

## 怎么测

- 每行只在能拼出整串输入的候选里比（`whole`），各家候选相同；分最高的候选和答案（`best`）相同即为对。
- 每家按自己的做法打分，代码按各家公开的文件格式自己实现，没有运行任何一家的程序。

| n-gram | 上文 | 切词 | 分数 |
|---|---|---|---|
| 知意 ng1017 | 上文最后 12 字 | 在模型词表里做 Viterbi，词最长 8 字，beam 8 | ln P(候选 \| 上文)：三元，按 ARPA 回退 |
| 幽默 | 同上 | 同上（同一份代码读它的 ARPA） | 同上 |
| CyIME | 上文末尾的汉字段（最多 12 字），整段放得下时从句首 `<s>` 起；它的语料就是按汉字段、一行一句训练的 | 在它的 FNV 哈希表里查到的子串都算词，词最长 8 字，beam 8 | ln P(候选 \| 上文)：三元，回退照它的 `t9_ngram_format.h` |
| 万象，给上文 | 上文最后一行作前文 | 按万象词库的最优路径 | Rime 组句的算法：每个词 ln(词频) − ln(1e8)，加 octagram 搭配分（万象 schema：长度 2–6，搭配 −14，非搭配 −6，弱搭配 −100，句尾 −18）；一个词多个读音时取最大词频 |
| 万象，不给上文 | 无（万象默认 `contextual_suggestions: false`） | 同上 | 同上 |

需要注意的偏差：

- **知意 ng1017 是从幽默的 lm.arpa 做出来的**：把它与 CSL 科学文本、本地网页按 0.6 混合，再剪枝到 4M 二元 / 4M 三元（79 MB）。所以“幽默”一栏是原版 1.1 GB，“知意”一栏是混合并剪到约 1/14 大小后的版本。
- 幽默的 n-gram 统计用了 CC-MAIN-2026-39 的第 10–1509 号 WET 文件，v2 的题取自同一次抓取的第 2000–99999 号文件。文件号不重叠，但属于同一个月的网页，可能对它有利；v3（CC-MAIN-2026-43）没有这个问题。

- 万象的分不是概率，只比同一行内候选的顺序，不影响比较。
- CyIME 的语料是聊天和字幕，v2、v3 的网页题对它是领域外。错题里常见 js 选成“就是”而不是“技术”、yf 选成“衣服”而不是“研发”。CyIME 如果改用上文最后 12 字、不从句首起，结果几乎一样（v2 68.7，v3 71.9）。
- v3 是 2026 年 10 月的网页。万象语法文件 10-26 更新，可能有少量重叠，只会对它有利。

## 结果（首选准确率 %）

| | 计分行数 | 知意 ng1017 | 幽默 | 万象，给上文 | 万象，不给上文 | CyIME |
|---|---|---|---|---|---|---|
| v2 全部 | 5,211 | 84.5 | **85.5** | 80.8 | 75.5 | 68.4 |
| v3 全部 | 10,388 | 85.0 | **85.2** | 82.2 | 77.2 | 71.7 |
| v3 全拼 | 7,423 | 88.1 | **88.2** | 87.3 | 83.7 | 78.5 |
| v3 首字母 | 709 | 66.4 | **67.8** | 52.0 | 35.7 | 32.3 |
| v3 部分 / 混拼 | 2,256 | 80.5 | **81.0** | 74.8 | 68.9 | 61.5 |
| v3 短词 | 7,735 | 83.9 | **84.0** | 81.3 | 75.1 | 70.0 |
| v3 两词 | 913 | 88.0 | **88.4** | 85.5 | 83.7 | 76.6 |
| v3 长句 | 1,740 | 88.4 | **89.1** | 84.4 | 83.2 | 76.6 |

### 大小和计算量（图中以知意 ng1017 为 1.0）

| | 知意 ng1017 | 幽默 | 万象 | CyIME |
|---|---|---|---|---|
| 文件大小 | 79 MB（1.0） | 1.13 GB（14.3） | 473 MB（5.98） | 213 MB（2.69） |
| 条目数 | 8.35M（1.0） | 39.6M（4.75） | 7.94M（0.95） | 29.2M（3.50） |
| 每行查询次数 | 81（1.0） | 81（1.00） | 57（0.71） | 66（0.82） |

- 文件大小：发布的文件。幽默为解压后的 `lm.arpa` 文本（它的发布包是 362 MB 的 `lm.tar.zst`），万象为语法文件加默认导入的 7 个词库的文本。
- 条目数：一元、二元、三元条目之和；万象为语法里的条目加词库词条（`cost.py` 从文件里数）。
- 查询次数：给 v3 一行的整串候选打分时查概率的次数，按计分行平均。三元查询算一次（回退在内）；万象算语法查询加词频查询（给不给上文次数相同，不给上文时语法查询只查句尾）。ng1017 和幽默用同一份代码、同样的切词，所以次数相同。

幽默对知意 ng1017（逐行配对）：v2 +1.07（95% 区间 +0.42 ~ +1.80，只有幽默对 202 行、只有知意对 146 行，符号检验 p = 0.003）；v3 +0.26（−0.22 ~ +0.74，343 对 316 行，p = 0.29，不显著）。

## 复现

需要 Python 3 和 numpy。先下载上表的文件，解开幽默的 `lm.tar.zst` 和万象的 `dicts.zip`，然后（幽默的 ARPA 第一次会在旁边解析出一个 `.npz` 缓存，要几分钟）：

```bash
python run.py --v2 neutral-v2.jsonl --v3 neutral-v3.jsonl --zhiyi zh.ngram --youmo lm.arpa --cyime t9_ngram-v3-ls-tri-c3.lm --wanxiang-gram wanxiang-lts-zh-hans.gram --wanxiang-dict dicts/dicts
python cost.py --v3 neutral-v3.jsonl --zhiyi zh.ngram --youmo lm.arpa --cyime t9_ngram-v3-ls-tri-c3.lm --wanxiang-gram wanxiang-lts-zh-hans.gram --wanxiang-dict dicts/dicts
python chart.py --v2 neutral-v2.jsonl --v3 neutral-v3.jsonl
```

## 文件

| 文件 | 内容 |
|---|---|
| `run.py` | 给每行的整串候选打分，写 `scores/` 和 `results.json` |
| `cost.py` | 数文件大小、条目数和每行查询次数，写 `cost.json` |
| `cost.json` | 各 n-gram 的文件大小、条目数和每行查询次数 |
| `chart.py` | 从 `scores/` 和 `cost.json` 画 `media/ngram_compare.zh-CN.svg`：v2、v3 总分（含 95% bootstrap 区间），文件大小、条目数、查询次数（以 ng1017 为 1.0），和 `media/ngram_pareto.zh-CN.svg`（v3 准确率对查询次数的帕累托前沿） |
| `ngram_lm.py` | 知意 n-gram（`.ngram`）和幽默 ARPA 的读取和打分 |
| `cyime_ngram.py` | CyIME 紧凑格式 v3 的读取和打分 |
| `wanxiang_gram.py` | 万象 octagram 语法（Darts 双数组）和词库的读取，Rime 组句打分 |
| `scores/` | 每个 n-gram 在每套题上的分数，`{"id", "scores"}`，不比的候选为 null |
| `results.json` | 各组的行数和准确率 |
