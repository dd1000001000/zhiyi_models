# 选词 / 组句模型对比：知意、PinyinGPT、水杉、幽默、吵吵

几个输入法的神经排序模型在[中立选词测试集 v2](../../benchmarks/neutral-v2/) 和 [v3](../../benchmarks/neutral-v3/) 上的首选准确率，只用模型，不用任何引擎和 n-gram；另附一项整句组句测试。Qwen3.5-2B-Base 作为通用语言模型的参考。测试日期：2026-11-03。

> **说明**：测试结果仅供参考。除知意外，各家模型都是按其公开的文件格式和源码自己实现读取和打分（PinyinGPT 按其论文代码的输入方式），没有运行各家的输入法，难免与原作者的实际用法有出入；测试集、打分方式的偏差见下文。如果发现任何问题（读取或打分有误、版本不对、对哪一家不公平等），请到 [Issues](https://github.com/dd1000001000/zhiyi_models/issues) 提出，我们会核对并更正。

![模型对比](media/models_compare.zh-CN.svg)

![准确率与计算量的帕累托前沿](media/models_pareto.zh-CN.svg)

## 测了哪些

| 模型 | 版本和文件 | 大小 | sha256 | 来源 | 许可 |
|---|---|---|---|---|---|
| 知意 ML-pumpkin-1026 | Release `model-ml-pumpkin-1026` 的 `laya-ml-pumpkin-1026-int8g.zip` 里的 `laya.int8g.onnx`，12 层 | 92 MB | `bbf51918…7a06` | 本仓库 | Apache-2.0 |
| 知意 ML-Flash-1031 | Release `model-ml-flash-1031` 的 `laya-ml-flash-1031-int8g.zip` 里的 `laya.int8g.onnx`，8 层 | 75 MB | `4cd47566…92ca` | 本仓库 | Apache-2.0 |
| 知意 pumpkin-1103 | 内部版本，12 层，同结构续训后导出的 int8 ONNX；**未发布，权重不公开** | 83 MB | `5ca51a94…eb08` | — | — |
| 知意 Flash-1103 | 内部版本，8 层，同上；**未发布，权重不公开** | 75 MB | `6396bffe…d7b7` | — | — |
| PinyinGPT-Concat | Hugging Face `aihijo/transformers4ime-pinyingpt-concat` 的 `pytorch_model.bin`：GPT-2 12 层 × 768，词表 21,128 + 443 个拼音 token | 489 MB | `1c5ebb9e…74bb` | Tan 等，ACL 2022（腾讯 AI Lab）；代码 [VisualJoyce/Transformers4IME](https://github.com/VisualJoyce/Transformers4IME) `8f1573e` | CC BY-NC-SA 4.0 |
| GPT2-zh-21k | Hugging Face `aihijo/gpt2-zh-21k` 的 `model.safetensors`：同一论文里冻结使用的中文 GPT | 421 MB | `645284b6…78c6` | 同上 | CC BY-NC-SA 4.0 |
| 水杉 rerank-v1 8x448 | [metasequoiaime/chinese-ime-lm](https://github.com/metasequoiaime/chinese-ime-lm) Release `rerank-v1`（2026-11-01）的 `rerank-8x448.ort` 及其 `.model.json`、`.vocab.json` | 31 MB | `ad729f94…4a93` | 同左 | Apache-2.0 |
| 水杉 rerank-v1 6x256 | 同上，`rerank-6x256.ort` | 8.2 MB | `ceeb3a8d…bad4` | 同左 | Apache-2.0 |
| 幽默输入法 句子模型 25M | [Sraw/youmo-ime](https://github.com/Sraw/youmo-ime) Release `sentence-models-20261001` 的 `sentence-model-large.safetensors` | 25 MB | `6f7fcb72…669a` | 同左 | Apache-2.0 |
| 吵吵拼音 CLMG v2 | [yangchaoliuyang/chaoinput](https://github.com/yangchaoliuyang/chaoinput) Release `v3.3.16`：`ChaoInput-v3.3.16-win64.zip` 里的 `data/lmt.bin` 和 `data/dict.db`，结构按源码包的 `ChaoInput.LmTrain.Gpu/model.py` | 115 MB（`lmt.bin`） | `f9052324…f8b8`（`dict.db` `51fd6c85…fd1b`） | 同左 | MIT |
| Qwen3.5-2B-Base（参考） | Hugging Face `Qwen/Qwen3.5-2B-Base` | 4.55 GB | `928acbf1…df2d` | 通义千问 | Apache-2.0 |

测试集：`neutral-v2.jsonl`（sha256 `1d624a68…6a06ff`，计分 5,211 行）、`neutral-v3.jsonl`（`5d9dc858…fe4b`，计分 10,388 行），都在本仓库的同名 Release 里。组句测试用我们内部的中立集 v1（12,000 个长分句，`35131f7f…4a5a`），**未公开**，所以组句结果别人暂时无法复现。

## 怎么测

- 每行只在能拼出整串输入的候选里比（`whole`），各家候选相同；分最高的候选和答案（`best`）相同即为对。
- 每家按自己的方式打分，代码见下表的文件。知意以外的打分和以前的私有测试结果逐一核对过：吵吵、水杉、幽默的分数差在 1e-4 以内，Qwen（bf16）首选全部一致。

| 模型 | 上文 | 拼音怎么给 | 候选怎么给 | 分数 |
|---|---|---|---|---|
| 知意 | 最后 128 个词元 | 打的码按首选候选的读音切成音节槽，全拼或首字母照原样 | 整页候选一起进模型（包括不能拼出整串的） | 选词头对每个候选的 logit |
| PinyinGPT-Concat | `[CLS]` 加上文最后 100 字（它训练时最长 128 个 token） | `[SEP]` 后接每个音节的拼音 token。**它发布的模型只用首字母训练过**（仓库的 `pretrain_pinyingpt.sh`：`--abbr_mode full`），所以输入一律用首字母 token（`[z]`）；每个字的概率只在所打拼音允许的字里算（全拼音节用该音节的字，只打首字母时用该字母的字） | 候选的字逐个强制喂入，位置号沿用对应拼音 token 的位置（同其 `pinyingpt_concat.py`） | 各字受约束的 log P 之和 |
| GPT2-zh-21k | 同上 | 不进模型，只做约束（同其 `pinyingpt_compatible.py`） | 候选逐字接在上文后 | 同上 |
| 水杉 | `<bos>` 加最后 64 字 | 不给，它是字级语言模型 | 按它运行时逐键的方式：候选截成 1 字、2 字……逐页喂入，前缀共用缓存 | 最后一页的分数（每字平均 log P） |
| 幽默 | 只给上文最后一行（去掉 `[URL]` 和换行，它的词表里没有） | 不给 | 候选逐字接在上文后 | 各字 log P 之和 |
| 吵吵 | 最后 64 字 | 只用来查它词典里这个候选的词频 | 每个候选单独算 | log(10 + 词频) + 上文向量 · 候选字向量的平均 |
| Qwen3.5-2B（参考） | `<\|endoftext\|>` 加上文 | 不给 | 上文加候选整串一起分词 | log P(上文 + 候选) − log P(上文) |

需要注意的偏差：

- **PinyinGPT 的输入方式**：如果按所打的样子喂全拼 token（`[zhong]`），它的成绩是 v2 88.2、v3 89.4，比按训练方式喂首字母低 5–6 分；图和主表用的是后者。论文的做法是全拼交给冻结的 GPT（即 GPT2-zh-21k）、简拼交给 PinyinGPT，这里两者都单独列出。
- **上文长度**：PinyinGPT 只看最后 100 字。知意也在只给最后 100 字时重跑了一遍，四个模型变化都在 0.4 分以内（pumpkin-1103：v2 93.5 → 93.5，v3 93.8 → 93.6），所以差距不来自上文长度。
- **计算量不同**：PinyinGPT、GPT2 对每个候选单独跑一遍（上文可共用缓存），每个字算一次整个词表的 softmax；知意一次前向给整页候选打分。v3 每行的理论计算量 PinyinGPT 是 ML-Flash-1031 的 4.2 倍，文件大 6.5 倍。
- PinyinGPT 的拼音表里没有某个字的这个读音时，把这个字加进该位置允许的字里，不让它直接得零分；它拼不出的候选（拉丁字母、词表外的字）记 −1e4，这些候选在两套题里都不是答案。
- 水杉、幽默在各自的输入法里还要加上解码器 / n-gram 的分数，这里只比模型本身，不能代表它们输入法的整体表现。
- PinyinGPT、GPT2-zh-21k 为 CC BY-NC-SA 4.0，这里只做评测，不分发其权重。
- v3 是 2026 年 10 月的网页，晚于各家的训练数据；知意 1103 两个模型训练时没有用到 v2、v3、v1 的任何内容。

## 结果：选词（首选准确率 %）

| | 行数 | 知意 pumpkin-1103 | 知意 Flash-1103 | 知意 ML-pumpkin-1026 | 知意 ML-Flash-1031 | PinyinGPT-Concat | GPT2-zh-21k | 水杉 8x448 | 水杉 6x256 | 幽默 25M | 吵吵 | Qwen 2B（参考） |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v2 全部 | 5,211 | 93.5 | 93.3 | 93.1 | 93.2 | 94.1 | 90.2 | 90.0 | 88.6 | 86.8 | 67.2 | **97.8** |
| v3 全部 | 10,388 | 93.8 | 93.3 | 93.1 | 93.1 | 94.1 | 90.5 | 91.2 | 90.0 | 86.8 | 73.1 | **97.7** |
| v3 全拼 | 7,423 | 95.3 | 95.0 | 94.8 | 94.8 | 95.2 | 92.1 | 92.7 | 91.9 | 89.5 | 82.1 | **98.0** |
| v3 首字母 | 709 | 83.9 | 82.4 | 82.5 | 82.5 | 87.7 | 82.2 | 82.1 | 78.7 | 70.1 | 50.9 | **95.6** |
| v3 部分 / 混拼 | 2,256 | 92.2 | 91.2 | 90.7 | 91.1 | 92.5 | 87.5 | 89.1 | 87.5 | 83.2 | 50.7 | **97.3** |
| v3 短词 | 7,735 | 93.5 | 92.9 | 92.7 | 92.8 | 93.6 | 90.0 | 90.2 | 88.8 | 85.8 | 77.2 | **97.3** |
| v3 两词 | 913 | 95.1 | 93.5 | 94.2 | 93.3 | 95.6 | 91.7 | 93.5 | 93.1 | 89.6 | 66.3 | **98.7** |
| v3 长句 | 1,740 | 94.7 | 94.9 | 94.1 | 94.6 | 95.4 | 92.0 | 94.3 | 93.9 | 89.7 | 58.8 | **98.7** |

PinyinGPT-Concat 对知意（逐行配对，McNemar）：

- 对 pumpkin-1103：v2 +0.6（95% 区间 −0.2 ~ +1.2，p = 0.11），v3 +0.2（−0.3 ~ +0.7，p = 0.37），都不显著。
- 对 ML-Flash-1031：v2 +0.9（+0.2 ~ +1.6，p = 0.019），v3 +0.9（+0.5 ~ +1.4，p = 0.0003）。
- 差距主要在只打首字母的行（v3 87.7 对 82.5–83.9）。

### 大小和计算量（图中以知意 ML-Flash-1031 为 1.0）

| 模型 | 文件大小 | 总参数量 | 理论计算量（v3 每行乘加） |
|---|---|---|---|
| 知意 pumpkin-1103 | 83 MB（1.10） | 80.8M（1.10） | 3.6 G（1.40） |
| 知意 Flash-1103 | 75 MB（1.00） | 73.2M（1.00） | 2.6 G（1.00） |
| 知意 ML-pumpkin-1026 | 92 MB（1.22） | 84.4M（1.15） | 3.6 G（1.40） |
| 知意 ML-Flash-1031 | 75 MB（1.00） | 73.2M（1.00） | 2.6 G（1.00） |
| PinyinGPT-Concat | 489 MB（6.48） | 102.4M（1.40） | 10.9 G（4.21） |
| GPT2-zh-21k | 421 MB（5.58） | 102.1M（1.40） | 8.8 G（3.42） |
| 水杉 8x448 | 31 MB（0.41） | 24.9M（0.34） | 1.4 G（0.56） |
| 水杉 6x256 | 8.2 MB（0.11） | 7.9M（0.11） | 0.33 G（0.13） |
| 幽默 25M | 25 MB（0.34） | 24.7M（0.34） | 0.98 G（0.38） |
| 吵吵 CLMG v2 | 115 MB（1.52） | 28.7M（0.39） | 1.2 G（0.47） |
| Qwen3.5-2B（参考） | 4.55 GB（60.3） | 1.88B（25.7） | 124 G（48.2） |

- 文件大小：发布的模型文件。知意为 int8 ONNX，水杉为 `.ort`，幽默为 int8 safetensors，吵吵为 fp32 的 `lmt.bin`（不含词典 `dict.db`），PinyinGPT / GPT2 为 fp32 权重，Qwen 为 bf16 且含视觉部分。
- 总参数量：文件里的张量个数。知意按 ONNX 的权重数，其中约 4,875 万是词嵌入；水杉的 `.ort` 无法直接读出张量，按其 `model.json` 的结构算；Qwen 只算语言模型部分。
- 理论计算量：从头给一行的整串候选打分的乘加次数，按 v3 计分行平均（`cost.py`）。上文只算一次（因果模型的上文 K/V 各候选共用），各词元过线性层，注意力按各词元看到的长度算，逐字算整个词表 softmax 的模型另加输出层（水杉 6x256 是自归一化的，只算候选字那一项）。知意为整页一次前向（编码器 + 选词头，不算组句头）；在输入法里知意 Flash 的上文可以缓存，这里没有扣除。

## 结果：组句（P@1 / P@5 %）

整句的拼音一次打完，模型直接给出整句，和原句完全相同才算对。题目是中立集 v1 的 12,000 个长分句（4–16 字），全拼、末字只打首字母、混拼、全部首字母各 3,000 句，上文为空、1–32 字、33–128 字各 4,000 句。

| | 行数 | 知意 pumpkin-1103 | 知意 Flash-1103 | 知意 ML-pumpkin-1026 | 知意 ML-Flash-1031 | PinyinGPT-Concat | GPT2-zh-21k |
|---|---|---|---|---|---|---|---|
| 全部 | 12,000 | 33.2 / 44.0 | 30.6 / 42.2 | 30.7 / 41.8 | 30.5 / 41.5 | **48.2** / 64.2 | 31.4 / 46.4 |
| 全拼 | 3,000 | 58.3 / 72.8 | 55.0 / 71.1 | 55.4 / 70.8 | 55.2 / 70.0 | **63.0** / 78.8 | 42.9 / 62.4 |
| 末字只打首字母 | 3,000 | 45.6 / 60.3 | 42.1 / 58.0 | 41.5 / 56.8 | 41.6 / 57.4 | **59.7** / 76.9 | 43.2 / 62.1 |
| 混拼 | 3,000 | 21.9 / 32.0 | 19.5 / 29.8 | 19.8 / 29.6 | 19.2 / 29.3 | **41.5** / 57.6 | 24.4 / 37.5 |
| 全部首字母 | 3,000 | 6.9 / 10.9 | 5.9 / 10.0 | 6.2 / 9.9 | 5.9 / 9.1 | **28.6** / 43.2 | 15.2 / 23.4 |
| 无上文 | 4,000 | 22.1 / 30.9 | 19.7 / 29.4 | 20.2 / 28.9 | 19.6 / 29.2 | **33.4** / 49.4 | 19.6 / 31.8 |
| 上文 1–32 字 | 4,000 | 34.5 / 46.1 | 31.1 / 43.8 | 31.6 / 43.5 | 31.2 / 42.6 | **50.7** / 67.3 | 32.4 / 47.9 |
| 上文 33–128 字 | 4,000 | 42.9 / 55.0 | 41.2 / 53.5 | 40.5 / 53.0 | 40.6 / 52.5 | **60.5** / 75.7 | 42.2 / 59.4 |

- 知意：组句头一次前向，所有字位同时给出，各字位取前 5 个字后组合出 n-best。知意输入法里的组句还会多轮修正并加 n-gram，这里没有用。
- PinyinGPT / GPT2-zh-21k：受拼音约束的 beam search，16 束，同论文的评测设置；PinyinGPT 的输入同上，用首字母 token。逐字解码，每句的计算量比知意的一次前向大得多。
- 只比原句完全相同，同音的其他合理写法也算错。

## 复现

需要 Python 3、numpy、PyTorch（CUDA）、transformers、safetensors、tokenizers、pypinyin、onnx、onnxruntime；知意的 ONNX 在 DirectML 上跑时用 onnxruntime-directml（加 `--cpu` 则用 CPU），所以分两个环境跑。先下载上表的文件，解开知意的 int8 zip 和吵吵的两个 zip：

```bash
python run.py --v2 neutral-v2.jsonl --v3 neutral-v3.jsonl --zhiyi zhiyi-pumpkin-1026=laya-ml-pumpkin-1026-int8g --zhiyi zhiyi-flash-1031=laya-ml-flash-1031-int8g --metasequoia rerank-v1
```

```bash
python run.py --v2 neutral-v2.jsonl --v3 neutral-v3.jsonl --youmo sentence-model-large.safetensors --chaoinput ChaoInput-v3.3.16-win64/data,ChaoInput-src-v3.3.16/src/ChaoInput.LmTrain.Gpu/model.py --qwen Qwen3.5-2B-Base --pinyingpt transformers4ime-pinyingpt-concat --gpt2-zh-21k gpt2-zh-21k
```

```bash
python cost.py --v3 neutral-v3.jsonl --zhiyi zhiyi-flash-1031=laya-ml-flash-1031-int8g --metasequoia rerank-v1 --youmo sentence-model-large.safetensors --chaoinput ChaoInput-v3.3.16-win64/data,ChaoInput-src-v3.3.16/src/ChaoInput.LmTrain.Gpu/model.py --qwen Qwen3.5-2B-Base --pinyingpt transformers4ime-pinyingpt-concat --gpt2-zh-21k gpt2-zh-21k
```

```bash
python chart.py --v2 neutral-v2.jsonl --v3 neutral-v3.jsonl
```

其他选项：`run.py --context_chars 100` 只给知意最后 100 字上文（结果名加 `-ctx100`）；`run.py --pinyingpt-typed <目录>` 按所打的样子给 PinyinGPT 喂拼音（`pinyingpt-concat-typed`）。组句用 `compose.py --v1 <v1 测试集> --zhiyi … --pinyingpt … --gpt2-zh-21k …`。

## 文件

| 文件 | 内容 |
|---|---|
| `run.py` | 选词：给每行的候选打分，写 `scores/` 和 `results.json` |
| `compose.py` | 组句：每行输出前 5 句，写 `compose/`（不随仓库发布：很多输出就是未公开的 v1 原句）和 `results.json` 的 `compose` |
| `cost.py` | 数文件大小、参数量，算每行理论计算量，写 `cost.json` |
| `chart.py` | 从 `scores/` 和 `cost.json` 画 `media/models_compare.zh-CN.svg`（v2、v3 总分，文件大小、参数量、计算量）和 `media/models_pareto.zh-CN.svg`（v3 准确率对计算量的帕累托前沿） |
| `zhiyi.py` | 知意 int8 ONNX：选词序列（整页）和组句序列、拼音槽对齐、组句的拼音约束 |
| `pinyingpt.py` | PinyinGPT-Concat / GPT2-zh-21k：受约束的候选打分和 beam search |
| `metasequoia.py` | 水杉 `.ort` 的逐键打分 |
| `youmo.py` | 幽默句子模型（chinese-ime-lm 格式）的 PyTorch 实现和打分 |
| `chaoinput.py` | 吵吵 CLMG v2（`lmt.bin`）和词典先验 |
| `qwen.py` | Qwen 的 log P(上文 + 候选) − log P(上文) |
| `scores/` | 每个模型在每套题上的分数，`{"id", "scores"}`，不比的候选为 null；含 `-ctx100` 和 `pinyingpt-concat-typed` 两种附加测试 |
| `results.json` | 各组的行数和准确率；`compose` 为组句的 P@1 / P@5 |
| `cost.json` | 各模型的文件大小、参数量和每行乘加次数 |
