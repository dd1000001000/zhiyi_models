# qj-score：用青简自己的代码给候选打分

`qj-score` 调青简源码里的 `qingjian-neural`（含章 `CharScorer`）和 `qingjian-lm`（二元 `BigramModel`），给每行的候选打分，供 `../qingjian.py` 和 `../../ngram/run.py` 调用。它不是青简的一部分，只在对比时编译。

测试用的青简源码：[qingjian-team/qingjian](https://github.com/qingjian-team/qingjian) `main` 分支 `97dde30`（2026-11-03），Rust 1.96.0（青简的 `rust-toolchain.toml`）。

```bash
git clone https://github.com/qingjian-team/qingjian && cd qingjian && git checkout 97dde30
cp -r <本目录>/qj-score tools/
# 根 Cargo.toml 的 workspace members 里加上 "tools/qj-score"
cargo build --release -p qj-score
```

```bash
qj-score p2c  hanzhang-tongbian-small.qjm < rows.jsonl > scores.jsonl
qj-score char hanzhang-zhiwei-small.qjm   < rows.jsonl > scores.jsonl
qj-score lm   lm.qj                       < rows.jsonl > scores.jsonl
```

每行输入 `{"id", "context", "code", "cands": [...]}`，输出 `{"id", "scores": [...]}`；各模式的打分方式见 `src/main.rs` 的文件头。青简代码按 GPL-3.0-or-later 许可，本目录的代码同样按 GPL-3.0-or-later 发布。
