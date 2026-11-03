//! Score given candidates with Qingjian's own models, for the neutral-set comparison (not part of Qingjian).
//!
//!   qj-score p2c  <hanzhang-tongbian-small.qjm>  < rows.jsonl > scores.jsonl
//!   qj-score char <hanzhang-zhiwei-small.qjm>    < rows.jsonl > scores.jsonl
//!   qj-score lm   <lm.qj>                        < rows.jsonl > scores.jsonl
//!
//! A row is {"id", "context", "code", "cands": [...]}; the output {"id", "scores": [...]} has a score per candidate,
//! all null where Qingjian would not score the row (a candidate with a character outside the model's vocabulary:
//! its SentenceScorer then keeps the static order; P2C input longer than the model's context).
//! - p2c:  CharScorer::score_p2c(code, cands), log P(candidate | keys typed), as its P2cScorer.
//! - char: CharScorer::score(context, cands), log P(candidate | text before the caret), as its CharScorer.
//! - lm:   ln P(candidate | context) from its BigramModel: best word split (Viterbi over the model's words, at most
//!         8 characters, state = previous word) of the context's last 12 characters + candidate, minus that of the
//!         context; a character the model lacks costs ln(1e-9) (Core would fall back to its dictionary).

use std::cell::Cell;
use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::Path;

use qingjian_core::sentence::LanguageModel;
use qingjian_lm::BigramModel;
use qingjian_neural::CharScorer;
use serde_json::{Value, json};

const CTX: usize = 12;
const MAX_WORD: usize = 8;
/// ln(1e-9): a character the bigram model lacks (Core would fall back to its dictionary's frequency).
const UNKNOWN: f64 = -20.723_265_836_946_41;

thread_local! {
    /// Bigram lookups made (printed to stderr at the end: the comparison's compute measure).
    static LOOKUPS: Cell<u64> = const { Cell::new(0) };
}

fn best(lm: &BigramModel, unknown: f64, chars: &[char], start: usize, init: &HashMap<Option<String>, f64>)
        -> HashMap<Option<String>, f64> {
    let n = chars.len();
    let mut at: Vec<HashMap<Option<String>, f64>> = vec![HashMap::new(); n + 1];
    at[start] = init.clone();
    for i in start..n {
        if at[i].is_empty() {
            continue;
        }
        let states: Vec<(Option<String>, f64)> = at[i].iter().map(|(k, v)| (k.clone(), *v)).collect();
        for j in i + 1..=(i + MAX_WORD).min(n) {
            let word: String = chars[i..j].iter().collect();
            for (prev, score) in &states {
                LOOKUPS.with(|n| n.set(n.get() + 1));
                let lp = match lm.log_prob(prev.as_deref(), &word) {
                    Some(lp) => lp,
                    None if j == i + 1 => unknown,
                    None => continue,
                };
                let t = score + lp;
                let slot = at[j].entry(Some(word.clone())).or_insert(f64::NEG_INFINITY);
                if t > *slot {
                    *slot = t;
                }
            }
        }
    }
    std::mem::take(&mut at[n])
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (mode, path) = (args[1].as_str(), Path::new(&args[2]));
    let scorer = if mode == "lm" { None } else { Some(CharScorer::load(path).expect("model")) };
    let lm = if mode == "lm" { Some(BigramModel::from_path(path).expect("lm")) } else { None };
    if let Some(m) = &lm {
        eprintln!("entries {} {}", m.word_count(), m.bigram_count());
    }
    let unknown = UNKNOWN;
    let stdin = std::io::stdin();
    let mut out = std::io::BufWriter::new(std::io::stdout());
    for line in stdin.lock().lines() {
        let row: Value = serde_json::from_str(&line.expect("line")).expect("json");
        let cands: Vec<&str> = row["cands"].as_array().unwrap().iter().map(|c| c.as_str().unwrap()).collect();
        let context = row["context"].as_str().unwrap_or("");
        let code = row["code"].as_str().unwrap_or("");
        let scores: Vec<Option<f64>> = match mode {
            "p2c" | "char" => {
                let s = scorer.as_ref().unwrap();
                if !cands.iter().all(|c| s.vocab().knows(c)) {
                    vec![None; cands.len()]
                } else {
                    let r = if mode == "p2c" { s.score_p2c(code, &cands) } else { s.score(context, &cands) };
                    match r {
                        Ok(v) => v.into_iter().map(Some).collect(),
                        Err(_) => vec![None; cands.len()],
                    }
                }
            }
            _ => {
                let m = lm.as_ref().unwrap();
                let ctx: Vec<char> = context.chars().collect();
                let ctx = &ctx[ctx.len().saturating_sub(CTX)..];
                let mut init = HashMap::new();
                init.insert(None, 0.0);
                let s0 = if ctx.is_empty() { init.clone() } else { best(m, unknown, ctx, 0, &init) };
                let base = s0.values().cloned().fold(f64::NEG_INFINITY, f64::max);
                cands.iter().map(|c| {
                    let mut all: Vec<char> = ctx.to_vec();
                    all.extend(c.chars());
                    let end = best(m, unknown, &all, ctx.len(), &s0);
                    let v = end.values().cloned().fold(f64::NEG_INFINITY, f64::max);
                    if v.is_finite() { Some(v - base) } else { None }
                }).collect()
            }
        };
        writeln!(out, "{}", json!({"id": row["id"], "scores": scores})).unwrap();
    }
    eprintln!("lookups {}", LOOKUPS.with(Cell::get));
}
