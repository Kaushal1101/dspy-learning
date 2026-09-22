# Financial Sentiment Optimization Results

## Setup

| | |
|---|---|
| **Dataset** | `zeroshot/twitter-financial-news-sentiment` (9,543 rows total) |
| **Sample used** | 500 rows (300 train / 100 val / 100 test) |
| **Inference model** | `gpt-4.1-nano` |
| **Optimizer model** | `gpt-4o-mini` (teacher for Bootstrap, reflection LM for GEPA) |
| **Task** | 3-class sentiment classification: bearish / bullish / neutral |

**Test label distribution:** neutral: 64, bullish: 22, bearish: 14

---

## Baseline Prompt

```
Classify the sentiment of the financial news headline.
Reply with one of: bearish, bullish, neutral.
```

Zero-shot — no examples, no elaboration.

---

## Overall Results

| Variant | Accuracy | Latency | Cost/call | Cost/1k calls | Delta |
|---|---|---|---|---|---|
| Baseline | 71.0% | 705 ms | $0.000024 | $0.024 | — |
| GEPA | 77.0% | 914 ms | $0.000064 | $0.064 | +6.0pp |
| **Bootstrap** | **85.0%** | **706 ms** | **$0.000036** | **$0.036** | **+14.0pp** |

---

## Per-class Breakdown

### GEPA vs Baseline

| Class | N | Baseline | GEPA | Delta |
|---|---|---|---|---|
| bearish | 14 | 11/14 (78.6%) | 10/14 (71.4%) | -7.1pp |
| bullish | 22 | 19/22 (86.4%) | 16/22 (72.7%) | -13.6pp |
| neutral | 64 | 41/64 (64.1%) | 51/64 (79.7%) | +15.6pp |
| **TOTAL** | 100 | 71/100 (71.0%) | 77/100 (77.0%) | **+6.0pp** |

### Bootstrap vs Baseline

| Class | N | Baseline | Bootstrap | Delta |
|---|---|---|---|---|
| bearish | 14 | 10/14 (71.4%) | 13/14 (92.9%) | +21.4pp |
| bullish | 22 | 19/22 (86.4%) | 17/22 (77.3%) | -9.1pp |
| neutral | 64 | 42/64 (65.6%) | 55/64 (85.9%) | +20.3pp |
| **TOTAL** | 100 | 71/100 (71.0%) | 85/100 (85.0%) | **+14.0pp** |

---

## Optimized Prompts

### GEPA — Rewritten instruction (excerpt)

GEPA rewrote the instruction text across 31 iterations. Final version:

```
Classify the sentiment of the provided financial news headline based on
its implications for stock performance or market outlook. Determine
whether the sentiment is bearish, bullish, or neutral.

- Bullish: positive outlook — growth, strong performance, favorable forecasts.
  Keywords: "strong", "growth", "buy", "profit", "positive", "record".
- Bearish: negative outlook — warnings, disappointing results, negative forecasts.
  Keywords: "risk", "loss", "decline", "down", "recession", "warning".
- Neutral: no clear positive or negative sentiment — informational announcements,
  upcoming reports, factual statements without market bias.

Examples:
- "U.S. Stocks Waver as Viral Outbreak Claims More Lives" → bearish
- "Company X reports record profits" → bullish
- "Company Z's Q2 earnings report released" → neutral
```

> Note: GEPA's verbose rewrite increased prompt length significantly,
> raising cost per call from $0.000024 → $0.000064 (+2.7×).

### Bootstrap — Original instruction + 2 selected demos

Bootstrap kept the original instruction unchanged and injected 2 examples:

```
Classify the sentiment of the financial news headline.
Reply with one of: bearish, bullish, neutral.

[Demo 1]
Sentence: Bids are building up on $REKR could pop over 3.04 love this chart.
Sentiment: bullish

[Demo 2]
Sentence: Wall Street sees some positive coronavirus news, with signs the
curve may be bending in parts of the world.
Sentiment: bullish

[Live input]
Sentence: {your sentence here}
```

> Note: Bootstrap's minimal additions kept latency virtually identical
> (705ms → 706ms) and cost increase modest ($0.000024 → $0.000036, +50%).

---

## Key Observations

- **Bootstrap outperforms GEPA** (+14pp vs +6pp) on this task.
- **GEPA over-indexed on neutral** (+15.6pp on neutral, but regressed on bearish and bullish). With neutral making up 64% of the test set, this inflated overall accuracy while hurting minority classes.
- **Bootstrap's cost and latency are nearly unchanged** — few-shot demos add tokens but far fewer than GEPA's verbose instruction rewrite.
- **GEPA is better suited for underspecified prompts** where the instruction text itself needs improving. Here the original prompt was already clear enough that rewriting it didn't help.
- **The 500-row sample is small** — val/test sets of 100 examples mean 1 example = 1pp, making scores noisy. Re-running with the full dataset (5,000+ rows) would produce more reliable results.

---

## Saved Files

| File | Contents |
|---|---|
| `/tmp/financial_gepa_optimized.json` | GEPA optimized DSPy program |
| `/tmp/financial_gepa_stats.json` | GEPA baseline + optimized stats |
| `/tmp/financial_bootstrap_optimized.json` | Bootstrap optimized DSPy program |
| `/tmp/financial_bootstrap_stats.json` | Bootstrap baseline + optimized stats |
