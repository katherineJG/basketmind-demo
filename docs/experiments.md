# Experiments

A short log of what I ran and what each result changed. Validation Recall@10 (policy `full`) unless stated. The generated
reports hold the full tables: [`baseline_report.md`](baseline_report.md), [`gru_tuning.md`](gru_tuning.md),
[`matrix_factorisation_tuning.md`](matrix_factorisation_tuning.md), [`neural_tuning.md`](neural_tuning.md) and
[`final_comparison.md`](final_comparison.md).

## 1. Baselines

Popularity, item-item CF, matrix factorisation and a GRU on the same catalogue, examples and metrics. Item-item CF led
(validation 0.138). The untuned GRU scored 0.055, no better than popularity (0.053), so I tuned it.

## 2. Tuning the baselines (validation only)

- GRU: 30 Optuna trials, 0.055 to 0.134 (a roughly 9x larger learning rate and heavier dropout mattered most).
- Matrix factorisation: 40 trials, 0.105 to 0.125 (larger embeddings and many more epochs).

## 3. BasketMind, first runs

- Default settings (learning rate 0.0003): validation 0.056, test 0.047, equal to popularity.
- History-swap diagnostic (replace each customer's history with another's): the model lost only 7% of its score, so it was
  largely ignoring history. With the learning rate raised to 0.003 (validation 0.091) it lost 53%, which ruled out a
  wiring bug and pointed at training. Numbers are in `reports/data/history_swap.json`.
- Optuna, 24 trials (15 finished, 9 pruned): best 0.101. The learning rate stopped mattering inside the range searched;
  dropout hurt (rank correlation -0.68).

## 4. Ablations (best settings, 50-epoch budget, validation)

| Variant | Seeds | Val. Recall@10 |
|---|---:|---:|
| Graph + customer | 3 | 0.117 ± 0.002 |
| Full hybrid | 3 | 0.102 ± 0.002 |
| Transformer + customer | 1 | 0.091 |
| Transformer only | 1 | 0.080 |

Graph + customer beat the full hybrid on every seed. The graph branch helps; the Transformer branch does not.

## 5. Negative result: untied output embeddings

A separate output item table (plus the graph vectors) scored lower than sharing one table: graph + customer 0.094 against
0.115, full hybrid 0.093 against 0.102 (single seed). The option stays in the code, off by default.

## 6. Final evaluation

Graph + customer was selected on validation. Test, once per run: Recall@10 0.125 ± 0.004, against item-item CF 0.145 and the
tuned GRU 0.131. See [`final_comparison.md`](final_comparison.md).

## 7. Evaluation rigour (analysis of the finished models)

Per-example test results for every model, with customer-level bootstrap intervals, paired comparisons and slices. Full
tables: [`evaluation_rigour.md`](evaluation_rigour.md). The baselines were refitted to get per-example outputs and reproduced the
reported numbers to four decimals; the BasketMind checkpoints are the same ones scored before.

- **Gaps that hold up.** Graph + customer beats the full hybrid and matrix factorisation (about +0.017 Recall@10, interval
  excludes zero). The tuned GRU beats BasketMind on Recall@10 (-0.0051, 95% CI [-0.0088, -0.0016]), NDCG@10 and MRR; item-item CF
  beats the GRU on Recall@10 and NDCG@10 but ties it on MRR (-0.0014, CI [-0.011, +0.008]).
- **The overall ranking hides a crossover by history length.** With 1 to 10 earlier baskets item-item CF beats both neural
  models by 2 to 7 points of Recall@10. With 11 or more (46% of test examples) the GRU beats CF (+0.0146 [+0.0052, +0.0240]) and
  BasketMind is level with it (+0.0055, inconclusive).
- **Cold start (customers with no training basket, 21% of test examples).** CF leads (0.173 against 0.115 BasketMind and
  0.107 GRU); BasketMind beats the GRU here (+0.0075 [+0.0011, +0.0139]); matrix factorisation collapses to 0.034.
- **Drift.** Most models score lower in October and November than in July to September (item-item CF 0.125 and 0.137 against
  about 0.16); the full hybrid is the exception.
- **Narrow recommendations.** BasketMind (graph + customer) ever recommends only 15.2% of the catalogue, against 36.3% for the GRU
  and 43.3% for item-item CF.

## 8. Cancellation labels

A 30-day window label ("the customer cancels this product within 30 days") flags 1.83% of purchase lines and is 85%
attributable to a real preceding purchase; a 90-day label is too noisy (68%). 91% of cancellations link to an earlier purchase,
and the 9% that do not are concentrated in the first three months of the data. Verdict: usable as a weak label with split-boundary
cutting, not yet built. See [`cancellation_labels.md`](cancellation_labels.md).

## 9. Repeat purchases and a missing baseline

Full tables: [`repeat_purchases.md`](repeat_purchases.md). Analysis of the saved test results (nothing retrained), plus two new
baselines tuned on validation and evaluated on test once each.

- **Repeats dominate.** 55.3% of in-catalogue target items are repeats of something the customer bought before, and 91% of test
  examples contain at least one. (An earlier raw count gave 48%: it included target items outside the catalogue, which are mostly
  new.) Between 90% and 96% of every personalised model's Recall@10 is earned on repeat targets.
- **The baseline I left out is the strongest.** Ranking a customer's own past items by recency-weighted purchase count
  (decay 0.9 per basket, chosen on validation) scores test Recall@10 0.224 [0.210, 0.236], against 0.145 for item-item CF and 0.125
  for BasketMind. Blending in item-item CF (weight 0.25, chosen on validation) gives 0.225 and adds nothing on repeat targets
  (+0.0002 [-0.0015, +0.0018]); it only adds a little on new items. Both chosen settings are inside their grids, not at an edge.
- **Sanity checks.** Under the new-items-only policy the purely personal model scores 0.024, which is just its popularity
  tie-break, and the blend scores 0.061, the same as item-item CF's own new-item score. Validation (0.210) and test (0.224) agree.
- **Where BasketMind loses.** It trails item-item CF by 0.024 on repeat targets and leads it by 0.004 on new targets (both
  intervals exclude zero), and trails personal frequency by 0.110 on repeat targets. It spends only 48% of its top-10 slots on the
  customer's history, against 98% for personal frequency, and those history slots hit about as often per slot (0.220 against
  0.221). Its other slots hit 0.031 of the time. The next bullet tests whether spending those slots on history items helps.
- **A repeat score closes the gap, and the network adds almost nothing on top.** I added `lam * count / max(count)` (the
  personal-frequency term, recency decay 0.9) to the unchanged BasketMind logits. `lam` and the decay were chosen per seed on
  validation (decay 0.9 for all three seeds; weight 16, 32 and 32; validation was flat from 16 to 64, so the grid did not limit
  it), with no retraining, and the three seeds were then evaluated on test once. Test Recall@10 goes from 0.125 to 0.228
  (seeds 0.227, 0.229, 0.227), against 0.224 for personal frequency alone. The gain over personal frequency is small: +0.0034
  [+0.0020, +0.0048] on repeat targets and +0.0003 [-0.0000, +0.0006] on new targets. Against the personal frequency + CF blend
  it is +0.0032 on repeats and -0.0007 on new targets. So the network does order a customer's own items slightly better than
  counting does, but the whole improvement over the old BasketMind (+0.1135 on repeat targets) comes from the count term.
- **What the repeat score costs.** 98% of the top-10 slots now hold history items, and the part of Recall@10 earned on new
  targets falls from 0.012 to 0.001 (-0.0113 [-0.0123, -0.0104]). Under the new-items-only policy the model is unchanged (0.049
  against 0.061 for item-item CF), so it still loses on discovery. It is a better answer to "what will this customer buy
  again?", not to "what should we show them that is new?".
- **Consequence for earlier claims.** The documented baseline set did not include this baseline, so "item-item CF is the strongest
  baseline" (README, final comparison, course report) was true only of the set I had run. The earlier per-slice finding stands as a
  statement about CF versus the neural models, not about the best achievable score.

## Open questions

Why the Transformer branch does not help (and why the GRU overtakes item-item CF only with long histories); whether a
repeat score *learned* inside the network (rather than added afterwards) would beat the +0.003 the logits add over counting,
which the evidence so far does not support; whether anything beats item-item CF on genuinely new items, where BasketMind
(0.049) and the GRU (0.052) both trail it (0.061); whether the narrow coverage of BasketMind's recommendations can be widened
without losing accuracy. The repeat score is analysis only: the API still serves the unchanged BasketMind model.
