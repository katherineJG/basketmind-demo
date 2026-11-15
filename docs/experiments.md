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

## Open questions

Why the Transformer branch does not help (and why the GRU overtakes item-item CF only with long histories); whether a
repeat-purchase path closes the gap (item-item CF also leads on new items, so probably not entirely); whether the narrow
coverage of BasketMind's recommendations can be widened without losing accuracy.
