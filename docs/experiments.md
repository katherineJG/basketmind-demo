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

## Open questions

Why the Transformer branch does not help; whether a repeat-purchase path closes the gap (item-item CF also leads on new
items, so probably not entirely); how the scores break down for cold-start and short-history customers.
