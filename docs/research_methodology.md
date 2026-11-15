# Research methodology

## Task

Next-basket prediction. Given a customer's earlier baskets, rank every product in the catalogue; the target is the set of
products in their next basket. Supervised learning with a weighted binary cross-entropy over the catalogue.

## Data and cleaning

UCI Online Retail II, both sheets, with the file's column names mapped to the blueprint's canonical names. Every line is
labelled with an event type and none is deleted; lines that do not belong in the model are counted in
[`data_quality_report.md`](data_quality_report.md). Lines without a customer id are never merged into a fictitious customer.
Invoices present in both sheets are labelled `duplicate_overlap` and ignored.

## Splits and examples

- **By date, never random.** Train 1 Dec 2009 to 31 Dec 2010, validation 1 Jan to 30 Jun 2011, test from 1 Jul 2011. The
  boundaries come from the blueprint; the data ends on 9 Dec 2011, so the test window is about five months.
- **Examples.** Each basket with at least one earlier basket from the same customer. History = the customer's baskets with a
  strictly earlier timestamp (ties excluded), truncated to the last 50. The split of an example is the split of its target.
- **Catalogue.** The products seen in training baskets. Target items outside it are dropped from the metrics and counted;
  examples left with no target are skipped. Coverage is 88.8% of validation and 70.2% of test target items.
- **Leakage controls.** Normalisers use training data only; graph snapshots use baskets before the start of the example's
  month; `verify_temporal_integrity` runs over every example; the test file `test_final_eval.py` checks that model
  selection ignores test scores.

## Evaluation

Recall@5/10/20, Precision@10, NDCG@10 and MRR, ranked over the whole catalogue. Two candidate policies: `full` (repeat
purchases allowed) and `exclude_history` (the customer's history items are removed from the candidates and the targets, so
only discovery counts). Stochastic models run three seeds (42, 43, 44) and are reported as mean ± standard deviation.

## Selection and tuning protocol

1. Hyperparameters for GRU, matrix factorisation and BasketMind were searched with Optuna (TPE), scoring validation
   Recall@10 only. The BasketMind search used median pruning and capped trials at 20 epochs.
2. Early stopping, learning-rate scheduling and the best checkpoint use validation Recall@10.
3. The final BasketMind variant was chosen by mean validation Recall@10 over three seeds, a rule fixed before looking at test.
4. The test split is evaluated once per run from its best-validation checkpoint; `evaluate_test.py` refuses to repeat it.

## Deviations from the blueprint and known limits

- A weighted GraphSAGE layer replaces PyG's `SAGEConv`, because the stock layer ignores edge weights.
- The GRU baseline loops over `GRUCell` because `nn.GRU`'s backward pass fails on ROCm for Windows.
- The cancellation/reversal head is not trained: cancellation labels were not validated.
- The quantity head is trained but its metrics (MAE, RMSE, WAPE) are not evaluated.
- Purchase recency is measured to the target basket's timestamp (predictions are assumed to be made when the customer
  arrives). `recency_reference: none` in `configs/features.yaml` gives a time-blind variant, which is untested.
- Not yet done: confidence intervals, paired comparisons, cold-start and short-history breakdowns.
- GPU runs with the same seed start from identical weights but are not bit-identical (summation order varies, about 1e-6).
