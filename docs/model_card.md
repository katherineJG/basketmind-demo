# Model card: BasketMind

## Model

BasketMind is a hybrid neural network for next-basket recommendation: a causal Transformer over purchase history, a
weighted GraphSAGE encoder over a PPMI product co-purchase graph, and a customer-context network, fused into one 128-d customer
vector that scores every product. Variants switch branches on or off ([architecture](architecture.md)).

- **Selected variant:** graph + customer branches, chosen on validation. About 0.79M parameters.
- **Best deep learning model in this project:** the tuned GRU baseline, which scored slightly higher on validation and test.
- **Strongest method overall:** personal frequency (rank a customer's own past products by recency-weighted purchase count),
  which learns nothing. It was added after the course report, which names item-item collaborative filtering here; item-item CF
  is the strongest of the baselines the report compared. See [repeat purchases](repeat_purchases.md).
- **What the API serves:** the unchanged BasketMind (graph + customer) model. Adding a personal-frequency score to its logits
  (analysis only, not served) lifts test Recall@10 from 0.125 to 0.228, level with personal frequency alone.

## Intended use

Ranking a short list of products per customer from past baskets, as a research and teaching prototype. Possible downstream
uses are reminders, offers and cross-selling, but I have not measured any commercial effect.

**Out of scope:** decisions that materially affect individual customers, use on other retailers or periods without
re-evaluation, and any claim of production readiness.

## Data

UCI Online Retail II (CC BY 4.0): an online retailer, December 2009 to December 2011, 1,067,371 invoice lines. After cleaning:
36,594 baskets, 5,852 customers, 4,619 products. About 22.8% of lines have no customer id and are excluded from histories.
Time-based split: train to Dec 2010, validation Jan to Jun 2011, test Jul to 9 Dec 2011.

## Performance (test, repeat purchases allowed)

| Model | Recall@10 | NDCG@10 | MRR |
|---|---:|---:|---:|
| Personal frequency (added after the report) | 0.224 | 0.293 | 0.470 |
| Item-item CF | 0.145 | 0.188 | 0.326 |
| Tuned GRU | 0.131 ± 0.001 | 0.175 | 0.325 |
| BasketMind (graph + customer) | 0.125 ± 0.004 | 0.164 | 0.314 |

Mean ± standard deviation over three seeds. On new items only, test Recall@10 falls to 0.052 (GRU) and 0.049 (BasketMind),
against 0.061 for item-item CF; personal frequency scores 0.024 there, the same as popularity, because it only repeats the past.
Full tables: [`final_comparison.md`](final_comparison.md).

## Limitations

- One retailer, two years. The test period includes the pre-Christmas months.
- Only 70.2% of test target items exist in the training catalogue; new products cannot be recommended.
- Weak on short histories: with 1 to 10 earlier baskets item-item CF beats BasketMind by 3 to 6 points of Recall@10; BasketMind is
  level with CF only at 11 or more baskets. On cold-start customers it scores 0.115 against 0.173 for CF.
- Repeat purchases carry the score: 55% of target items are repeats and 90% or more of BasketMind's Recall@10 is earned on them
  ([repeat purchases](repeat_purchases.md)). It beats neither personal frequency nor, on new items, item-item CF.
- Narrow recommendations: only 15.2% of the catalogue is ever recommended (36.3% for the GRU, 43.3% for item-item CF).
- Three seeds. Customer-level 95% intervals and paired tests are in [`evaluation_rigour.md`](evaluation_rigour.md): the tuned GRU leads
  BasketMind on Recall@10 by 0.0051 (CI [0.0016, 0.0088]), though BasketMind's own seed-to-seed spread (0.0044) is similar in size.
- Tuning budgets were unequal across models.
- The quantity head (how many units, given a purchase) is evaluated in [`quantity_head.md`](quantity_head.md). Test MAE is 7.9 units
  against 9.7 for a per-product median, but 7.1 for simply repeating the customer's last quantity, which it does not beat.
- Scores are offline. Real-world effect would need an online test.

## Ethical considerations

The data set is public and pseudonymous (customer ids only), but it describes real purchasing behaviour. Personalised
recommendation can reinforce existing purchase patterns, and the model has not been checked for differences in quality
across customer groups or countries. Keep customer data local and do not link it to other identifiers.
