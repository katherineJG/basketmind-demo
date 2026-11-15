# Model card: BasketMind

## Model

BasketMind is a hybrid neural network for next-basket recommendation: a causal Transformer over purchase history, a
weighted GraphSAGE encoder over a PPMI product co-purchase graph, and a customer-context network, fused into one 128-d customer
vector that scores every product. Variants switch branches on or off ([architecture](architecture.md)).

- **Selected variant:** graph + customer branches, chosen on validation. About 0.79M parameters.
- **Best deep learning model in this project:** the tuned GRU baseline, which scored slightly higher on validation and test.
- **Strongest method overall:** item-item collaborative filtering, which is not a deep learning model.

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
| Item-item CF | 0.145 | 0.188 | 0.326 |
| Tuned GRU | 0.131 ± 0.001 | 0.175 | 0.325 |
| BasketMind (graph + customer) | 0.125 ± 0.004 | 0.164 | 0.314 |

Mean ± standard deviation over three seeds. On new items only, test Recall@10 falls to 0.052 (GRU) and 0.049 (BasketMind).
Full tables: [`final_comparison.md`](final_comparison.md).

## Limitations

- One retailer, two years. The test period includes the pre-Christmas months.
- Only 70.2% of test target items exist in the training catalogue; new products cannot be recommended.
- Cold-start and short-history customers are not yet analysed (35% of test customers have no training baskets).
- Three seeds, no confidence intervals or paired tests yet; close comparisons are inconclusive.
- Tuning budgets were unequal across models, and the quantity head is trained but not evaluated.
- Scores are offline. Real-world effect would need an online test.

## Ethical considerations

The data set is public and pseudonymous (customer ids only), but it describes real purchasing behaviour. Personalised
recommendation can reinforce existing purchase patterns, and the model has not been checked for differences in quality
across customer groups or countries. Keep customer data local and do not link it to other identifiers.
