# Data dictionary

Source: UCI Online Retail II (CC BY 4.0), two sheets (2009-2010 and 2010-2011) of one UK online gift retailer.
The file's column names differ from the blueprint; `configs/data.yaml` maps them. Raw data is never modified.

## Raw columns (after renaming)

| Column | Type | Meaning |
|---|---|---|
| InvoiceNo | string | Invoice id. Prefix `C` = cancellation, `A` = bad-debt adjustment (source: `Invoice`) |
| StockCode | string | Product code, upper-cased. Products are 5 digits plus optional letters |
| Description | string | Product text; 4,382 nulls |
| Quantity | int | Units; negative for cancellations and adjustments |
| InvoiceDate | datetime | Line timestamp (minute resolution; some invoices span several minutes) |
| UnitPrice | float | GBP per unit (source: `Price`) |
| CustomerID | string (nullable) | Customer id, stored as a string; ~22.8% of rows are null and stay null (source: `Customer ID`) |
| Country | string | Customer country |
| source_sheet | string | Added by us: which sheet the row came from |

## Added labels

| Column | Meaning |
|---|---|
| event_type | One of `purchase`, `cancellation`, `adjustment`, `non_product`, `invalid_price`, `duplicate_overlap` (rules in `data/clean.py`, first match wins in that priority order) |
| has_customer | CustomerID is present |

## Processed datasets (`data/processed/`, not in Git)

| File | Contents |
|---|---|
| purchase_events.parquet | Rows labelled `purchase` (includes anonymous rows) |
| cancellation_events.parquet | Rows labelled `cancellation`, kept for the reversal task (M4) |
| other_events.parquet | adjustment, non_product, invalid_price, duplicate_overlap rows |
| baskets.parquet | One row per customer invoice: CustomerID, InvoiceNo, timestamp (earliest line), Country, items (sorted stock codes), quantities, n_items, total_quantity, total_value, split |
| customer_sequences.parquet | One row per customer: invoices, timestamps, splits in chronological order, n_baskets |
| next_basket_examples.parquet | One row per basket that has history: CustomerID, target_invoice, target_timestamp, split (of the target), history_invoices (last 50 strictly earlier baskets) |
| split_metadata.json, data_quality_report.json | Split boundaries/counts and every figure in `docs/data_quality_report.md` |

## Rules worth knowing

- Anonymous invoices (no CustomerID) are labelled `purchase` but excluded from baskets: they cannot be placed in a sequence and are never merged into a fictitious customer.
- Repeated stock codes within an invoice are summed. Exact duplicate lines are kept (they may be genuine repeat scans).
- History uses strictly earlier timestamps only; same-timestamp baskets are not each other's history.
- Splits are by basket time (provisional boundaries from the blueprint). Examples carry the split of their target; history may reach back into earlier splits because it is past data.
