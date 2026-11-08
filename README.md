# Instamart Customer Intelligence & Support Copilot

## Overview
A reproducible, offline capstone project using fictional synthetic Instamart-style data. It will combine a trusted SQLite data foundation, customer analytics, guarded support-ticket triage, policy retrieval and a refund agent. All monetary values are Indian Rupees (INR).

## Reproduce
1. `python -m venv .venv`
2. Activate the environment (`.venv\\Scripts\\activate` on Windows or `source .venv/bin/activate` on macOS/Linux).
3. `python -m pip install --upgrade pip`
4. `pip install -r requirements.txt`
5. `python generate_data.py`
6. `python src/part_a_data_cleaning.py`
7. `python src/part_a_sql_analysis.py`
8. `python src/part_b_customer_analysis.py`
9. `python src/part_c_ticket_triage.py`
10. `python src/part_d_rag_agent.py`

## Part A — Data Foundation: Clean, Store, Query

### Raw-data audit
The audit script prints each raw table's shape, dtypes, and per-column blank/missing counts and percentages. The expected raw checkpoints are: 13,084 order rows; 119 exact duplicate rows; 656 pending rows; 12,309 distinct order IDs; 120 non-numeric amounts; 106 negative delivery durations; 2,541 blank delivery durations; 61 blank ages; 6 ages outside 10–100; 24 city spellings; 10 nonblank gender spellings; and 30 blank genders.

### Cleaning rules
- Customers: trim city names and apply title case; normalize gender to `male`, `female`, or `unknown`; set ages outside 10–100 to missing and impute missing ages with the city median using `groupby().transform()`; parse signup dates as datetimes.
- Orders: parse timestamps; remove exact duplicate rows; keep the latest timestamp for each order ID; parse amounts and reconstruct missing amounts from catalogue price × quantity × (1 − discount percentage); set negative delivery times to missing; impute missing completed-order delivery times using the completed-order median for the product category; keep cancelled delivery times missing.
- Products: retain the supplied catalogue and numeric INR prices.

### Cleaning changes and checkpoints
- Customer city counts: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184. Gender counts: female 610, male 560, unknown 30. The 61 blank ages and 6 out-of-range ages are repaired using city-median imputation; final missing ages: 0.
- Orders: 119 exact duplicate rows removed, then 656 older duplicate order records removed; final 12,309 unique orders. The 120 malformed amounts become numeric missing values; after latest-record selection, 112 missing final amounts are reconstructed from catalogue price, quantity and discount. All 106 negative delivery times are set missing; missing completed-order delivery times are imputed using product-category medians. Final statuses: 11,110 completed and 1,199 cancelled; missing amounts: 0; missing completed delivery times: 0; missing cancelled delivery times: 1,199.
- Completed-order revenue is INR 2,998,668.80. The validation script checks 14 conditions and prints exactly `ALL CHECKPOINTS PASSED` when they all hold.

### SQL analysis
Seven queries are stored in `sql/queries.sql` and run through `pd.read_sql_query` by `src/part_a_sql_analysis.py`.

| Query | Result |
|---|---:|
| Q1 — WHERE then HAVING | 30 customers |
| Q2 — city × category revenue | 30 rows; total INR 2,998,668.80; largest cell Bengaluru / Electronics = INR 233,909.45 |
| Q3 — products above average catalogue price | 11 products |
| Q4 — customers with no completed order | 25 customers |
| Q5 — spend above average | 550 customers; average spend INR 2,552.06 |
| Q6 — city spend ranking | Top customers: Bengaluru C0082 (INR 12,856.20), Chennai C0930 (INR 10,151.40), Delhi C0291 (INR 13,742.60), Hyderabad C0743 (INR 12,730.05), Mumbai C0693 (INR 14,766.55), Pune C0426 (INR 10,377.10); 8 rows have RANK different from ROW_NUMBER |
| Q7 — daily running revenue | 183 days; final cumulative revenue INR 2,998,668.80 |

Tied spend values share a RANK position, while ROW_NUMBER assigns a distinct sequential number to every row, so ties make the two rankings differ.

### pandas cross-check and exploratory analysis
The pandas pivot table reproduces the Q2 grand total of INR 2,998,668.80. City aggregates (sum / mean / count) are: Bengaluru INR 523,643.40 / INR 276.47 / 1,894; Chennai INR 463,629.15 / INR 272.72 / 1,700; Delhi INR 502,829.80 / INR 243.62 / 2,064; Hyderabad INR 528,589.90 / INR 284.95 / 1,855; Mumbai INR 479,058.50 / INR 297.00 / 1,613; Pune INR 500,918.05 / INR 252.48 / 1,984.

Completed-order amount statistics: mean INR 269.91; median INR 135.00; skewness 4.61; Q1 INR 64.80; Q3 INR 297.00; IQR INR 232.20; lower fence −INR 283.50; upper fence INR 645.30; outliers below = 0 and above = 897. The mean substantially exceeds the median, indicating a strongly right-skewed order-value distribution.

The data describes completed synthetic grocery and convenience orders. Most order amounts are modest, but a smaller number of high-value orders pull the mean above the median. The 897 upper-fence outliers reflect unusually large order values, especially where higher-priced products or quantities are involved. The strongest absolute correlation is between order amount and discount percentage (r = −0.095); it is weak, and correlation of this size does not establish causation.

## Part B — Customer Segments and Lapse Prediction

Run `python src/part_b_customer_analysis.py`. It writes `outputs/customer_segments.csv`, `outputs/lapse_model.joblib`, and `outputs/retention_list.csv`.

### RFM and segmentation
RFM is calculated for customers with at least one completed order, as of `2025-10-01 00:00:00`. The 1,175-customer RFM table has median recency 19 days (minimum 0, maximum 180), mean frequency 9.46 (maximum 33), and mean monetary value INR 2,552.06 (maximum INR 14,766.55).

K-means uses StandardScaler on the three RFM columns, `n_init=10`, and `random_state=42`:

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1,820.761 | 0.465 |
| 3 | 1,268.235 | 0.445 |
| 4 | 743.071 | 0.526 |
| 5 | 617.367 | 0.480 |
| 6 | 503.171 | 0.482 |
| 7 | 408.460 | 0.416 |
| 8 | 366.503 | 0.412 |

K=4 is selected because it is a clear elbow, has the strongest silhouette among the tested values, and gives four actionable customer groups. Inertia cannot increase as K increases because adding clusters can only keep or reduce within-cluster squared distances; therefore K=8 will have lower inertia than K=2, but that does not show that eight groups are stable or useful to the business.

| Segment ID | Business name | Customers | Mean recency (days) | Mean frequency | Mean monetary (INR) | Suggested action |
|---:|---|---:|---:|---:|---:|---|
| 1 | Premium Loyalists | 138 | 24.12 | 6.77 | 7,129.70 | Protect with priority support and relevant loyalty benefits. **Premium** |
| 2 | Frequent Regulars | 419 | 8.77 | 19.40 | 3,410.02 | Encourage repeat orders with personalized bundles. |
| 0 | Occasional Shoppers | 436 | 28.86 | 3.77 | 1,057.62 | Use low-cost reminders and category recommendations. |
| 3 | Dormant Customers | 182 | 109.04 | 2.20 | 685.99 | Test a win-back offer and monitor response. |

Cluster numbers are arbitrary labels, not a ranking or an ordinal scale. The Premium flag is assigned only to the segment with the highest mean monetary value. PCA on the scaled RFM matrix explains 0.6343 and 0.2180 of variance in its first two components (total 0.8523).

### Lapse definition, features and leakage controls
The cohort contains every customer with at least one completed order before `2025-09-01`; lapsed means no completed order from 1–30 September 2025. The cohort has 1,156 customers, of whom 443 (38.32%) lapsed. The stratified 80/20 split contains 924 training and 232 test rows. Features are built only from orders strictly before the cutoff and include recency, frequency, monetary, mean discount, cancellation rate, mean delivery time, favourite product category, city, age, gender and tenure. Segment labels are not model features. Feature missing-value count is zero.

The frequency scaler's training-split mean is 8.1147, versus 8.0285 for the full cohort. They differ because the scaler is fitted only on training rows, whereas the full-cohort mean also includes held-out test rows. Every encoder and scaler is fitted inside the training-only pipeline; no encoder, scaler or resampler saw test rows. No order dated on or after 2025-09-01 entered the model features. The untouched test set was used only after model tuning for final evaluation.

### Model tuning and evaluation
The all-negative baseline has test accuracy 61.64% and lapsed-class F1 = 0. Logistic regression was tuned over C = {0.01, 0.1, 1, 10, 100}; its best C was 100 and best cross-validated F1 was 0.6506. Random forest (`n_estimators=100`, `random_state=42`) was tuned over max_depth = {3, 5, 8} and min_samples_leaf = {5, 10, 20}; its best settings were depth 8 and leaf size 10, with cross-validated F1 0.6866.

| Model | Confusion matrix (TN, FP / FN, TP) | Lapsed precision | Lapsed recall | Lapsed F1 |
|---|---|---:|---:|---:|
| Tuned Logistic Regression | [[111, 32], [31, 58]] | 0.6444 | 0.6517 | **0.6480** |
| Tuned Random Forest | [[110, 33], [32, 57]] | 0.6333 | 0.6404 | 0.6369 |

Logistic regression is selected because it has the higher measured test F1 (0.6480 versus 0.6369) and offers more transparent coefficient-based interpretation. Missing a customer who is about to lapse can mean losing future orders, while wrongly targeting a loyal customer wastes a retention incentive and can reduce margin; the team should balance those costs when setting a campaign threshold. The saved model was reloaded with joblib and reproduced the original test predictions exactly.

### Class-balance experiment and interpretation
Random oversampling duplicated only lapsed training rows (with replacement, `random_state=42`) until the training classes were balanced at 570 each. On the unchanged test set, logistic regression moved from precision 0.6444 / recall 0.6517 / F1 0.6480 to precision 0.5891 / recall 0.8539 / F1 0.6972. Recall moved most: the retention team catches more genuinely lapsing customers, at the cost of more false-positive offers to loyal customers.

The five largest random-forest importances were monetary (0.2804), frequency (0.2779), recency (0.1091), average delivery minutes (0.0597), and cancellation rate (0.0518). The three largest absolute logistic coefficients were frequency (negative, −1.6973), favourite category Electronics (negative, −1.1352), and favourite category Grocery (negative, −0.4493); within this fitted model, higher frequency and those category indicators are associated with lower predicted lapse odds, holding other encoded features fixed.

The Premium test-customer retention list contains the 10 highest predicted lapse probabilities; 3 of those 10 actually lapsed. **This retention list is illustrative and not a leakage-free evaluation:** the Premium flag comes from whole-window RFM, which includes the September period used to define the lapse label, even though the model's own features use only pre-cutoff data.

## Part C — Support Ticket Triage with Guardrails
To be completed.

## Part D — Policy RAG Assistant and Refund Agent

Run `python src/part_d_rag_agent.py`. This part uses the five supplied policy text files, the clean SQLite database from Part A and the offline `mock_llm.py`; it makes no network calls and requires no API key.

### Chunking, retrieval and grounded answers

Each policy document is split into 40-word chunks with a 10-word overlap (the next chunk starts 30 words after the previous start). The resulting 15 chunks are: `refund_policy.txt` 4, `delivery_policy.txt` 3, `escalation_policy.txt` 3, `payment_policy.txt` 3 and `privacy_policy.txt` 2. `TfidfVectorizer()` is fitted on those chunk texts; retrieval transforms a question with the same fitted vectorizer and ranks chunks using cosine similarity. Term frequency (TF) measures how often a term occurs in a chunk, while inverse document frequency (IDF) downweights terms that occur in many chunks. TF-IDF is lexical rather than semantic: two passages with the same meaning but different wording may not match well.

`retrieve(query, k=3)` returns source, score and text for the top chunks. `answer(question)` sends a `POLICY ASSISTANT` system instruction and a user message formatted as `CONTEXT:` / source-prefixed chunks / `QUESTION:`; the model is instructed to answer only from this context and otherwise use its fallback. For `refund policy late delivery`, all top three chunks come from `refund_policy.txt`.

| Chunking | Answerable expected document found | Answerable expected fact in answer | Unanswerable fallback |
|---|---:|---:|---:|
| 40 words, 10-word overlap | 12/12 | 12/12 | 2/2 |
| 40 words, zero overlap | 12/12 | 10/12 | 2/2 |

With zero overlap, the question “What delivery time makes a completed order eligible for a full refund?” loses the key phrase `more than 45 minutes`: the sentence is split at a chunk boundary, so the retrieved fragment says only that the customer “receives a full refund of the order amount.” The zero-overlap payment question about a failed-but-deducted payment also returns a truncated fragment. The evaluation details are saved in `outputs/policy_eval_overlap.csv` and `outputs/policy_eval_no_overlap.csv`.

### Refund tools and policy rules

`TOOLS_SPEC` documents the five callable tools and their parameter schemas; `TOOL_MAP` maps their names to Python functions:

- `get_order(order_id)` reads the order status, amount and delivery time from `data/instamart.db`.
- `search_policy(query)` returns the top three policy passages.
- `calculate_refund(order_amount, delivery_minutes)` returns the full order amount only when delivery took more than 45 minutes; otherwise it returns INR 0.
- `issue_refund(order_id, refund_inr)` appends a timestamped action to `outputs/refund_ledger.jsonl` for an automatically approved refund up to INR 1000.
- `escalate_to_human(order_id, refund_inr)` appends a timestamped escalation action for a refund above INR 1000.

Only completed orders are eligible. A completed order delivered in more than 45 minutes receives a full refund; delivery in 45 minutes or less receives INR 0. Refunds above INR 1000 are escalated and are not issued automatically. The supplied scenarios are treated as raised inside the policy's seven-day request window.

### ReAct scenario results

The agent begins with an `AGENT` system message and the scenario's user message. It follows tool calls by appending an assistant tool-call message and then a tool observation message. The seven supplied scenarios produced:

| Scenario | Status | Refund amount (INR) | Model calls | Tokens used |
|---|---|---:|---:|---:|
| S1 | REFUND | 128.25 | 5 | 952 |
| S2 | REFUND | 135.00 | 5 | 952 |
| S3 | ESCALATE | 1,999.00 | 5 | 939 |
| S4 | DENIED | 0.00 | 4 | 649 |
| S5 | DENIED | 0.00 | 3 | 370 |
| S6 | NOT_FOUND | 0.00 | 2 | 94 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,066 |

S1 requires five model calls, while the loop scenario S7 deliberately uses all six allowed calls and ends in `FALLBACK_MAX_ITERATIONS`. The scenario log is `outputs/agent_runs.jsonl`; the compact table is `outputs/agent_scenario_results.csv`. Issued and escalated actions are recorded in `outputs/refund_ledger.jsonl`.

### Safe stopping and resumability

The agent enforces three independent limits. It checks elapsed wall-clock time at the start of every iteration, stops before executing a requested tool if cumulative model tokens exceed the token limit, and returns a handoff summary after the maximum number of iterations. The S1 demonstration with `token_limit=200` ends as `FALLBACK_TOKEN_LIMIT` after 3 model calls (387 tokens); the `time_limit_seconds=0` demonstration ends as `FALLBACK_TIMEOUT` before any model call (0 calls, 0 tokens).

After each tool observation, a JSON checkpoint in `state/` stores the run ID, scenario ID, complete message history, token count, step number, tools already executed and call count. The S1 crash demonstration raises a simulated crash immediately after the second tool execution, after saving its checkpoint, and then resumes. The tools executed across the crash and resume, in order, are exactly `get_order`, `search_policy`, `calculate_refund`, `issue_refund`; none is repeated. The resumed run finishes as `REFUND` for INR 128.25.

### ReAct versus plan-and-execute

The measured ReAct flow used five model calls for S1, three for S5 and two for S6. A plan-and-execute design could ask for a complete tool plan in one model call and invoke the tools locally, using another model call only when an observation invalidates the plan; this can reduce model-call and prompt-token overhead on predictable cases. However, ReAct sees each observation before deciding the next action, so it can stop immediately for a missing order, a cancelled order or a zero-refund result rather than blindly following a stale plan. For refunds, I would keep ReAct because policy compliance and avoiding an incorrect payment outweigh the extra calls; the iteration, token and timeout limits bound that cost and provide a human handoff.

### Part D output files

- `outputs/policy_eval_overlap.csv` and `outputs/policy_eval_no_overlap.csv` — grounded-answer evaluation with and without overlap.
- `outputs/agent_scenario_results.csv` and `outputs/agent_runs.jsonl` — all seven scenario outcomes and step-by-step tool observations.
- `outputs/refund_ledger.jsonl` — simulated issued and escalated refund actions.
- `state/*.json` — resumable agent checkpoints, including the S1 crash-and-resume demonstration.

The complete Part D validation prints `PART D VALIDATION: ALL CHECKS PASSED`.

## Limitations and Next Steps
All records are synthetic and fictional. The Premium retention list is illustrative and not a leakage-free evaluation because its Premium flag is derived from whole-window RFM, including the September label window. Model performance is specific to this generated dataset; real deployment would require fresh temporal validation, monitoring, threshold selection and privacy review. Parts A–D are implemented against the supplied synthetic data and offline mock model. The project is a reproducible assessment demonstration, not a production payment system; real deployment would require authorization, audit, security, privacy and operational controls.

## Part C — Support Ticket Triage with Guardrails

Run `python src/part_c_ticket_triage.py`. The pipeline reads `data/raw/tickets.csv` with `keep_default_na=False`, processes rows in file order, and calls only the supplied offline `mock_llm.py`.

### Input guardrails and privacy
There are 164 tickets: 4 empty/whitespace-only tickets are marked `skipped_empty`, and 4 tickets whose estimated input length exceeds 400 tokens are marked `flagged_long` (the long inputs estimate to 734 tokens). Neither group is sent to the model. Before a prompt is assembled, the ticket text is redacted in the required order: email addresses → `[EMAIL]` (55 substitutions), 10-digit phone numbers → `[PHONE]` (91), and `ORD-######` identifiers → `[ORDER_ID]` (132), for 278 substitutions total. Reapplying all three regular expressions to the redacted prompt texts finds 0 residual matches. Each model user message is exactly the redacted ticket text, without any wrapper or additional instruction.

### Prompts, retries and output validation
Both system prompts contain `CLASSIFY`, name the four allowed categories (`billing`, `delivery`, `technical`, `fraud`) and require JSON-only output with `category` and `reason`. The zero-shot prompt contains no `EXAMPLES:` line. The few-shot prompt includes labelled examples for fraud, billing and technical. Every call is wrapped in up to three retries for `TransientLLMError`, using exponential backoff with jitter (`0.05 × 2**n × uniform(0.9, 1.15)`) and a log line containing ticket ID, retry number and wait. Each full run resets the mock immediately before processing; the mock produced 19 transient failures per full run and every one recovered on retry. A valid JSON object with an allowed category is `classified_json`; otherwise the parser extracts lowercase `[a-z]+` words and accepts exactly one allowed category as `classified_keyword`. All other outputs, including valid JSON with an unsupported category, become `flagged_invalid_output` and go to human review.

### Zero-shot versus few-shot results
Both full runs had 164 input rows: 4 skipped empty, 4 flagged long, 147 classified (135 JSON and 12 keyword), and 9 flagged invalid. Accuracy is calculated only over the 147 classified tickets.

| Run | Classified | Accuracy | Total tokens | Tokens / classified ticket |
|---|---:|---:|---:|---:|
| Zero-shot | 147 | 85.7% (126/147) | 12,052 | 81.99 |
| Few-shot | 147 | 100.0% (147/147) | 22,504 | 153.09 |

Zero-shot confusion table (rows = true category, columns = predicted category):

| True \\ Predicted | billing | delivery | technical | fraud |
|---|---:|---:|---:|---:|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| technical | 10 | 0 | 29 | 0 |
| fraud | 11 | 0 | 0 | 23 |

The few-shot run costs 10,452 additional tokens overall (71.10 more tokens per classified ticket), but improves accuracy by 14.3 percentage points and eliminates all 21 zero-shot classification errors (11 fraud and 10 technical tickets predicted as billing). For a support workflow where fraud and technical misroutes have meaningful downstream cost, the extra prompt tokens are worth paying for; for low-risk high-volume triage, a cheaper zero-shot pass followed by confidence or rule-based review may be preferable.

### Budget, routing and saved outputs
The zero-shot budget demonstration uses `TOKEN_BUDGET=6000`: the successful calls consume 6,046 tokens in total, and 78 later eligible tickets are marked `budget_exhausted` because the running total was already at or above the budget before their call. The final few-shot run uses `TOKEN_BUDGET=100000`. Routing is applied in this order: guardrail/invalid/budget statuses → `human_review` / NORMAL; fraud → `security_team` / HIGH; Premium customer → `premium_desk` / HIGH; otherwise → `support_queue` / NORMAL. Customers absent from the segment table are treated as not Premium.

- `outputs/triage_budget_demo.csv`: zero-shot budget demonstration.
- `outputs/triage_results.csv`: final 164-row few-shot result with columns `ticket_id, customer_id, status, category, segment_name, priority, route, tokens_used`.

The final routing counts are 102 `support_queue`, 34 `security_team`, 17 `human_review`, and 11 `premium_desk`. Model-generated categories that passed the JSON or keyword rules are routed automatically under the stated rules; skipped, long, invalid-output and budget-exhausted tickets are routed to human review.
