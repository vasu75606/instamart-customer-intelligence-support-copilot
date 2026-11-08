from pathlib import Path
import json, random, re, sys, time
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
DATA = ROOT / 'data' / 'raw'
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))
import mock_llm

ALLOWED = {'billing', 'delivery', 'technical', 'fraud'}
MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3

ZERO_SHOT_SYSTEM = '''You are a support ticket classifier. CLASSIFY each customer message into exactly one allowed category: billing, delivery, technical, fraud. Return a JSON-only reply of the form {"category": "...", "reason": "..."}. Do not include markdown or additional text.'''
FEW_SHOT_SYSTEM = '''You are a support ticket classifier. CLASSIFY each customer message into exactly one allowed category: billing, delivery, technical, fraud. Return a JSON-only reply of the form {"category": "...", "reason": "..."}. Do not include markdown or additional text.
EXAMPLES:
Customer: My card was used without permission and my account was hacked.
JSON: {"category": "fraud", "reason": "unauthorized account activity"}
Customer: The invoice shows that I was charged twice.
JSON: {"category": "billing", "reason": "duplicate charge"}
Customer: The app crashes whenever I try to log in.
JSON: {"category": "technical", "reason": "app login failure"}'''

PATTERNS = [
    (re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+'), '[EMAIL]', 'email'),
    (re.compile(r'\b[6-9]\d{9}\b'), '[PHONE]', 'phone'),
    (re.compile(r'ORD-\d{6}'), '[ORDER_ID]', 'order_id'),
]

def redact(text):
    counts = {}
    value = text if isinstance(text, str) else ''
    for pattern, replacement, name in PATTERNS:
        value, n = pattern.subn(replacement, value)
        counts[name] = n
    return value, counts

def classify_reply(content):
    try:
        obj = json.loads(content)
        if isinstance(obj, dict) and obj.get('category') in ALLOWED:
            return obj['category'], 'classified_json'
        return '', 'flagged_invalid_output'
    except (json.JSONDecodeError, TypeError):
        words = re.findall(r'[a-z]+', str(content).lower())
        found = [w for w in words if w in ALLOWED]
        if len(found) == 1:
            return found[0], 'classified_keyword'
        return '', 'flagged_invalid_output'

def call_with_retry(messages, ticket_id):
    for retry_no in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except mock_llm.TransientLLMError:
            if retry_no >= MAX_RETRIES:
                raise
            wait = BASE_DELAY * (2 ** retry_no) * random.uniform(0.9, 1.15)
            print(f'RETRY ticket_id={ticket_id} retry={retry_no} wait={wait:.5f}s')
            time.sleep(wait)
    raise RuntimeError('unreachable retry state')

def run_pipeline(tickets, system_prompt, token_budget=None):
    mock_llm.reset()
    random.seed(42)
    total_tokens = 0
    rows = []
    prompt_audit = []
    for row in tickets.itertuples(index=False):
        raw = row.text if isinstance(row.text, str) else ''
        redacted, _ = redact(raw)
        if not raw.strip():
            rows.append(dict(ticket_id=row.ticket_id, customer_id=row.customer_id, status='skipped_empty', category='', tokens_used=0, true_category=row.true_category))
            continue
        if mock_llm.est_tokens(raw) > MAX_INPUT_TOKENS:
            rows.append(dict(ticket_id=row.ticket_id, customer_id=row.customer_id, status='flagged_long', category='', tokens_used=0, true_category=row.true_category))
            continue
        if token_budget is not None and total_tokens >= token_budget:
            rows.append(dict(ticket_id=row.ticket_id, customer_id=row.customer_id, status='budget_exhausted', category='', tokens_used=0, true_category=row.true_category))
            continue
        # The user message is exactly the redacted ticket text, with no wrapper.
        assert not re.search(PATTERNS[0][0], redacted)
        assert not re.search(PATTERNS[1][0], redacted)
        assert not re.search(PATTERNS[2][0], redacted)
        messages = [{'role':'system','content':system_prompt}, {'role':'user','content':redacted}]
        prompt_audit.append(messages[-1]['content'])
        response = call_with_retry(messages, row.ticket_id)
        used = response.usage.total_tokens
        total_tokens += used
        content = response.choices[0].message.content or ''
        category, status = classify_reply(content)
        rows.append(dict(ticket_id=row.ticket_id, customer_id=row.customer_id, status=status, category=category, tokens_used=used, true_category=row.true_category))
    # Redaction assertion across all prompts actually sent.
    for msg in prompt_audit:
        assert not re.search(PATTERNS[0][0], msg)
        assert not re.search(PATTERNS[1][0], msg)
        assert not re.search(PATTERNS[2][0], msg)
    return pd.DataFrame(rows), total_tokens

def main():
    tickets = pd.read_csv(DATA / 'tickets.csv', keep_default_na=False)
    print('TICKET AUDIT rows:', len(tickets), '| empty:', int(tickets.text.str.strip().eq('').sum()),
          '| long (>400 est tokens):', int(tickets.text.map(mock_llm.est_tokens).gt(MAX_INPUT_TOKENS).sum()))
    counts = {'email':0, 'phone':0, 'order_id':0}
    redacted_texts=[]
    for text in tickets.text:
        red, c = redact(text); redacted_texts.append(red)
        for k,v in c.items(): counts[k] += v
    print('REDACTION COUNTS:', counts, '| total:', sum(counts.values()))
    assert all(not p.search(t) for t in redacted_texts for p,_,_ in PATTERNS)
    print('REDACTION RESIDUALS: 0')

    zero, zero_tokens = run_pipeline(tickets, ZERO_SHOT_SYSTEM)
    few, few_tokens = run_pipeline(tickets, FEW_SHOT_SYSTEM)
    for label, frame, tokens in [('ZERO-SHOT',zero,zero_tokens),('FEW-SHOT',few,few_tokens)]:
        classified = frame[frame.status.isin(['classified_json','classified_keyword'])]
        acc = (classified.category == classified.true_category).mean()
        print(f'\n{label} STATUS COUNTS:\n{frame.status.value_counts().to_string()}')
        print(f'{label} classified={len(classified)} accuracy={acc:.4%} total_tokens={tokens} tokens_per_classified={tokens/len(classified):.2f}')
    zero_class = zero[zero.status.isin(['classified_json','classified_keyword'])]
    confusion = pd.crosstab(zero_class.true_category, zero_class.category, dropna=False).reindex(index=['billing','delivery','technical','fraud'], columns=['billing','delivery','technical','fraud'], fill_value=0)
    print('\nZERO-SHOT CONFUSION MATRIX\n', confusion.to_string())

    budget, budget_tokens = run_pipeline(tickets, ZERO_SHOT_SYSTEM, token_budget=6000)
    budget.to_csv(OUT / 'triage_budget_demo.csv', index=False)
    exhausted = int(budget.status.eq('budget_exhausted').sum())
    print(f'\nBUDGET DEMO: budget=6000 tokens_used={budget_tokens} budget_exhausted={exhausted}')
    if exhausted == 0:
        budget, budget_tokens = run_pipeline(tickets, ZERO_SHOT_SYSTEM, token_budget=5000)
        budget.to_csv(OUT / 'triage_budget_demo.csv', index=False)
        exhausted = int(budget.status.eq('budget_exhausted').sum())
        print(f'BUDGET DEMO RETRY: budget=5000 tokens_used={budget_tokens} budget_exhausted={exhausted}')
    assert exhausted > 0

    # Final few-shot production run with budget and segment-aware routing.
    final, final_tokens = run_pipeline(tickets, FEW_SHOT_SYSTEM, token_budget=100000)
    seg = pd.read_csv(OUT / 'customer_segments.csv')
    seg = seg[['customer_id','segment_name','is_premium']]
    final = final.merge(seg, on='customer_id', how='left')
    final['is_premium'] = final['is_premium'].fillna(False).astype(bool)
    routes=[]; priorities=[]
    for r in final.itertuples(index=False):
        if r.status in {'skipped_empty','flagged_long','flagged_invalid_output','budget_exhausted'}:
            routes.append('human_review'); priorities.append('NORMAL')
        elif r.category == 'fraud':
            routes.append('security_team'); priorities.append('HIGH')
        elif r.is_premium:
            routes.append('premium_desk'); priorities.append('HIGH')
        else:
            routes.append('support_queue'); priorities.append('NORMAL')
    final['route']=routes; final['priority']=priorities
    final['segment_name']=final['segment_name'].fillna('')
    final['category']=final['category'].fillna('')
    final[['ticket_id','customer_id','status','category','segment_name','priority','route','tokens_used']].to_csv(OUT/'triage_results.csv', index=False)
    print('\nFINAL ROUTING COUNTS\n', final.route.value_counts().to_string())
    print('FINAL STATUS COUNTS\n', final.status.value_counts().to_string())
    print('FINAL OUTPUT rows:', len(final), '| tokens:', final_tokens)
    assert len(final) == 164

if __name__ == '__main__':
    main()
