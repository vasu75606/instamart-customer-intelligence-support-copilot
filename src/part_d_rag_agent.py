from __future__ import annotations

import json
import re
import sqlite3
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = ROOT / 'data' / 'policies'
DATA_DIR = ROOT / 'data' / 'raw'
DB_PATH = ROOT / 'data' / 'instamart.db'
OUT = ROOT / 'outputs'
STATE = ROOT / 'state'
OUT.mkdir(exist_ok=True)
STATE.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))
import mock_llm


def load_chunks(overlap: int = 10) -> list[dict[str, str]]:
    """Split policy documents into 40-word chunks with the requested overlap."""
    if not 0 <= overlap < 40:
        raise ValueError('overlap must be between 0 and 39')
    chunks: list[dict[str, str]] = []
    for path in sorted(POLICY_DIR.glob('*.txt')):
        words = path.read_text(encoding='utf-8').split()
        start = 0
        while start < len(words):
            end = min(start + 40, len(words))
            chunks.append({'source': path.name, 'text': ' '.join(words[start:end])})
            if end >= len(words):
                break
            start += 40 - overlap
    return chunks


CHUNKS = load_chunks(10)
VECTORIZER = TfidfVectorizer()
CHUNK_MATRIX = VECTORIZER.fit_transform([c['text'] for c in CHUNKS])


def retrieve(query: str, k: int = 3, chunks=None, vectorizer=None, matrix=None) -> list[dict[str, Any]]:
    chunks = CHUNKS if chunks is None else chunks
    vectorizer = VECTORIZER if vectorizer is None else vectorizer
    matrix = CHUNK_MATRIX if matrix is None else matrix
    qv = vectorizer.transform([query])
    scores = cosine_similarity(qv, matrix).ravel()
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
    return [{'source': chunks[i]['source'], 'score': float(scores[i]), 'text': chunks[i]['text']} for i in order]


def answer(question: str) -> dict[str, Any]:
    hits = retrieve(question, k=3)
    context = '\n'.join(f"[{h['source']}] {h['text']}" for h in hits)
    user_content = f'CONTEXT:\n{context}\nQUESTION:\n{question}'
    messages = [
        {'role': 'system', 'content': 'You are a POLICY ASSISTANT. Answer only from the supplied policy context. If the context does not contain the answer, say exactly that you could not find the information.'},
        {'role': 'user', 'content': user_content},
    ]
    response = mock_llm.chat(messages)
    return {'answer': response.choices[0].message.content, 'sources': list(dict.fromkeys(h['source'] for h in hits)), 'tokens_used': response.usage.total_tokens}


def evaluate_rag(overlap: int) -> dict[str, Any]:
    chunks = load_chunks(overlap)
    vec = TfidfVectorizer()
    matrix = vec.fit_transform([c['text'] for c in chunks])
    questions = pd.read_csv(DATA_DIR / 'policy_eval.csv', keep_default_na=False)
    rows = []
    for r in questions.itertuples(index=False):
        hits = retrieve(r.question, 3, chunks, vec, matrix)
        context = '\n'.join(f"[{h['source']}] {h['text']}" for h in hits)
        messages = [
            {'role': 'system', 'content': 'You are a POLICY ASSISTANT. Answer only from the supplied context. If the answer is absent, say you could not find the information.'},
            {'role': 'user', 'content': f"CONTEXT:\n{context}\nQUESTION:\n{r.question}"},
        ]
        response = mock_llm.chat(messages)
        text = response.choices[0].message.content or ''
        expected_doc_found = (r.expected_doc == 'NONE' or r.expected_doc in [h['source'] for h in hits])
        expected_fact_found = (r.expected_doc == 'NONE' or r.expected_fact in text)
        fallback = text.startswith('I could not find')
        rows.append({'question': r.question, 'expected_doc': r.expected_doc,
                     'retrieved_sources': '|'.join(h['source'] for h in hits),
                     'answer': text, 'expected_doc_found': expected_doc_found,
                     'expected_fact_found': expected_fact_found, 'fallback': fallback})
    frame = pd.DataFrame(rows)
    answerable = frame[frame.expected_doc != 'NONE']
    unanswerable = frame[frame.expected_doc == 'NONE']
    return {
        'chunks': chunks, 'rows': frame,
        'answerable_doc_hits': int(answerable.expected_doc_found.sum()),
        'answerable_fact_hits': int(answerable.expected_fact_found.sum()),
        'unanswerable_fallbacks': int(unanswerable.fallback.sum()),
        'answerable_total': len(answerable), 'unanswerable_total': len(unanswerable),
    }


# Tool descriptions and parameter schemas are retained as a compact JSON-style tool registry.
TOOLS_SPEC = [
    {'name': 'get_order', 'description': 'Look up an order and its final delivery details.', 'parameters': {'type': 'object', 'properties': {'order_id': {'type': 'string'}}, 'required': ['order_id']}},
    {'name': 'search_policy', 'description': 'Retrieve the most relevant policy passages for a query.', 'parameters': {'type': 'object', 'properties': {'query': {'type': 'string'}}, 'required': ['query']}},
    {'name': 'calculate_refund', 'description': 'Calculate a late-delivery refund under the refund policy.', 'parameters': {'type': 'object', 'properties': {'order_amount': {'type': 'number'}, 'delivery_minutes': {'type': ['integer', 'null']}}, 'required': ['order_amount', 'delivery_minutes']}},
    {'name': 'issue_refund', 'description': 'Record an automatically approved refund in the simulated ledger.', 'parameters': {'type': 'object', 'properties': {'order_id': {'type': 'string'}, 'refund_inr': {'type': 'number'}}, 'required': ['order_id', 'refund_inr']}},
    {'name': 'escalate_to_human', 'description': 'Record a refund requiring human approval in the simulated ledger.', 'parameters': {'type': 'object', 'properties': {'order_id': {'type': 'string'}, 'refund_inr': {'type': 'number'}}, 'required': ['order_id', 'refund_inr']}},
]


def get_order(order_id: str) -> dict[str, Any]:
    with sqlite3.connect(DB_PATH) as con:
        row = con.execute('SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?', (order_id,)).fetchone()
    if row is None:
        return {'error': 'order not found'}
    return {'order_id': row[0], 'status': row[1], 'order_amount': float(row[2]),
            'delivery_minutes': None if row[3] is None else int(row[3])}


def search_policy(query: str) -> dict[str, Any]:
    return {'results': retrieve(query, k=3)}


def calculate_refund(order_amount: float, delivery_minutes: int | None) -> dict[str, float]:
    amount = round(float(order_amount), 2) if delivery_minutes is not None and int(delivery_minutes) > 45 else 0.0
    return {'refund_inr': amount}


def _ledger(tool_name: str, order_id: str, refund_inr: float) -> dict[str, float]:
    entry = {'timestamp': datetime.now(timezone.utc).isoformat(), 'tool_name': tool_name,
             'order_id': order_id, 'refund_inr': round(float(refund_inr), 2)}
    with (OUT / 'refund_ledger.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + '\n')
    return {'refund_inr': entry['refund_inr']}


def issue_refund(order_id: str, refund_inr: float) -> dict[str, Any]:
    order = get_order(order_id)
    if 'error' in order:
        return {'error': 'order not found'}
    if order['status'] != 'completed':
        return {'error': 'only completed orders can be refunded'}
    if float(refund_inr) > 1000:
        return {'error': 'refund above INR 1000 requires human approval'}
    return _ledger('issue_refund', order_id, refund_inr)


def escalate_to_human(order_id: str, refund_inr: float) -> dict[str, Any]:
    order = get_order(order_id)
    if 'error' in order:
        return {'error': 'order not found'}
    if order['status'] != 'completed':
        return {'error': 'only completed orders can be refunded'}
    if float(refund_inr) <= 1000:
        return {'error': 'refunds up to INR 1000 are eligible for automatic approval'}
    return _ledger('escalate_to_human', order_id, refund_inr)


TOOL_MAP = {'get_order': get_order, 'search_policy': search_policy,
            'calculate_refund': calculate_refund, 'issue_refund': issue_refund,
            'escalate_to_human': escalate_to_human}


class SimulatedCrash(RuntimeError):
    pass


def _tool_calls_as_dicts(tool_calls) -> list[dict[str, Any]]:
    out = []
    for tc in tool_calls or []:
        out.append({'id': getattr(tc, 'id', ''), 'name': getattr(tc, 'name', ''),
                    'arguments': getattr(tc, 'arguments', {})})
    return out


def _checkpoint_path(run_id: str) -> Path:
    return STATE / f'{run_id}.json'


def _save_checkpoint(run_id: str, scenario_id: str, messages: list[dict[str, Any]],
                     tokens_used: int, step_number: int, tools_executed: list[dict[str, Any]],
                     user_message: str, llm_calls: int) -> None:
    payload = {'run_id': run_id, 'scenario_id': scenario_id, 'messages': messages,
               'tokens_used': tokens_used, 'step_number': step_number,
               'tools_executed': tools_executed, 'user_message': user_message,
               'llm_calls': llm_calls}
    _checkpoint_path(run_id).write_text(json.dumps(payload, indent=2), encoding='utf-8')


def run_agent(user_message: str, scenario_id: str, max_iterations: int = 6,
              token_limit: int = 6000, time_limit_seconds: float = 10,
              run_id: str | None = None, crash_after_tool_executions: int | None = None) -> dict[str, Any]:
    run_id = run_id or f'{scenario_id.lower()}_{uuid.uuid4().hex[:10]}'
    messages: list[dict[str, Any]] = [
        {'role': 'system', 'content': 'You are an AGENT. Use the available tools to follow the refund policy and never invent order facts.'},
        {'role': 'user', 'content': user_message},
    ]
    return _agent_loop(run_id, scenario_id, user_message, messages, 0, 0, [],
                       max_iterations, token_limit, time_limit_seconds, crash_after_tool_executions)


def _agent_loop(run_id: str, scenario_id: str, user_message: str, messages: list[dict[str, Any]],
                tokens_used: int, llm_calls: int, tools_executed: list[dict[str, Any]],
                max_iterations: int, token_limit: int, time_limit_seconds: float,
                crash_after_tool_executions: int | None = None) -> dict[str, Any]:
    started = time.monotonic()
    steps: list[dict[str, Any]] = list(tools_executed)
    while llm_calls < max_iterations:
        if time.monotonic() - started >= time_limit_seconds:
            return {'status': 'FALLBACK_TIMEOUT', 'final_answer': 'FALLBACK_TIMEOUT: hand off to a human agent; time limit reached.',
                    'steps': steps, 'llm_calls': llm_calls, 'tokens_used': tokens_used, 'run_id': run_id}
        response = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += int(response.usage.total_tokens)
        if tokens_used > token_limit:
            _save_checkpoint(run_id, scenario_id, messages, tokens_used, len(steps), steps, user_message, llm_calls)
            return {'status': 'FALLBACK_TOKEN_LIMIT', 'final_answer': 'FALLBACK_TOKEN_LIMIT: hand off to a human agent; token limit reached.',
                    'steps': steps, 'llm_calls': llm_calls, 'tokens_used': tokens_used, 'run_id': run_id}
        msg = response.choices[0].message
        calls = _tool_calls_as_dicts(getattr(msg, 'tool_calls', []))
        if not calls:
            final = msg.content or 'FALLBACK: hand off to a human agent; model returned no answer.'
            status = final.split(':', 1)[0].strip()
            _save_checkpoint(run_id, scenario_id, messages, tokens_used, len(steps), steps, user_message, llm_calls)
            return {'status': status, 'final_answer': final, 'steps': steps,
                    'llm_calls': llm_calls, 'tokens_used': tokens_used, 'run_id': run_id}
        messages.append({'role': 'assistant', 'content': None, 'tool_calls': calls})
        for call in calls:
            name = call['name']
            args = call['arguments']
            if isinstance(args, str):
                args = json.loads(args)
            if name not in TOOL_MAP:
                observation = {'error': f'unknown tool: {name}'}
            else:
                observation = TOOL_MAP[name](**args)
            step = {'tool_name': name, 'arguments': args, 'observation': observation}
            steps.append(step)
            tools_executed.append(step)
            messages.append({'role': 'tool', 'tool_call_id': call['id'], 'name': name,
                             'content': json.dumps(observation, ensure_ascii=False)})
            _save_checkpoint(run_id, scenario_id, messages, tokens_used, len(steps), steps,
                             user_message, llm_calls)
            if crash_after_tool_executions is not None and len(tools_executed) == crash_after_tool_executions:
                raise SimulatedCrash(f'simulated crash after tool execution {crash_after_tool_executions}')
    final = 'FALLBACK_MAX_ITERATIONS: hand off to a human agent; maximum model calls reached.'
    _save_checkpoint(run_id, scenario_id, messages, tokens_used, len(steps), steps, user_message, llm_calls)
    return {'status': 'FALLBACK_MAX_ITERATIONS', 'final_answer': final, 'steps': steps,
            'llm_calls': llm_calls, 'tokens_used': tokens_used, 'run_id': run_id}


def resume_agent(run_id: str, max_iterations: int = 6, token_limit: int = 6000,
                 time_limit_seconds: float = 10) -> dict[str, Any]:
    path = _checkpoint_path(run_id)
    payload = json.loads(path.read_text(encoding='utf-8'))
    return _agent_loop(run_id, payload['scenario_id'], payload['user_message'], payload['messages'],
                       int(payload['tokens_used']), int(payload['llm_calls']), payload['tools_executed'],
                       max_iterations, token_limit, time_limit_seconds)


def _refund_amount(result: dict[str, Any]) -> float:
    for step in result['steps']:
        if step['tool_name'] in {'calculate_refund', 'issue_refund', 'escalate_to_human'}:
            value = step['observation'].get('refund_inr')
            if value is not None:
                return float(value)
    return 0.0


def run_scenarios() -> pd.DataFrame:
    scenarios = pd.read_csv(DATA_DIR / 'agent_scenarios.csv', keep_default_na=False)
    records = []
    with (OUT / 'agent_runs.jsonl').open('w', encoding='utf-8') as log:
        for r in scenarios.itertuples(index=False):
            result = run_agent(r.message, r.scenario_id)
            row = {'scenario_id': r.scenario_id, 'status': result['status'],
                   'final_answer': result['final_answer'], 'refund_amount': _refund_amount(result),
                   'llm_calls': result['llm_calls'], 'tokens_used': result['tokens_used']}
            records.append(row)
            log.write(json.dumps({'scenario_id': r.scenario_id, 'status': result['status'],
                                  'final_answer': result['final_answer'], 'llm_calls': result['llm_calls'],
                                  'tokens_used': result['tokens_used'], 'steps': result['steps']}) + '\n')
            expected_status = r.expected_outcome
            assert (result['status'] == expected_status or
                    (expected_status == 'FALLBACK' and result['status'] == 'FALLBACK_MAX_ITERATIONS'))
            assert abs(row['refund_amount'] - float(r.expected_refund_inr)) <= 0.01
    frame = pd.DataFrame(records)
    frame.to_csv(OUT / 'agent_scenario_results.csv', index=False)
    return frame


def run_demos() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    scenarios = pd.read_csv(DATA_DIR / 'agent_scenarios.csv', keep_default_na=False)
    s1 = scenarios.loc[scenarios.scenario_id == 'S1'].iloc[0]
    try:
        run_agent(s1.message, 'S1_CRASH_RESUME', run_id='s1_crash_resume', crash_after_tool_executions=2)
    except SimulatedCrash:
        pass
    resumed = resume_agent('s1_crash_resume')
    expected_tools = ['get_order', 'search_policy', 'calculate_refund', 'issue_refund']
    actual_tools = [s['tool_name'] for s in resumed['steps']]
    assert actual_tools == expected_tools, (actual_tools, expected_tools)
    assert resumed['status'] == 'REFUND' and abs(_refund_amount(resumed) - 128.25) < 0.01

    token_demo = run_agent(s1.message, 'S1_TOKEN_DEMO', token_limit=200, run_id='s1_token_demo')
    timeout_demo = run_agent(s1.message, 'S1_TIMEOUT_DEMO', time_limit_seconds=0, run_id='s1_timeout_demo')
    assert token_demo['status'] == 'FALLBACK_TOKEN_LIMIT'
    assert timeout_demo['status'] == 'FALLBACK_TIMEOUT'
    return resumed, token_demo, timeout_demo


def main():
    print('POLICY CHUNKS:', len(CHUNKS), pd.Series([c['source'] for c in CHUNKS]).value_counts().sort_index().to_dict())
    top = retrieve('refund policy late delivery', 3)
    print('TOP-3 REFUND QUERY:', [(x['source'], round(x['score'], 4)) for x in top])
    assert len(CHUNKS) == 15
    assert all(x['source'] == 'refund_policy.txt' for x in top)

    # Independent batch evaluation; reset the deterministic mock before each complete run.
    mock_llm.reset()
    overlap_eval = evaluate_rag(10)
    mock_llm.reset()
    no_overlap_eval = evaluate_rag(0)
    overlap_eval['rows'].to_csv(OUT / 'policy_eval_overlap.csv', index=False)
    no_overlap_eval['rows'].to_csv(OUT / 'policy_eval_no_overlap.csv', index=False)
    print('RAG OVERLAP=10:', overlap_eval['answerable_doc_hits'], '/', overlap_eval['answerable_total'],
          'docs;', overlap_eval['answerable_fact_hits'], '/', overlap_eval['answerable_total'],
          'facts;', overlap_eval['unanswerable_fallbacks'], '/', overlap_eval['unanswerable_total'], 'fallbacks')
    print('RAG OVERLAP=0:', no_overlap_eval['answerable_doc_hits'], '/', no_overlap_eval['answerable_total'],
          'docs;', no_overlap_eval['answerable_fact_hits'], '/', no_overlap_eval['answerable_total'],
          'facts;', no_overlap_eval['unanswerable_fallbacks'], '/', no_overlap_eval['unanswerable_total'], 'fallbacks')
    assert overlap_eval['answerable_doc_hits'] == 12
    assert overlap_eval['answerable_fact_hits'] == 12
    assert overlap_eval['unanswerable_fallbacks'] == 2
    assert no_overlap_eval['answerable_fact_hits'] < 12

    # Reset before independent agent scenario batch.
    mock_llm.reset()
    if (OUT / 'refund_ledger.jsonl').exists():
        (OUT / 'refund_ledger.jsonl').unlink()
    results = run_scenarios()
    print('\nSCENARIO RESULTS\n', results.to_string(index=False))
    resumed, token_demo, timeout_demo = run_demos()
    print('\nCRASH/RESUME:', resumed['status'], _refund_amount(resumed),
          [s['tool_name'] for s in resumed['steps']])
    print('TOKEN DEMO:', token_demo['status'], token_demo['llm_calls'], token_demo['tokens_used'])
    print('TIMEOUT DEMO:', timeout_demo['status'], timeout_demo['llm_calls'], timeout_demo['tokens_used'])
    assert results.loc[results.scenario_id == 'S1', 'llm_calls'].iloc[0] == 5
    assert results.loc[results.scenario_id == 'S7', 'status'].iloc[0] == 'FALLBACK_MAX_ITERATIONS'
    print('\nPART D VALIDATION: ALL CHECKS PASSED')


if __name__ == '__main__':
    main()
