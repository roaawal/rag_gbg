"""
The actual "front door" of the system: classify_route() runs first for
every question, then dispatch to whichever route it picked. This is what
chat.py (and anything else) should call instead of hitting rag.py or
advanced_rag.py directly, so the router is never accidentally bypassed.
"""
import time
import uuid

import requests

import config
import generation
import router
import rag
import advanced_rag
import evaluate
from logging_utils import append_record


def _answer_simple_direct(question: str) -> dict:
    total_start = time.perf_counter()
    prompt = generation.build_direct_prompt(question)
    try:
        result = generation.call_llm(
            prompt, arabic=generation.is_arabic(question), stage="direct_generation"
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(generation.connection_error_message())

    input_tokens = result["input_tokens"]
    output_tokens = result["output_tokens"]
    cost = (
        (input_tokens / 1000) * config.COST_PER_1K_INPUT_TOKENS
        + (output_tokens / 1000) * config.COST_PER_1K_OUTPUT_TOKENS
    )
    return {
        "record_id": str(uuid.uuid4()),
        "approach": "simple_llm_direct",
        "question": question,
        "chunk_ids": [],
        "retrieved_chunks": [],
        "context": "",
        "answer": result["text"],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": cost,
        "latency_seconds": result["latency_seconds"],
        "total_latency_seconds": time.perf_counter() - total_start,
    }


def answer(question: str, top_k: int = None, log: bool = True, verbose: bool = True) -> dict:
    """
    Routes and answers one question end to end. Always returns a record
    with at least: record_id, approach, question, chunk_ids, context,
    answer, input_tokens, output_tokens, cost, latency_seconds,
    total_latency_seconds -- plus route/router_reason/techniques_used,
    and (for advanced_rag) the extra fields advanced_rag.answer_advanced()
    adds. This shared schema is what lets evaluate.py score every route
    the same way.
    """
    tracking_token = generation.start_call_tracking()
    decision = router.classify_route(question)
    route = decision["route"]

    if verbose:
        print(f"\n[router] route={route}  techniques={decision['techniques']}")
        print(f"[router] reason: {decision['reason']}")

    if route == "simple_llm_direct":
        record = _answer_simple_direct(question)
    elif route == "advanced_rag":
        record = advanced_rag.answer_advanced(question, decision["techniques"], top_k=top_k)
    else:  # "rag"
        record = rag.answer_question_full(question, top_k=top_k)

    record["route"] = route
    record["router_reason"] = decision["reason"]
    record.setdefault("techniques_used", decision["techniques"])

    evaluated = evaluate.evaluate_record(record, log=False)
    calls = generation.stop_call_tracking(tracking_token)
    evaluated["llm_calls"] = calls
    evaluated["total_input_tokens"] = sum(c["input_tokens"] for c in calls)
    evaluated["total_output_tokens"] = sum(c["output_tokens"] for c in calls)
    evaluated["total_cost"] = sum(c["cost"] for c in calls)
    evaluated["total_latency_seconds"] = sum(c["latency_seconds"] for c in calls)
    evaluated["executed_path"] = [c["stage"] for c in calls]

    if log:
        append_record(config.EVAL_LOG, evaluated)

    return evaluated
