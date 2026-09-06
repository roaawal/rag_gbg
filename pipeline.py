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
import reference_lookup
from logging_utils import append_record


def _answer_direct(question: str) -> dict:
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
        "approach": "direct",
        "question": question,
        "executed_steps": ["router", "generation"],
        "retrieval_stats": {"num_candidates": 0, "num_after_reranking": None, "num_after_compression": None},
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


def _print_router_decision(decision: dict):
    print(f"\n[router]")
    print(f"route={decision['route']}")
    print(f"complexity={decision['complexity']}")
    print(f"requires_documents={decision['requires_documents']}")
    print(f"multiple_information_needs={decision['multiple_information_needs']}")
    print(f"multiple_search_perspectives={decision['multiple_search_perspectives']}")
    print(f"has_metadata_constraints={decision['has_metadata_constraints']}")
    print(f"is_vague_or_conversational={decision['is_vague_or_conversational']}")
    print(f"techniques={decision['techniques']}")
    print(f"reason={decision['reason']}")


def _print_summary(evaluated: dict):
    print(
        f"\n[evaluation]\n"
        f"context_relevance={evaluated.get('context_relevance')}\n"
        f"faithfulness={evaluated.get('faithfulness')}\n"
        f"answer_relevance={evaluated.get('answer_relevance')}\n"
        f"correctness={evaluated.get('correctness')}"
    )
    print(
        f"\n[cost]\n"
        f"total_input_tokens={evaluated.get('total_input_tokens')}\n"
        f"total_output_tokens={evaluated.get('total_output_tokens')}\n"
        f"total_cost={evaluated.get('total_cost')}\n"
        f"total_latency_ms={evaluated.get('total_latency_ms')}"
    )


def answer(question: str, top_k: int = None, log: bool = True, verbose: bool = True,
           reference: str = None) -> dict:
    """
    Routes and answers one question end to end. Always returns a record
    with at least: record_id, approach, question, chunk_ids, context,
    answer, input_tokens, output_tokens, cost, latency_seconds,
    total_latency_seconds -- plus route/router_reason/router_decision/
    techniques_used/executed_steps/retrieval, and (for advanced_rag) the
    extra fields advanced_rag.answer_advanced() adds. This shared schema
    is what lets evaluate.py score every route the same way.

    `reference` is an optional ground-truth answer to score `correctness`
    against. If omitted (the normal case -- chat.py never passes one),
    reference_lookup.find_reference() checks `question` against
    reference_answers.json's known evaluation questions and uses that
    match's answer if found -- so asking one of those questions verbatim
    (or with minor copy/paste differences) gets a real correctness score
    instead of the usual None, with no extra plumbing needed at the
    call site. Pass `reference` explicitly to override/skip the lookup.
    """
    question_id = str(uuid.uuid4())
    tracking_token = generation.start_call_tracking()
    decision = router.classify_route(question)
    route = decision["route"]

    if verbose:
        _print_router_decision(decision)

    if route == "direct":
        record = _answer_direct(question)
    elif route == "advanced_rag":
        record = advanced_rag.answer_advanced(question, decision, top_k=top_k)
    else:  # "rag"
        record = rag.answer_question_full(question, top_k=top_k)

    record["route"] = route
    record["router_reason"] = decision["reason"]
    record["router_decision"] = decision
    record.setdefault("techniques_used", decision["techniques"])
    record.setdefault("executed_steps", ["router", "generation"])
    record.setdefault(
        "retrieval_stats",
        {"num_candidates": len(record.get("chunk_ids", [])), "num_after_reranking": None, "num_after_compression": None},
    )

    if reference is None:
        reference = reference_lookup.find_reference(question)

    # evaluate.py scores Direct/Basic/Advanced identically -- one fixed
    # rubric, applied every time, regardless of route. `reference` (explicit
    # or auto-matched above) is what lets `correctness` be scored instead of
    # staying None -- evaluate_record() already handles reference=None
    # gracefully, so this is a no-op for any question with no known reference.
    evaluated = evaluate.evaluate_record(record, reference=reference, log=False)
    calls = generation.stop_call_tracking(tracking_token)
    evaluated["llm_calls"] = calls
    evaluated["total_input_tokens"] = sum(c["input_tokens"] for c in calls)
    evaluated["total_output_tokens"] = sum(c["output_tokens"] for c in calls)
    evaluated["total_cost"] = sum(c["cost"] for c in calls)
    evaluated["total_latency_seconds"] = sum(c["latency_seconds"] for c in calls)
    evaluated["total_latency_ms"] = evaluated["total_latency_seconds"] * 1000
    evaluated["executed_path"] = [c["stage"] for c in calls]

    # Structured record shape for downstream consumers/logging (question_id,
    # nested router_decision/retrieval/evaluation) -- kept alongside the
    # existing flat fields above so nothing already reading them breaks.
    evaluated["question_id"] = question_id
    evaluated["retrieval"] = evaluated.get("retrieval_stats")
    evaluated["evaluation"] = {
        "context_relevance": evaluated.get("context_relevance"),
        "faithfulness": evaluated.get("faithfulness"),
        "answer_relevance": evaluated.get("answer_relevance"),
        "correctness": evaluated.get("correctness"),
    }

    if verbose:
        _print_summary(evaluated)

    if log:
        append_record(config.EVAL_LOG, evaluated)

    return evaluated
