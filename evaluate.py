"""
Per-question evaluation: scores any RAG record (from basic_rag.answer()
or rag.answer_question_full()) against four metrics, using ONE fixed
rubric so scores are comparable across approaches:

  - Context Relevance : did retrieval return useful evidence?
  - Faithfulness       : is the answer supported by the retrieved context?
  - Answer Relevance   : does the answer actually answer the question?
  - Correctness        : matches a trusted reference answer (only scored
                          when a reference is supplied -- otherwise null)

CAVEAT: the judge model is, by default, the same local model
(config.JUDGE_MODEL) used for generation. Even a 7B-class model is a
comparatively weak judge next to a frontier model -- treat these scores
as indicative/directional, useful for spotting large regressions between
approaches, not as ground truth. Point config.JUDGE_MODEL at a stronger
model in LM Studio for anything you need to trust more.
"""
import json
import re
import time
import uuid

import config
import generation
from logging_utils import append_record, read_records


_RUBRIC = """You are a strict evaluator for a document question-answering system.
Score the given ANSWER on these four metrics, each on a 1-5 integer scale:

1. context_relevance (1-5): Does the CONTEXT contain information that is
   actually relevant to answering the QUESTION? 5 = context is squarely
   on-topic and covers what's needed; 3 = partially relevant, mixed with
   irrelevant material; 1 = context is unrelated to the question.

2. faithfulness (1-5): Is every factual claim in the ANSWER actually
   supported by the CONTEXT (no hallucination, no outside knowledge)?
   5 = fully grounded, nothing invented; 3 = mostly grounded but some
   unsupported claims or extrapolation; 1 = answer contradicts or
   fabricates content not present in the context.

3. answer_relevance (1-5): Does the ANSWER directly address what the
   QUESTION actually asked, without padding or going off-topic?
   5 = fully and directly answers it; 3 = partially answers or includes
   significant irrelevant content; 1 = does not address the question.

4. correctness (1-5 or null): ONLY if a REFERENCE ANSWER is provided below,
   compare the ANSWER's key facts against it and score how correct it is.
   5 = matches the reference's key facts; 3 = partially correct; 1 =
   contradicts the reference. If no REFERENCE ANSWER is provided, output
   null for this field -- do not guess a score.

Respond with ONLY a single JSON object, no markdown fences, no extra text:
{"context_relevance": <1-5>, "faithfulness": <1-5>, "answer_relevance": <1-5>, "correctness": <1-5 or null>, "notes": "<one short sentence explaining the weakest score>"}
"""


def _extract_json(text: str) -> dict:
    """Judge models sometimes wrap JSON in markdown fences or add stray
    text around it -- pull out the first {...} block and parse that."""
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in judge response: {text[:200]!r}")
    return json.loads(match.group(0))


def _call_judge(question: str, context: str, answer: str, reference: str = None) -> dict:
    reference_block = (
        f"\nREFERENCE ANSWER:\n{reference}\n" if reference else "\n(No reference answer supplied.)\n"
    )
    user_prompt = f"""QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}
{reference_block}
Score the ANSWER now."""

    result = generation.call_completion(
        messages=[
            {"role": "system", "content": _RUBRIC},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=config.JUDGE_MAX_NEW_TOKENS,
        temperature=config.JUDGE_TEMPERATURE,
        stage="evaluation_judge",
        model=config.JUDGE_MODEL,
        timeout=180,
    )
    raw_text = result["text"]

    try:
        scores = _extract_json(raw_text)
    except (ValueError, json.JSONDecodeError) as e:
        scores = {
            "context_relevance": None,
            "faithfulness": None,
            "answer_relevance": None,
            "correctness": None,
            "notes": f"JUDGE PARSE FAILURE: {e}. Raw: {raw_text[:300]!r}",
        }
    return scores


def evaluate_record(rag_record: dict, reference: str = None, log: bool = True) -> dict:
    """
    Scores one RAG record (as produced by basic_rag.answer() or
    rag.answer_question_full() -- both share the same schema) against
    the fixed four-metric rubric, and logs the combined record to
    config.EVAL_LOG.
    """
    start = time.perf_counter()
    try:
        scores = _call_judge(
            question=rag_record["question"],
            context=rag_record["context"],
            answer=rag_record["answer"],
            reference=reference,
        )
    except Exception as exc:
        scores = {
            "context_relevance": None,
            "faithfulness": None,
            "answer_relevance": None,
            "correctness": None,
            "notes": f"JUDGE CALL FAILURE: {type(exc).__name__}: {exc}",
        }
    judge_latency = time.perf_counter() - start

    combined = dict(rag_record)
    if not rag_record.get("context"):
        scores["context_relevance"] = None
        scores["faithfulness"] = None
    if not reference:
        scores["correctness"] = None
    judge_notes = scores.get("notes")
    if not isinstance(judge_notes, str) or not judge_notes.strip():
        scored_metrics = {
            name: scores.get(name)
            for name in ("context_relevance", "faithfulness", "answer_relevance", "correctness")
            if isinstance(scores.get(name), (int, float))
        }
        if scored_metrics:
            weakest_metric = min(scored_metrics, key=scored_metrics.get)
            judge_notes = (
                f"The judge omitted notes; the weakest available score was "
                f"{weakest_metric} ({scored_metrics[weakest_metric]}/5)."
            )
        else:
            judge_notes = "The judge did not return an explanatory note or usable scores."
    combined.update({
        "eval_record_id": str(uuid.uuid4()),
        "reference_answer": reference,
        "context_relevance": scores.get("context_relevance"),
        "faithfulness": scores.get("faithfulness"),
        "answer_relevance": scores.get("answer_relevance"),
        "correctness": scores.get("correctness"),
        "judge_notes": judge_notes,
        "judge_model": config.JUDGE_MODEL,
        "judge_latency_seconds": judge_latency,
        "evaluation_rubric": "fixed_1_to_5_context_faithfulness_relevance_correctness_v1",
        "reference_available": bool(reference),
    })

    if log:
        append_record(config.EVAL_LOG, combined)

    return combined


def summarize(path: str = None) -> dict:
    """
    Aggregates every logged eval record by `approach`, returning mean
    scores per metric so you can compare basic_rag vs hybrid_rag (or
    any other approach tag) at a glance.
    """
    path = path or config.EVAL_LOG
    records = read_records(path)

    by_approach = {}
    for r in records:
        approach = r.get("approach", "unknown")
        by_approach.setdefault(approach, []).append(r)

    summary = {}
    metrics = ["context_relevance", "faithfulness", "answer_relevance", "correctness"]
    for approach, recs in by_approach.items():
        approach_summary = {"n": len(recs)}
        for metric in metrics:
            values = [r[metric] for r in recs if r.get(metric) is not None]
            approach_summary[metric] = round(sum(values) / len(values), 2) if values else None
        approach_summary["avg_input_tokens"] = round(
            sum(r.get("input_tokens", 0) for r in recs) / len(recs), 1
        )
        approach_summary["avg_output_tokens"] = round(
            sum(r.get("output_tokens", 0) for r in recs) / len(recs), 1
        )
        approach_summary["avg_cost"] = round(sum(r.get("cost", 0) for r in recs) / len(recs), 6)
        approach_summary["avg_latency_seconds"] = round(
            sum(r.get("latency_seconds", 0) for r in recs) / len(recs), 2
        )
        summary[approach] = approach_summary

    return summary


def print_summary(path: str = None):
    summary = summarize(path)
    if not summary:
        print("No eval records found yet -- run evaluate_record() on some questions first.")
        return
    for approach, s in summary.items():
        print(f"\n=== {approach} (n={s['n']}) ===")
        print(f"  context_relevance : {s['context_relevance']}")
        print(f"  faithfulness      : {s['faithfulness']}")
        print(f"  answer_relevance  : {s['answer_relevance']}")
        print(f"  correctness       : {s['correctness']}")
        print(f"  avg input tokens  : {s['avg_input_tokens']}")
        print(f"  avg output tokens : {s['avg_output_tokens']}")
        print(f"  avg cost          : ${s['avg_cost']}")
        print(f"  avg latency (s)   : {s['avg_latency_seconds']}")


if __name__ == "__main__":
    print_summary()
