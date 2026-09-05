"""
Basic RAG baseline -- the plain version of this pipeline, kept separate
from rag.py's hybrid/section-tagged pipeline so later improvements have
something fixed to compare against.

Flow: Question -> Embedding -> Vector Search (pure dense, no BM25) ->
Top-K Chunks -> Prompt -> LLM -> Answer.

Every call to answer() logs one JSONL record to config.BASIC_RAG_LOG with
everything needed to reproduce or compare the run later: the question,
retrieved chunk ids, the context actually sent to the model, the answer,
token counts, cost, and latency.
"""
import time
import uuid

import requests

import config
from rag import _dense_only_retrieve, _estimate_tokens
from logging_utils import append_record


# Deliberately simpler and more generic than rag.py's Arabic-forcing,
# procedure-formatting system message -- this is the "baseline" prompt,
# not the tuned one.
_BASIC_SYSTEM_MESSAGE = (
    "You are a document question-answering assistant. Follow these rules "
    "strictly:\n"
    "1. Use ONLY the supplied context for any claim about the documents -- "
    "never use outside knowledge to answer the question.\n"
    "2. Answer the user's actual question directly.\n"
    "3. Do not invent facts that are not supported by the context.\n"
    "4. If the context does not contain enough information to answer, "
    "say plainly that there is insufficient information -- do not guess.\n"
    "5. When you make a claim from the context, mention which chunk id "
    "and/or page it came from.\n"
    "Answer in the same language the question was asked in."
)


def _build_basic_prompt(question: str, hits: list) -> str:
    context_blocks = []
    for h in hits:
        context_blocks.append(
            f"[chunk_id: {h['id']} | source: {h['source']} | page: {h['page']}]\n{h['text']}"
        )
    context = "\n\n---\n\n".join(context_blocks)
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


def _call_llm(prompt: str) -> dict:
    start = time.perf_counter()
    response = requests.post(
        f"{config.LM_STUDIO_BASE_URL}/chat/completions",
        json={
            "model": config.LM_STUDIO_MODEL,
            "messages": [
                {"role": "system", "content": _BASIC_SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": config.MAX_NEW_TOKENS,
            "temperature": config.TEMPERATURE,
        },
        timeout=180,
    )
    latency_seconds = time.perf_counter() - start
    response.raise_for_status()
    data = response.json()
    text = data["choices"][0]["message"]["content"].strip()

    usage = data.get("usage") or {}
    input_tokens = usage.get("prompt_tokens") or _estimate_tokens(_BASIC_SYSTEM_MESSAGE + prompt)
    output_tokens = usage.get("completion_tokens") or _estimate_tokens(text)

    return {
        "text": text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_seconds": latency_seconds,
    }


def _cost(input_tokens: int, output_tokens: int) -> float:
    return (
        (input_tokens / 1000) * config.COST_PER_1K_INPUT_TOKENS
        + (output_tokens / 1000) * config.COST_PER_1K_OUTPUT_TOKENS
    )


def answer(question: str, top_k: int = None, log: bool = True) -> dict:
    """
    Runs the basic RAG flow for one question. Returns (and, by default,
    logs to config.BASIC_RAG_LOG) a full record:
        record_id, approach, question, chunk_ids, context, answer,
        input_tokens, output_tokens, cost, latency_seconds,
        total_latency_seconds (includes retrieval time, not just the LLM call)
    """
    top_k = top_k or config.TOP_K
    total_start = time.perf_counter()

    hits = _dense_only_retrieve(question, top_k)
    context = "\n\n---\n\n".join(h["text"] for h in hits)

    prompt = _build_basic_prompt(question, hits)
    try:
        result = _call_llm(prompt)
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Couldn't reach LM Studio's local server. Make sure LM Studio is "
            "open, the configured model is loaded, and the Local Server is "
            "started (Developer / Local Server tab -> Start Server)."
        )

    record = {
        "record_id": str(uuid.uuid4()),
        "approach": "basic_rag",
        "question": question,
        "chunk_ids": [h["id"] for h in hits],
        "context": context,
        "answer": result["text"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "cost": _cost(result["input_tokens"], result["output_tokens"]),
        "latency_seconds": result["latency_seconds"],
        "total_latency_seconds": time.perf_counter() - total_start,
    }

    if log:
        append_record(config.BASIC_RAG_LOG, record)

    return record


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        rec = answer(q)
        print(f"\nQ: {q}")
        print(f"A: {rec['answer']}")
        print(f"\n[chunks: {rec['chunk_ids']}]")
        print(f"[tokens in/out: {rec['input_tokens']}/{rec['output_tokens']}  "
              f"cost: ${rec['cost']:.6f}  latency: {rec['latency_seconds']:.2f}s]")
    else:
        print('Usage: python basic_rag.py "your question"')
