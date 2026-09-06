"""
Advanced RAG orchestration.

Applies the selected query-understanding techniques, then always runs
reranking, contextual compression, and CRAG. Order of operations:

  1. self_query      -> metadata filter + cleaned question
  2. rewriting        -> cleaned question, disambiguated
  3. decomposition OR multi_query -> a list of search queries
     (decomposition takes priority if the router picked both, since it
     means the question genuinely has separate information needs;
     multi_query is about phrasing the SAME need several ways)
  4. hyde              -> one extra hypothetical-passage query added to
     the search-query list
    5. hybrid retrieval, once per search query, fused across queries via RRF
    6. reranking          -> keep the LLM-rescored top few
    7. crag               -> grade retrieval quality, one corrective
     re-retrieval attempt if graded ambiguous/incorrect
    8. contextual_compression -> trim the final chunks to relevant sentences
  9. final grounded generation (generation.py -- same module every route uses)

If none of the five query-understanding techniques were selected, the
raw question is used as the single search query. The three retrieval
improvement stages above are mandatory for every advanced route.
"""
import time
import uuid

import requests

import config
import generation
import rag
import query_transform
import retrieval_improve


def _multi_query_retrieve(queries: list, where: dict, pool_top_k: int):
    """Runs hybrid retrieve() once per query string, then RRF-fuses the
    results across queries (on top of the dense+BM25 fusion retrieve()
    already does internally for each individual query)."""
    id_lists = []
    hit_by_id = {}
    for q in queries:
        hits = rag.retrieve(q, top_k=pool_top_k, where=where)
        id_lists.append([h["id"] for h in hits])
        for h in hits:
            hit_by_id.setdefault(h["id"], h)

    fused_scores = rag.rrf_fuse(id_lists)
    ranked_ids = sorted(fused_scores.keys(), key=lambda i: fused_scores[i], reverse=True)
    return [hit_by_id[i] for i in ranked_ids if i in hit_by_id]


def answer_advanced(question: str, techniques: list, top_k: int = None) -> dict:
    """
    Runs the advanced_rag route for one question, applying the selected
    query-understanding techniques plus the mandatory retrieval-improvement
    stages. Returns
    the same record schema as rag.answer_question_full()/basic_rag.answer(),
    plus advanced-route-specific fields: techniques_used, search_queries,
    self_query_filter, crag_grade.
    """
    top_k = top_k or config.TOP_K
    total_start = time.perf_counter()
    # These stages define the advanced route's retrieval contract. The
    # router only chooses the optional query-understanding techniques.
    techniques = set(techniques) | {
        "reranking",
        "contextual_compression",
        "crag",
    }

    # 1. self_query
    where_filter = None
    cleaned_question = question
    if "self_query" in techniques:
        where_filter, cleaned_question = query_transform.self_query(question)

    # 2. rewriting
    if "rewriting" in techniques:
        cleaned_question = query_transform.rewrite_query(cleaned_question)

    # 3. decomposition OR multi_query
    if "decomposition" in techniques:
        search_queries = query_transform.decompose_query(cleaned_question)
    elif "multi_query" in techniques:
        search_queries = query_transform.multi_query(cleaned_question)
    else:
        search_queries = [cleaned_question]

    # 4. hyde -- one extra hypothetical-passage query, based on the
    # (possibly rewritten) primary question, not per sub-question, to
    # keep the extra LLM call bounded regardless of how many search
    # queries decomposition/multi_query produced.
    if "hyde" in techniques:
        hyde_passage = query_transform.hyde_answer(cleaned_question)
        search_queries = search_queries + [hyde_passage]

    # 5. retrieval, fused across every search query
    retrieval_pool = config.RERANK_CANDIDATE_POOL
    hits = _multi_query_retrieve(search_queries, where_filter, retrieval_pool)

    # 6. reranking
    hits = retrieval_improve.rerank(cleaned_question, hits, keep_top_k=top_k)

    # 7. CRAG
    crag_grade = None
    crag_corrected = False
    if "crag" in techniques:
        def _corrective_retrieve():
            wider_pool = retrieval_pool * config.CRAG_CORRECTED_POOL_MULTIPLIER
            corrected = _multi_query_retrieve(search_queries, where_filter, wider_pool)
            corrected = retrieval_improve.rerank(cleaned_question, corrected, keep_top_k=top_k)
            return corrected

        crag_result = retrieval_improve.crag_evaluate_and_correct(
            cleaned_question, hits, _corrective_retrieve, top_k
        )
        hits = crag_result["hits"]
        crag_grade = crag_result["grade"]
        crag_corrected = crag_result["corrected"]

    # 8. Compress after CRAG so corrected hits follow the same final path.
    hits = retrieval_improve.contextual_compression(cleaned_question, hits)

    # 9. final grounded generation -- same shared module every route uses
    context = "\n\n---\n\n".join(h["text"] for h in hits)
    prompt = generation.build_grounded_prompt(cleaned_question, hits)
    try:
        result = generation.call_llm(prompt, arabic=generation.is_arabic(cleaned_question))
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
        "approach": "advanced_rag",
        "question": question,
        "cleaned_question": cleaned_question,
        "techniques_used": sorted(techniques),
        "search_queries": search_queries,
        "self_query_filter": where_filter,
        "crag_grade": crag_grade,
        "crag_corrected": crag_corrected,
        "chunk_ids": [h["id"] for h in hits],
        "retrieved_chunks": [
            {
                "id": h["id"],
                "text": h["text"],
                "source": h["source"],
                "page": h["page"],
                "section": h.get("section"),
                "score": h["score"],
            }
            for h in hits
        ],
        "context": context,
        "answer": result["text"],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": cost,
        "latency_seconds": result["latency_seconds"],
        "total_latency_seconds": time.perf_counter() - total_start,
    }
