"""
Routes a question BEFORE any retrieval happens.

Output shape (always):
    {"route": "simple_llm_direct" | "rag" | "advanced_rag",
     "reason": "...",
     "techniques": ["..."]}

`techniques` is only meaningful for route == "advanced_rag": it's the
SUBSET of {rewriting, multi_query, decomposition, hyde, self_query,
reranking, contextual_compression, crag} this specific question actually
needs. The router is deliberately asked to pick as few as fit -- forcing
every question through every technique wastes latency/cost and, for
things like decomposition or HyDE, can actively hurt a question that
didn't need them (e.g. decomposing an already-atomic question invents
sub-questions that weren't asked).

Classification considers (per the task spec):
  - simple vs. complex wording
  - whether it needs the indexed documents at all
  - whether it bundles multiple distinct information needs
  - whether it would benefit from multiple search phrasings/angles
  - whether it carries a metadata constraint (a specific section, page,
    source file, date range, etc.)
  - whether the wording is vague/conversational and needs disambiguation
    before it's even a well-formed search query

The LLM is the primary classifier (it's the only thing that can actually
read intent), but a small local model won't always return clean JSON, so
_heuristic_classify() is a keyword/structure-based fallback that fires
whenever the LLM's output can't be parsed -- the router always returns
something usable, never raises.
"""
import json
import re

import requests

import config
import generation

_ROUTER_SYSTEM_PROMPT = """You are a routing classifier for a document question-answering system over an Arabic bank internal-procedures manual. For each user question, decide which of three routes it needs, considering:

- Is the question simple (a quick fact/definition/greeting) or complex (multi-part, needs reasoning)?
- Does answering it require information from the indexed procedure documents at all, or is it general/conversational (e.g. greetings, small talk, asking about your own capabilities, general knowledge unrelated to the bank's procedures)?
- Does it bundle MULTIPLE distinct information needs (e.g. "what is the annual inventory procedure AND who approves asset write-offs")?
- Would it benefit from searching with several different phrasings/angles because the wording is abstract or could be described several ways?
- Does it carry a METADATA constraint -- a specific section name, page number, source document, or date/version -- that should narrow the search rather than just being extra keywords?
- Is the wording vague, conversational, or dependent on unstated context (e.g. "what about the second one", "and if it's damaged?") such that it needs to be rewritten into a clear standalone query before searching?

Routes:
- "simple_llm_direct": no document retrieval needed at all (greetings, meta questions about the assistant, general knowledge unrelated to the manual, or a question already fully answered earlier in this exchange).
- "rag": needs the documents, but is a single, clear, well-formed, single-need question with no metadata constraint and no ambiguity -- plain hybrid retrieval + generation is enough.
- "advanced_rag": needs the documents AND has at least one complicating property above (multi-part, vague, needs multiple phrasings, has a metadata constraint, etc.) that plain retrieval would likely handle poorly.

If route is "advanced_rag", choose the SMALLEST set of techniques from this fixed list that actually addresses what's complicating this specific question -- do not include a technique "just in case":
- "rewriting": the wording is vague/conversational and needs to become a clear standalone query first.
- "multi_query": the question is abstract/could be phrased several different ways, and search recall would benefit from trying a few phrasings.
- "decomposition": the question bundles multiple distinct information needs that should be searched (and answered) somewhat separately.
- "hyde": the question is conceptual/asks "why" or "how does X work" in a way where a hypothetical answer passage would embed closer to the real answer than the bare question would.
- "self_query": the question names a specific section, page, source document, or date/version that should become a metadata filter.
- "reranking": the question is broad or ambiguous enough that initial hybrid retrieval is likely to surface some irrelevant chunks worth re-scoring.
- "contextual_compression": likely to retrieve chunks that are only partially relevant (long chunks with mixed content) where trimming to the relevant sentences would sharpen the final answer.
- "crag": there's real risk the corpus doesn't actually cover this question well and the system should double-check retrieval quality and self-correct rather than confidently answering from weak context.

If route is "rag" or "simple_llm_direct", techniques MUST be an empty list.

Respond with ONLY a single JSON object, no markdown fences, no extra text.
The value of "route" must be exactly ONE of these three strings: "simple_llm_direct", "rag", or "advanced_rag". Do not copy the pipe-separated list into the value.
The "reason" value is required and must be one short, non-empty sentence explaining the route choice.
Example format (choose one real route, do not output the placeholders):
{"route": "rag", "reason": "The question needs one document search.", "techniques": []}
"""


def _extract_json(text: str) -> dict:
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in router response: {text[:200]!r}")
    return json.loads(match.group(0))


def _heuristic_classify(question: str, fallback_reason: str = None) -> dict:
    """
    Non-LLM fallback used only when the router's LLM call fails outright
    (connection error) or returns something unparseable. Deliberately
    simple and conservative -- it leans toward "rag" (safe middle ground)
    rather than guessing at advanced techniques it can't justify from a
    quick heuristic.
    """
    q = question.strip()
    q_lower = q.lower()
    fallback_prefix = "Heuristic fallback"
    if fallback_reason:
        fallback_prefix += f" ({fallback_reason})"

    greeting_markers = ["hello", "hi ", "hey", "thanks", "thank you", "مرحبا", "شكرا", "أهلا", "السلام عليكم"]
    if len(q) < 25 and any(m in q_lower for m in greeting_markers):
        return {
            "route": "simple_llm_direct",
            "reason": f"{fallback_prefix}: short greeting/small-talk pattern, no document need detected.",
            "techniques": [],
        }

    if not q.endswith("?") and not q.endswith("؟") and len(q.split()) < 4:
        return {
            "route": "simple_llm_direct",
            "reason": f"{fallback_prefix}: very short, non-question fragment.",
            "techniques": [],
        }

    techniques = []
    metadata_markers = ["page", "صفحة", "section", "قسم", "الجزء", ".pdf", "version", "إصدار"]
    has_metadata = any(m in q_lower for m in metadata_markers)
    tokens = re.findall(r"[\w’']+", q_lower, re.UNICODE)
    interrogative_words = {
        "what", "who", "which", "when", "where", "why", "how",
        "ما", "ماذا", "من", "أي", "متى", "أين", "لماذا", "كيف", "هل",
    }
    interrogative_count = sum(
        token in interrogative_words
        or (token.startswith("و") and token[1:] in interrogative_words)
        for token in tokens
    )
    has_multi_need = interrogative_count >= 2
    is_long = len(q.split()) > 25

    if has_metadata:
        techniques.append("self_query")
    if has_multi_need or is_long:
        techniques.append("decomposition")

    if techniques:
        return {
            "route": "advanced_rag",
            "reason": f"{fallback_prefix}: detected multi-part wording and/or a metadata-like reference.",
            "techniques": techniques,
        }

    return {
        "route": "rag",
        "reason": f"{fallback_prefix}: looks like a single clear document question.",
        "techniques": [],
    }


def _looks_document_related(question: str) -> bool:
    """Recognize terms that indicate the question belongs to the indexed manual."""
    tokens = set(re.findall(r"[\w’']+", question.lower(), re.UNICODE))
    document_markers = {
        # Arabic terms found in, or strongly indicative of, procedure manuals.
        "وحدة", "الانذار", "الإنذار", "مركزي", "المركزي", "موجودات",
        "الموجودات", "ثابتة", "الجرد", "جرد", "إجراء", "إجراءات",
        "اعتماد", "اعتماد", "اعتمد", "شطب", "الأصناف", "مستودع",
        "مستودعات", "نموذج", "نماذج", "مسؤول", "المسؤول", "سياسة",
        "سياسات", "عملية", "عمليات", "البنك", "الفرع", "الفروع",
        # English equivalents for mixed-language questions.
        "procedure", "procedures", "inventory", "asset", "assets", "approval",
        "approve", "writeoff", "write-off", "warehouse", "branch", "manual",
    }
    return bool(tokens & document_markers)


def _validate(decision: dict, question: str) -> dict:
    """Clamps whatever the LLM returned to the actually-valid vocabulary,
    falling back to the heuristic if the structure is unusable."""
    if not isinstance(decision, dict):
        return _heuristic_classify(question, "classifier response was not an object")

    route = decision.get("route")
    if route not in config.VALID_ROUTES:
        return _heuristic_classify(question, f"invalid route {route!r}")

    techniques = decision.get("techniques") or []
    if not isinstance(techniques, list):
        techniques = []
    techniques = [
        technique
        for technique in techniques
        if isinstance(technique, str) and technique in config.VALID_TECHNIQUES
    ]

    if route != "advanced_rag":
        techniques = []
    elif not techniques:
        # advanced_rag was chosen but nothing survived validation -- fall
        # back to plain rag rather than running an "advanced" pipeline
        # with zero actual advanced techniques.
        route = "rag"

    reason = decision.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        reason = (
            f"Router model omitted a reason; using the validated {route!r} route."
        )

    if route == "simple_llm_direct" and _looks_document_related(question):
        return {
            "route": "rag",
            "reason": (
                f"{reason.strip()} Overridden: document-related terms were detected, "
                "so the indexed manual must be searched."
            ),
            "techniques": [],
        }

    return {"route": route, "reason": reason.strip(), "techniques": techniques}


def classify_route(question: str) -> dict:
    """
    The main entry point. Always returns a valid
    {"route", "reason", "techniques"} dict -- never raises, falling back
    to _heuristic_classify() on any LLM/parse failure.
    """
    raw_text = ""
    try:
        result = generation.call_completion(
            messages=[
                {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {question}"},
            ],
            max_tokens=config.ROUTER_MAX_NEW_TOKENS,
            temperature=config.ROUTER_TEMPERATURE,
            stage="router",
            model=config.ROUTER_MODEL,
            timeout=60,
        )
        raw_text = result["text"]
        decision = _extract_json(raw_text)
    except requests.exceptions.ConnectionError as exc:
        return _heuristic_classify(question, f"connection error: {exc}")
    except requests.exceptions.RequestException as exc:
        detail = f"request error: {exc}"
        if getattr(exc, "response", None) is not None:
            detail += f"; server response: {exc.response.text[:240]!r}"
        return _heuristic_classify(question, detail)
    except (ValueError, json.JSONDecodeError, KeyError, IndexError) as exc:
        detail = f"{type(exc).__name__}: {exc}"
        if raw_text:
            detail += f"; raw response: {raw_text[:240]!r}"
        return _heuristic_classify(question, detail)

    return _validate(decision, question)
