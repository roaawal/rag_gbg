"""
Shared "final LLM call" logic.

Whichever route the router sends a question down (simple_llm_direct, rag,
or advanced_rag), it ends here: one grounded-answer prompt template, one
LLM call function. This is deliberate -- the grounding rules (use only
the supplied context, don't invent facts, say so if the context is
insufficient, cite chunk/page ids) are exactly the kind of thing that
quietly drifts out of sync if every route builds its own prompt. Define
them once, reuse everywhere.

rag.py keeps thin backward-compatible aliases (_build_prompt, _generate,
_estimate_tokens) pointing at this module, so existing code that imports
those names (basic_rag.py) keeps working unmodified.
"""
import os
import re
import time
from contextvars import ContextVar

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import requests

import config

_call_ledger = ContextVar("rag_call_ledger", default=None)


def start_call_tracking():
    """Start a per-question ledger and return its context token."""
    return _call_ledger.set([])


def stop_call_tracking(token):
    """Restore the previous ledger context and return this question's calls."""
    calls = list(_call_ledger.get() or [])
    _call_ledger.reset(token)
    return calls


def _record_call(stage: str, model: str, input_tokens: int, output_tokens: int,
                latency_seconds: float):
    ledger = _call_ledger.get()
    if ledger is None:
        return
    cost = (
        (input_tokens / 1000) * config.COST_PER_1K_INPUT_TOKENS
        + (output_tokens / 1000) * config.COST_PER_1K_OUTPUT_TOKENS
    )
    ledger.append({
        "stage": stage,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": cost,
        "latency_seconds": latency_seconds,
        "executed": True,
    })


def _base_url_for(model: str) -> str:
    """
    Which server hosts this model. Different local-model servers (LM
    Studio, Ollama, ...) can each host a different subset of the three
    roles (generation/router/judge) -- this is what lets the router and
    judge run on Ollama (serving Qwen) while generation stays on LM
    Studio (serving Jais), without every call site needing to know or
    care which backend a given model actually lives on.
    Falls back to LM Studio's URL for any model not explicitly mapped,
    which keeps this backward-compatible with a single-server setup.
    """
    return config.MODEL_BASE_URLS.get(model, config.LM_STUDIO_BASE_URL)


def call_completion(messages: list, max_tokens: int, temperature: float,
                    stage: str, model: str = None, timeout: int = 180) -> dict:
    """Call whichever local OpenAI-compatible server hosts `model` and
    track this LLM call."""
    model = model or config.LM_STUDIO_MODEL
    base_url = _base_url_for(model)
    start = time.perf_counter()
    response = requests.post(
        f"{base_url}/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=timeout,
    )
    latency_seconds = time.perf_counter() - start
    if not response.ok:
        detail = response.text.strip()
        raise RuntimeError(
            f"The server at {base_url} rejected the {stage} request with HTTP "
            f"{response.status_code} for model {model!r}. Server response: {detail[:1000]}"
        )
    data = response.json()
    content = data["choices"][0]["message"]["content"].strip()
    usage = data.get("usage") or {}
    input_tokens = usage.get("prompt_tokens") or estimate_tokens(
        "\n".join(str(m.get("content", "")) for m in messages)
    )
    output_tokens = usage.get("completion_tokens") or estimate_tokens(content)
    _record_call(stage, model, input_tokens, output_tokens, latency_seconds)
    return {
        "data": data,
        "text": content,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_seconds": latency_seconds,
    }


def is_arabic(text: str) -> bool:
    """True if the text contains Arabic script characters."""
    return bool(re.search(r"[\u0600-\u06FF]", text))


def estimate_tokens(text: str) -> int:
    """
    Rough fallback token estimate when the server doesn't report usage.
    Not exact -- Arabic and English tokenize differently -- but good
    enough for relative cost/size comparisons across approaches.
    """
    return max(1, len(text.split()))


def build_grounded_prompt(question: str, hits: list) -> str:
    """
    The RAG-route final prompt. Requirements this MUST keep enforcing,
    regardless of which route or which advanced techniques ran upstream:
      - use ONLY the supplied context for document-grounded claims
      - answer the user's actual question
      - never invent facts not supported by the context
      - explicitly say so if the context can't fully support an answer
      - surface source/chunk identifiers when available
    """
    context_blocks = []
    for h in hits:
        section_label = f", Section: {h['section']}" if h.get("section") else ""
        context_blocks.append(
            f"[Chunk ID: {h['id']} | Source: {h['source']}, page {h['page']}{section_label}]\n{h['text']}"
        )
    context = "\n\n---\n\n".join(context_blocks) if context_blocks else "(No context was retrieved.)"

    if is_arabic(question):
        instruction = """أجب على السؤال باستخدام المعلومات الواردة في المقتطفات أدناه فقط.

قواعد صارمة يجب اتباعها:
- استخدم فقط المعلومات المرفقة في السياق أدناه لأي ادعاء متعلق بالوثيقة، ولا تستخدم معرفة خارجية.
- أجب على السؤال الفعلي الذي طرحه المستخدم.
- لا تختلق أي معلومة غير مدعومة بالسياق.
- إذا كان السياق لا يدعم إجابة كاملة، اذكر ذلك بوضوح صراحةً بدلاً من التخمين.
- أجب باللغة العربية الفصحى فقط من أول كلمة إلى آخر كلمة. ممنوع منعًا باتًا استخدام أي كلمة أو حرف باللغة الإنجليزية.
- تجاهل تمامًا أي معلومات إدارية أو شكلية من رأس أو تذييل الوثيقة (رقم الإصدار، تاريخ الإصدار، "للاستخدام الداخلي").
- إذا كان المحتوى يصف إجراءً أو خطوات، اذكر كل خطوة مرقّمة بالترتيب، مع التفاصيل المهمة (من المسؤول، ما هي النماذج أو السجلات المطلوبة، التوقيت). لا تلخص في جملة واحدة قصيرة.
- في نهاية الإجابة، اذكر معرّف المقتطف/رقم الصفحة المصدر (مثال: المصدر: صفحة 8، Chunk ID: ...)."""
    else:
        instruction = """Using ONLY the context excerpts below, give a complete, detailed \
    answer to the question. Treat the context as the complete source of truth: do not use \
    outside knowledge, typical industry practices, or facts remembered from training. Do not \
    introduce a domain term, abbreviation, process, role, or explanation unless it is directly \
    supported by the context. If the context does not fully support an answer, explicitly say \
    there is insufficient information rather than guessing. Do not summarize into a single short \
    sentence -- if the context describes a procedure, list every step in order, numbered, with \
    any relevant detail (who does it, what forms/records are involved, timing). Ignore \
    administrative letterhead/boilerplate (issue dates, revision numbers, "Internal Use" labels). \
    At the end, mention the source chunk id(s) and page number(s) the answer was drawn from."""

    return f"""{instruction}

Context:
{context}

Question: {question}

Detailed answer:"""


def build_direct_prompt(question: str) -> str:
    """
    The simple_llm_direct route's prompt: no document context at all. Used
    for genuinely general/conversational questions the router decided
    don't need the corpus. Still answers in the question's language and
    stays honest about not consulting any document.
    """
    if is_arabic(question):
        instruction = ("أجب على السؤال التالي بإيجاز ووضوح باللغة العربية الفصحى فقط. "
                        "هذا سؤال عام لا يتطلب الرجوع إلى وثائق الشركة.")
    else:
        instruction = ("Answer the following question directly and concisely. "
                        "This is a general question that does not require consulting "
                        "any company documents.")
    return f"{instruction}\n\nQuestion: {question}\n\nAnswer:"


def call_llm(prompt: str, arabic: bool, system_message: str = None,
             max_tokens: int = None, temperature: float = None,
             stage: str = "final_generation") -> dict:
    """
    Returns {"text": answer, "input_tokens": int, "output_tokens": int,
    "latency_seconds": float}.
    """
    if system_message is None:
        if arabic:
            system_message = ("أنت مساعد دقيق يجيب فقط بناءً على السياق المُعطى عند توفره. "
                               "تجيب دائمًا باللغة العربية الفصحى فقط، دون أي كلمات إنجليزية، "
                               "وتقدّم إجابات كاملة ومفصّلة تعدّد كل خطوة من خطوات أي إجراء بدلاً "
                               "من تلخيصه في جملة واحدة، وتتجاهل أي معلومات شكلية من رأس أو تذييل "
                               "الوثيقة. لا تضف أي مصطلح أو اختصار أو معلومة غير موجودة في السياق، "
                               "ولا تستخدم معرفة عامة أو معرفة سابقة لسد أي فجوة؛ اذكر صراحةً أن "
                               "المعلومات غير كافية عند الحاجة.")
        else:
            system_message = ("You are a meticulous assistant that answers questions "
                               "grounded in any provided context. You always give "
                               "complete, detailed answers -- enumerating every step "
                               "of a procedure rather than summarizing it into one "
                               "sentence -- in the same language the question was "
                               "asked, and you ignore administrative letterhead or "
                               "boilerplate from any source document. Never add a "
                               "domain term, abbreviation, process, role, or fact that "
                               "is not directly supported by the supplied context; do "
                               "not use general knowledge to fill gaps, and explicitly "
                               "say when the context is insufficient.")

    result = call_completion(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens or config.MAX_NEW_TOKENS,
        temperature=config.TEMPERATURE if temperature is None else temperature,
        stage=stage,
    )

    return {
        "text": result["text"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "latency_seconds": result["latency_seconds"],
    }


def connection_error_message() -> str:
    return (
        "Couldn't reach a local model server. This pipeline can call more than "
        "one: make sure LM Studio is open with config.LM_STUDIO_MODEL loaded and "
        "its Local Server started (Developer tab -> Start Server) for "
        "generation, AND, if config.ROUTER_MODEL/config.JUDGE_MODEL are mapped "
        "to Ollama in config.MODEL_BASE_URLS, that `ollama serve` is running and "
        "those models have been pulled (`ollama pull <model>`)."
    )
