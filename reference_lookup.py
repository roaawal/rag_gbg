"""
Ground-truth reference answers for a fixed set of known evaluation
questions (see reference_answers.json). pipeline.answer() calls
find_reference() on every incoming question; when it matches one of
these, the matched answer is passed to evaluate.evaluate_record() as
`reference`, so evaluate.py's `correctness` metric gets a real 0-5 score
instead of staying None. Any question that ISN'T in this set is scored
exactly as before (context_relevance/faithfulness/answer_relevance only,
correctness stays None) -- this only adds a reference when one actually
exists for the exact question asked.

Matching is intentionally forgiving of copy/paste noise (surrounding
whitespace, a trailing space before "؟", straight "?" vs Arabic "؟",
Arabic diacritics if present) but still requires the asked question to
be near-identical to a stored one -- this recognizes "the same
evaluation question, retyped", not "a different but similar-sounding
question", so it doesn't put words in ground truth's mouth for
questions that were never actually given a reference answer.
"""
import difflib
import json
import os
import re

_REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "reference_answers.json")

# Below this similarity ratio, treat the asked question as NOT the same
# question as any stored one -- chosen to absorb whitespace/punctuation
# drift from copy-pasting, not to guess across genuinely different
# questions. Raise this (closer to 1.0) if you ever see a false match;
# lower it only if legitimate retypings are being missed.
_FUZZY_MATCH_THRESHOLD = 0.90

_entries = None  # lazy-loaded, cached: [{"question", "answer", "_normalized"}]


def _normalize(text: str) -> str:
    """Collapse whitespace/punctuation noise that shouldn't affect
    whether two questions count as "the same question"."""
    text = text.strip()
    text = text.replace("?", "؟")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)  # strip Arabic diacritics (tashkeel), if present
    return text


def _load_entries():
    global _entries
    if _entries is not None:
        return _entries
    if not os.path.exists(_REFERENCE_PATH):
        _entries = []
        return _entries
    with open(_REFERENCE_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    _entries = [
        {
            "question": item["question"],
            "answer": item["answer"],
            "_normalized": _normalize(item["question"]),
        }
        for item in raw
    ]
    return _entries


def find_reference(question: str):
    """
    Returns the stored ground-truth answer string for `question` if it
    matches one of reference_answers.json's questions (exact match after
    normalization, or a close fuzzy match), else None.
    """
    entries = _load_entries()
    if not entries:
        return None

    normalized_q = _normalize(question)

    # Exact match after normalization first -- the common case for a
    # copy-pasted question, and avoids any fuzzy-match ambiguity.
    for entry in entries:
        if entry["_normalized"] == normalized_q:
            return entry["answer"]

    # Fuzzy fallback: catches retyped questions with minor differences
    # (a dropped/added word, different spacing) without matching across
    # genuinely different questions.
    best_ratio = 0.0
    best_answer = None
    for entry in entries:
        ratio = difflib.SequenceMatcher(None, normalized_q, entry["_normalized"]).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_answer = entry["answer"]

    if best_ratio >= _FUZZY_MATCH_THRESHOLD:
        return best_answer
    return None


def reload():
    """Force re-reading reference_answers.json on the next
    find_reference() call -- useful if you edit the file mid-session
    (e.g. in a long-running Streamlit process) without restarting."""
    global _entries
    _entries = None
