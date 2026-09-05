"""
Text chunking with Arabic-aware sentence boundaries.

Arabic uses its own punctuation marks in addition to the Latin ones
(e.g. '؟' Arabic question mark, '،' Arabic comma). We split on sentence
enders first so we never cut a sentence in half, then pack sentences
into ~CHUNK_SIZE character windows with overlap.

Added after inspecting a real procedure manual: its actual procedure
steps are laid out as tables with a recurring row-label column
("الجهة المنفذة" / "المنفذ" -- "executing department" / "executor"),
and PyMuPDF flattens each table row onto its own line, with the label on
one line and its value on the next. Left alone, a chunk boundary can land
between the label and its value, so a retrieved chunk shows "المنفذ"
with no indication of *who* -- which defeats the "who does what" detail
the system prompt is meant to preserve. _glue_role_labels() joins a
label-only line to the line immediately after it before sentence
splitting ever runs.
"""
import re

# Sentence-ending punctuation: Latin . ! ? and Arabic ؟ ۔ plus newlines
_SENTENCE_END_RE = re.compile(r"(?<=[.!?؟۔])\s+|\n+")

# Table-of-contents pages are full of dot-leaders ("......... 8") connecting
# a heading to a page number -- real prose essentially never has these.
_DOT_LEADER_RE = re.compile(r"\.{4,}")

# Recurring procedure-table row labels observed in the sample manual. Kept
# as a short, explicit list rather than a broad heuristic (e.g. "any short
# line ending in a colon") to avoid accidentally gluing unrelated short
# lines together.
_ROLE_LABEL_LINES = {
    "الجهة املنفذة", "الجهة المنفذة",  # "executing department" (both with/without the alef-wasla PDF artifact)
    "املنفذ", "المنفذ",                  # "executor"
}


def _glue_role_labels(text: str) -> str:
    """
    Joins a line that is JUST a role-label (e.g. "المنفذ") to the
    non-empty line that follows it, so later sentence-splitting can't
    separate the label from its value.
    """
    lines = text.split("\n")
    glued = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped in _ROLE_LABEL_LINES:
            # look ahead for the next non-empty line to attach
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                glued.append(f"{stripped}: {lines[j].strip()}")
                i = j + 1
                continue
        glued.append(line)
        i += 1
    return "\n".join(glued)


def _looks_like_toc(text: str, min_dot_runs: int = 2) -> bool:
    """
    Heuristic: if a chunk contains multiple long runs of dots (dot-leaders),
    it's almost certainly a table-of-contents / index page rather than
    actual procedural content, and should be excluded from indexing so it
    never gets retrieved in place of the real answer.
    """
    return len(_DOT_LEADER_RE.findall(text)) >= min_dot_runs


def split_sentences(text: str):
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_END_RE.split(text)
    return [p.strip() for p in parts if p and p.strip()]


def chunk_text(text: str, chunk_size: int, overlap: int):
    """
    Packs sentences into chunks of roughly `chunk_size` characters,
    carrying `overlap` characters worth of trailing sentences into the
    next chunk so context isn't lost at boundaries.
    Returns a list of chunk strings.
    """
    text = _glue_role_labels(text)
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks = []
    current = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        if current_len + sentence_len > chunk_size and current:
            chunks.append(" ".join(current))

            overlap_sentences = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) > overlap:
                    break
                overlap_sentences.insert(0, s)
                overlap_len += len(s)

            current = overlap_sentences
            current_len = overlap_len

        current.append(sentence)
        current_len += sentence_len

    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_records(records, chunk_size: int, overlap: int):
    """
    Takes the page-level records from pdf_loader (optionally tagged with
    a "section" key -- see pdf_loader.section_for_page) and returns
    chunk-level records: {"source", "page", "chunk_id", "section", "text"}

    If a record has a "section" heading, it's prepended to each of its
    chunks' text before embedding -- this gives both the dense embedding
    and BM25 the section topic alongside the content, which helps match
    queries phrased close to the heading.

    Table-of-contents-like chunks are skipped (see _looks_like_toc) so
    they never get indexed and surface as false-positive retrieval hits.
    """
    chunked = []
    skipped_toc = 0
    for rec in records:
        section = rec.get("section")
        pieces = chunk_text(rec["text"], chunk_size, overlap)
        for i, piece in enumerate(pieces):
            if _looks_like_toc(piece):
                skipped_toc += 1
                continue
            text_for_storage = f"{section}\n{piece}" if section else piece
            chunked.append({
                "source": rec["source"],
                "page": rec["page"],
                "chunk_id": i,
                "section": section,
                "text": text_for_storage,
            })
    if skipped_toc:
        print(f"Skipped {skipped_toc} table-of-contents-like chunk(s) during indexing.")
    return chunked
