"""
PDF text extraction.

Uses PyMuPDF (fitz), which handles Arabic text extraction correctly
(preserves logical character order regardless of the PDF's visual RTL
rendering) far more reliably than pdfminer/pypdf for RTL scripts.

Two things were added after inspecting a real scanned manual (a Housing
Bank Arabic procedures document):

1. Repeated header/footer stripping. Corporate manuals commonly repeat
   an identical version/logo table as extracted TEXT on every page --
   left in, it pollutes every chunk's embedding and BM25 tokens with
   identical boilerplate. See strip_repeated_boilerplate().

2. A source-text quality check. Some PDFs have letters missing from
   common short words directly in the document's rendered glyphs (not
   an extraction bug -- confirmed by cross-checking PyMuPDF, poppler's
   pdftotext, and Tesseract OCR on the rendered page image, which all
   agreed). No extraction method fixes this; estimate_text_corruption()
   only flags it so you know to look for a cleaner source file.
"""
import os
import re
import glob
from collections import Counter, defaultdict

import fitz  # PyMuPDF

import config

# ---------------------------------------------------------------------------
# TOC entry parsing
# ---------------------------------------------------------------------------
# Real extracted TOC text turned out to use TWO different entry layouts in
# the same document (apparently because some entries were edited later,
# independently of the rest of the TOC):
#
#   Layout A (most entries): "<page_num> <item_num>.<heading> ....dots...."
#       e.g. "8 1.القرطاسية ملستودعات السنوي الجرد إجراءات ............."
#       -- the page number comes BEFORE the item number/heading.
#
#   Layout B (a minority of entries): "<item_num>.<heading> ....dots.... <page_num>"
#       e.g. "13.إجراءات صرف معامالت شراء ... .............................. 23"
#       -- the page number comes AFTER the heading and dot-leaders, which
#       is what a "normal" TOC line looks like.
#
# Rather than assume one layout (the original single-pattern regex matched
# ZERO entries against the real document), we try both and merge results.
_TOC_ENTRY_RE_PAGE_FIRST = re.compile(
    r"(\d{1,3})\s+\d{1,2}[)\]]?\.?\s*([^\d]{3,150}?)\.{4,}"
)
_TOC_ENTRY_RE_PAGE_LAST = re.compile(
    r"\.?(\d{1,2})[)\]]?\.?\s*([^\d.]{3,150}?)(\d{1,3})[.\s]{4,}"
)


def extract_pages(pdf_path: str):
    """
    Yields (page_number, text) tuples for a single PDF, page_number is 1-indexed.
    """
    doc = fitz.open(pdf_path)
    try:
        for i, page in enumerate(doc):
            text = page.get_text("text")
            if text and text.strip():
                yield i + 1, text
    finally:
        doc.close()#(page_number, page_text)


def strip_repeated_boilerplate(records: list, min_repeat_ratio: float = None) -> list:
    """
    Detects lines that repeat across many pages of ONE document (running
    headers/footers: version tables, logos, page-footer text) and removes
    them from each page's text before it ever reaches the chunker.

    `records` should be the page records for a SINGLE source document
    (repeat ratios are meaningless mixed across different documents).

    Detection is digit-agnostic: a header line containing a page number
    ("15 /2") still matches across pages after normalizing out digits, so
    a header that changes only its page number is still caught.

    Returns a new list of records with the same shape (source/page/text),
    boilerplate lines removed from `text`.
    """
    min_repeat_ratio = min_repeat_ratio if min_repeat_ratio is not None else config.HEADER_FOOTER_MIN_REPEAT_RATIO
    if len(records) < 3:
        # Too few pages to reliably distinguish "repeated header" from
        # "coincidentally similar content" -- skip stripping.
        return records

    def _normalize(line: str) -> str:
        # Collapse digits and whitespace so "15/2" and "15/3" (page-number
        # footers) are recognized as the same recurring line.
        line = re.sub(r"\d+", "#", line)
        line = re.sub(r"\s+", " ", line).strip()
        return line

    def _is_too_generic(normalized_line: str) -> bool:
        # A line that's just a bare page number (e.g. "15" wrapped onto its
        # own line from "15/2") normalizes down to something like "#" or
        # "#/" -- nearly every page has SOME standalone number like this,
        # but it's a coincidence, not a shared boilerplate sentence. If we
        # don't exclude these, stripping them also deletes unrelated bare
        # numbers elsewhere in the document that happen to look the same
        # after normalization -- e.g. this exact bug deleted every
        # per-entry page number on the table-of-contents page, since a TOC
        # page number ("9", "10", "11"...) normalizes to the same "#" as
        # the recurring page-footer number. Require at least 3 real
        # (non-#, non-whitespace) characters before treating a line as
        # real repeated boilerplate.
        real_chars = re.sub(r"[#\s/\-.]", "", normalized_line)
        return len(real_chars) < 3

    line_page_counts = Counter()
    per_page_lines = []
    for rec in records:
        lines = [l for l in rec["text"].split("\n")]
        normalized_lines = {_normalize(l) for l in lines if _normalize(l)}
        per_page_lines.append(lines)
        line_page_counts.update(normalized_lines)

    n_pages = len(records)
    boilerplate_normalized = {
        norm for norm, count in line_page_counts.items()
        if norm and count / n_pages >= min_repeat_ratio and not _is_too_generic(norm)
    }

    cleaned = []
    for rec, lines in zip(records, per_page_lines):
        kept_lines = [l for l in lines if _normalize(l) not in boilerplate_normalized]
        cleaned_text = "\n".join(kept_lines)
        new_rec = dict(rec)
        new_rec["text"] = cleaned_text
        cleaned.append(new_rec)

    if boilerplate_normalized:
        print(f"  Stripped {len(boilerplate_normalized)} repeated header/footer line(s) "
              f"from {records[0]['source']} (seen on >= {int(min_repeat_ratio*100)}% of its pages).")

    return cleaned


# Common short Arabic function words that should appear frequently in any
# formal Arabic document, paired with the "letter(s) missing" form that
# shows up when a document's font/shaping is broken. Padded with spaces so
# we're matching whole words, not substrings of longer words.
_CORRUPTION_PROBE_PAIRS = [
    (" من ", " ن "),
    (" على ", " عل "),
    (" التي ", " التم "),
]


def estimate_text_corruption(full_text: str) -> float:
    """
    Returns a rough 0-1 "corruption score" for a document's extracted text:
    the fraction of (correct + broken) occurrences of a few common Arabic
    function words that came out in the broken form.

    This can't distinguish an extraction-tool bug from a bug baked into
    the source PDF itself -- verifying that took cross-checking with an
    independent extractor (poppler) and OCR on the rendered page image,
    which isn't practical to do automatically for every document. It also
    can't fix anything. It only tells you a document is worth a closer
    look (and, if you have access to it, worth re-exporting from a
    cleaner source) before you trust generated answers drawn from it.
    """
    total_correct = 0
    total_broken = 0
    padded = f" {full_text} "
    for correct_form, broken_form in _CORRUPTION_PROBE_PAIRS:
        total_correct += padded.count(correct_form)
        total_broken += padded.count(broken_form)

    total = total_correct + total_broken
    if total == 0:
        return 0.0
    return total_broken / total


def load_pdfs_from_folder(folder: str):
    """
    Loads every PDF in `folder`.
    Returns a list of dicts: {"source": filename, "page": page_num, "text": text}

    Each document's pages have repeated header/footer boilerplate stripped
    (see strip_repeated_boilerplate) and are checked for the source-text
    corruption pattern described in estimate_text_corruption, before being
    handed back for chunking.
    """
    pdf_paths = sorted(glob.glob(os.path.join(folder, "*.pdf")))
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files found in '{folder}'. Put your PDFs there and re-run."
        )

    records = []
    for path in pdf_paths:
        filename = os.path.basename(path)
        print(f"Reading {filename} ...")
        doc_records = []
        for page_num, text in extract_pages(path):
            doc_records.append({"source": filename, "page": page_num, "text": text})
        print(f"  -> extracted {len(doc_records)} pages with text")

        doc_records = strip_repeated_boilerplate(doc_records)

        full_text = "\n".join(r["text"] for r in doc_records)
        corruption_score = estimate_text_corruption(full_text)
        if corruption_score >= config.TEXT_CORRUPTION_WARN_RATIO:
            print(f"  WARNING: {filename} looks like it has corrupted source text "
                  f"({corruption_score:.0%} of sampled function words came out malformed). "
                  f"This was confirmed (on the original sample document) to be baked into "
                  f"the PDF's rendered glyphs, not an extraction-tool bug -- no re-extraction "
                  f"method fixes it. Domain nouns are usually unaffected, but generated answers "
                  f"quoting prose from this file may contain garbled connective words. Get a "
                  f"cleaner source export if possible.")

        records.extend(doc_records)

    return records


def extract_toc_sections(records):
    """
    Scans a single document's page records for table-of-contents-style
    entries (heading + dot-leaders + page number) and returns an ordered
    list of (start_page, heading) tuples describing which page each named
    section begins on.

    `records` should be the page records for ONE source document only
    (page numbers are per-document, so mixing sources would misattribute
    sections).

    Real PDF text extraction routinely splits a single visual TOC line
    across several extracted lines (each dot-leader run can land on its
    own line), so raw per-line matching misses most entries. Whitespace
    is normalized to single spaces first, which reliably reassembles
    each entry onto one line for matching, regardless of exactly how the
    underlying PDF broke it up.

    Tries two entry layouts (see the regex comments above _TOC_ENTRY_RE_*)
    and merges whatever each one catches, since real documents have been
    observed to mix both layouts within the same TOC.
    """
    entries = []
    for rec in records:
        normalized = re.sub(r"\s+", " ", rec["text"]).strip()
        # Collapse a stray space that sometimes appears between an item
        # number and its following period/bracket (an RTL-extraction
        # artifact) -- e.g. "4 .heading" -> "4.heading".
        normalized = re.sub(r"(\d)\s+([).\]])", r"\1\2", normalized)

        for match in _TOC_ENTRY_RE_PAGE_FIRST.finditer(normalized):
            page_str, heading = match.group(1), match.group(2)
            try:
                page_num = int(page_str)
            except ValueError:
                continue
            heading = heading.strip()
            if heading and page_num > 0:
                entries.append((page_num, heading))

        for match in _TOC_ENTRY_RE_PAGE_LAST.finditer(normalized):
            heading, page_str = match.group(2), match.group(3)
            try:
                page_num = int(page_str)
            except ValueError:
                continue
            heading = heading.strip()
            if heading and page_num > 0:
                entries.append((page_num, heading))

    # De-duplicate (a heading could theoretically match on more than one
    # scanned page, or get caught by both regexes) while keeping the first
    # occurrence per page number, then sort by page.
    seen_pages = set()
    unique_entries = []
    for page_num, heading in entries:
        if page_num not in seen_pages:
            seen_pages.add(page_num)
            unique_entries.append((page_num, heading))
    unique_entries.sort(key=lambda x: x[0])

    if not unique_entries and records:
        print(f"  NOTE: no table-of-contents entries detected for {records[0]['source']}. "
              f"Section tagging is disabled for this document (chunks will have "
              f"section=None) -- retrieval and generation still work, they just lose "
              f"the 'prepend the section heading' relevance boost.")

    return unique_entries


def section_for_page(toc_entries, page_num):
    """
    Given the (start_page, heading) list from extract_toc_sections,
    returns the heading whose section the given page falls under (the
    last TOC entry whose start_page <= page_num), or None if the page
    comes before the first section (e.g. cover page, intro, the TOC
    page itself).
    """
    current = None
    for start_page, heading in toc_entries:
        if start_page <= page_num:
            current = heading
        else:
            break
    return current
