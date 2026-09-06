"""
Central configuration for the Arabic-friendly RAG pipeline.
Tweak these values instead of hunting through the other files.
"""
import os

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_store")

# --- Embedding model ---
EMBED_MODEL_NAME = "BAAI/bge-m3"

# --- Chunking ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# --- Retrieval ---
TOP_K = 4
COLLECTION_NAME = "arabic_pdf_docs"

# --- Hybrid retrieval (dense + BM25 keyword, fused via Reciprocal Rank
# Fusion) ---
# Procedure manuals lean heavily on exact terms (form names, codes,
# section headings) that pure semantic embeddings can under-weight --
# BM25 catches exact term overlap, dense catches paraphrases/synonyms.
# Combining both via RRF improves recall for this kind of document.
# No re-indexing required: BM25 is built from the same chunk text
# already stored in Chroma by build_index.py.
USE_HYBRID_RETRIEVAL = True
HYBRID_CANDIDATE_POOL = 20   # candidates each method contributes before fusion
RRF_K = 60                   # standard reciprocal-rank-fusion constant

# --- PDF cleanup (added after inspecting real scanned manuals) ---
# Corporate Arabic manuals like the Housing Bank procedures manual repeat
# an identical version/logo table (رقم النسخة / تاريخ الإصدار / اسم
# الاجراء ...) as extracted TEXT on every single page. Left alone, that
# boilerplate gets embedded into every chunk and dilutes both BM25 and
# dense similarity (every chunk starts to look alike). pdf_loader.py
# detects any line that repeats on at least this fraction of a document's
# pages and strips it before chunking. 0.5 caught 14/15 pages of the
# sample manual's header without touching real content (which naturally
# varies page to page).
HEADER_FOOTER_MIN_REPEAT_RATIO = 0.5

# --- Source-text quality check ---
# Some PDFs (confirmed on the Housing Bank sample: cross-checked with
# PyMuPDF, poppler's pdftotext, AND Tesseract OCR on the rendered page
# image -- all three agree) have common short Arabic words (من، على،
# التي) missing letters directly in the document's rendered glyphs, not
# as an extraction-tool bug. No extraction method fixes this -- it's
# baked into the source file. build_index.py can only flag it, not
# repair it: if you can get a cleaner re-export of the original file,
# do that; otherwise this is a known, accepted limitation (it mainly
# hits short connective words, not the domain nouns you actually search
# and retrieve on).
# If the ratio of "broken-looking" to "well-formed" instances of a set
# of common function words exceeds this, build_index.py prints a
# warning naming the affected source file.
TEXT_CORRUPTION_WARN_RATIO = 0.3

# Threshold for the SEPARATE mixed-script corruption check (see
# pdf_loader.estimate_mixed_script_corruption's docstring) -- a different
# corruption pattern (individual letters silently swapped for foreign-
# script codepoints, anywhere in the text, including domain nouns) that
# the function-word probe above can miss entirely. Lower threshold than
# TEXT_CORRUPTION_WARN_RATIO because even a small fraction of domain
# nouns getting corrupted this way meaningfully hurts retrieval (unlike
# the function-word pattern, which mostly hits connective words).
MIXED_SCRIPT_WARN_RATIO = 0.05

# --- Generation: via LM Studio's local server ---
# Given this machine's actual hardware (RTX 3050, 4GB VRAM; 16GB system
# RAM), running a model directly in Python (via transformers or raw
# llama-cpp-python) kept hitting memory/allocation walls. LM Studio
# already handles GGUF downloads and automatic GPU/CPU memory splitting
# far more robustly, and exposes a local OpenAI-compatible HTTP server --
# so generation now just calls that server instead of loading a model
# in this process at all.
#
# NOTE ON MODEL CHOICE: jais-family-2p7b-chat was tried first for its
# native Arabic tuning, but Jais uses a custom architecture (ALiBi
# position encoding instead of RoPE) that is NOT one of llama.cpp's
# core first-class architectures the way Llama/Qwen/Gemma/Mistral are.
# GGUF builds of it are community-converted, and mismatches between the
# GGUF and whatever llama.cpp build LM Studio ships tend to surface as
# load failures or crashes. Jais's context window is also a hard 2,048
# tokens (the length it was actually trained on), which is what caused
# the "request (6100 tokens) exceeds the available context size (2048
# tokens)" HTTP 400 -- a handful of retrieved chunks plus the grounding
# instructions routinely blew past that.
#
# Switched generation to Qwen2.5-3B-Instruct
# (lmstudio-community/Qwen2.5-3B-Instruct-GGUF): mainstream/well-
# supported architecture, strong Arabic capability in practice, and a
# 32k context window -- more than enough headroom for this pipeline's
# prompts that no pre-flight token budgeting/trimming is needed.
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_MODEL = "qwen2.5-3b-instruct"  # main answer-generation model -- lmstudio-community/Qwen2.5-3B-Instruct-GGUF, 32k context
# If you want the router and judge to use different models than the
# answer-generation model, load those model names in LM Studio and set
# the values below to match exactly.
# LM_STUDIO_MODEL = "jais-family-2p7b-chat"    # Arabic-native alternative -- NOTE: hard 2,048-token context, see above
# LM_STUDIO_MODEL = "qwen2.5-7b-instruct"      # heavier generation model

# --- Ollama (used for router + judge, since only Jais is loaded in LM
# Studio and LM Studio only serves one loaded model at a time) ---
# Ollama exposes the same OpenAI-compatible /v1/chat/completions shape LM
# Studio does, just on its own port, so it can run alongside LM Studio
# without the two fighting over the same model slot. Install Ollama, run
# `ollama pull qwen2.5:3b` and `ollama pull qwen2.5:7b` (Ollama serves
# Qwen2.5 directly -- no separate GGUF hunting needed), and `ollama serve`
# should already be running in the background after install (it installs
# as a background service on most platforms; run it manually if not).
OLLAMA_BASE_URL = "http://localhost:11434/v1"

# Maps each model name to the server that actually hosts it.
# generation.call_completion() looks a model up here to decide which
# server to call; any model NOT listed here falls back to
# LM_STUDIO_BASE_URL, so this only needs entries for models living
# somewhere other than the default LM Studio server.
MODEL_BASE_URLS = {
    "qwen2.5:3b": OLLAMA_BASE_URL,
    "qwen2.5:7b": OLLAMA_BASE_URL,
}

MAX_NEW_TOKENS = 800
TEMPERATURE = 0.2

# --- Per-stage model configuration ---
# Every LLM-driven stage of the pipeline gets its OWN configured model
# name instead of hardcoding one everywhere. All of them default to
# LM_STUDIO_MODEL (the only model actually loaded in LM Studio in the
# current single-GPU setup), so existing behavior is unchanged out of
# the box -- but each can be pointed at a different model (e.g. one of
# the Ollama-hosted Qwen models, or a future dedicated reranker/judge
# model) independently, and generation.call_completion() already knows
# how to route any model name to the right server via
# config.MODEL_BASE_URLS.
GENERATION_MODEL = LM_STUDIO_MODEL      # final grounded/direct answer
REWRITER_MODEL = LM_STUDIO_MODEL        # query rewriting
MULTI_QUERY_MODEL = LM_STUDIO_MODEL     # multi-query generation
DECOMPOSITION_MODEL = LM_STUDIO_MODEL   # question decomposition
HYDE_MODEL = LM_STUDIO_MODEL            # hypothetical-document generation
RERANKER_MODEL = LM_STUDIO_MODEL        # LLM-based reranking
COMPRESSION_MODEL = LM_STUDIO_MODEL     # contextual compression
CRAG_MODEL = LM_STUDIO_MODEL            # CRAG relevance grading

# --- Cost tracking ---
# Running fully locally via LM Studio costs $0 per token -- these are 0
# by default. Set them if you want to compare this pipeline's token
# usage against a paid API on an apples-to-apples cost basis (e.g. to
# answer "what would this have cost on a hosted model").
COST_PER_1K_INPUT_TOKENS = 0.0
COST_PER_1K_OUTPUT_TOKENS = 0.0

# --- Logging (per-question records for basic_rag.py / evaluate.py) ---
LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
BASIC_RAG_LOG = os.path.join(LOGS_DIR, "basic_rag_runs.jsonl")
EVAL_LOG = os.path.join(LOGS_DIR, "eval_runs.jsonl")

# --- LLM-as-judge evaluation ---
# Which model answers the judge prompts. Runs on Ollama (see
# MODEL_BASE_URLS above) so it's independent from both the generation
# model (Jais, on LM Studio) and doesn't need a second model loaded
# alongside Jais in LM Studio's single model slot.
JUDGE_MODEL = "qwen2.5:7b"
JUDGE_MAX_NEW_TOKENS = 400
JUDGE_TEMPERATURE = 0.0  # deterministic scoring, not creative generation

# --- Router ---
# Classifies each incoming question into a route BEFORE any retrieval
# happens. Also runs on Ollama, for the same reason as JUDGE_MODEL above.
ROUTER_MODEL = "qwen2.5:3b"
ROUTER_MAX_NEW_TOKENS = 300
ROUTER_TEMPERATURE = 0.0
VALID_ROUTES = {"direct", "rag", "advanced_rag"}
# The router only ever SELECTS from these four query-understanding
# techniques. self_query is intentionally not in this list: it isn't a
# question of "should we try a different phrasing", it's a metadata
# filter, so it's driven directly by the router's `has_metadata_constraints`
# boolean instead (see router.py / advanced_rag.py). reranking,
# contextual_compression, and crag are mandatory stages of the advanced
# route, not techniques the router chooses -- they're recorded in
# MANDATORY_ADVANCED_STAGES for logging/`techniques_used` purposes only.
VALID_TECHNIQUES = {"rewriting", "multi_query", "decomposition", "hyde"}
MANDATORY_ADVANCED_STAGES = {"reranking", "contextual_compression", "crag"}
VALID_COMPLEXITY = {"simple", "complex"}

# --- Advanced RAG: query understanding & transformation ---
MULTI_QUERY_N = 3            # how many paraphrased query variants to generate
DECOMPOSITION_MAX_SUBQS = 4  # cap on sub-questions from decomposition
HYDE_MAX_NEW_TOKENS = 250    # hypothetical-answer length is meant to be short

# --- Advanced RAG: retrieval improvement ---
RERANK_CANDIDATE_POOL = 15   # how many fused hits go INTO the LLM reranker
RERANK_KEEP_TOP_K = 5        # how many the reranker keeps for generation
CRAG_MAX_CORRECTIONS = 1     # how many corrective re-retrieval attempts CRAG gets
CRAG_CORRECTED_POOL_MULTIPLIER = 2  # widen HYBRID_CANDIDATE_POOL by this factor on correction
