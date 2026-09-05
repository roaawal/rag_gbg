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
# load failures or crashes -- this is an architecture-support problem,
# not a hardware-size problem (a 2.7B model is tiny for this GPU).
#
# Qwen2.5-Instruct is a mainstream, well-supported architecture with
# strong Arabic capability in practice, and was tried as an alternative
# to Jais here (see the note above about Jais's non-standard ALiBi
# architecture and llama.cpp/GGUF compatibility risk). Kept as a
# commented fallback in case Jais gives you trouble again -- if it
# crashes, this is the first thing to switch to.
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_MODEL = "jais-family-2p7b-chat"  # main answer-generation model
# If you want the router and judge to use different models than the
# answer-generation model, load those model names in LM Studio and set
# the values below to match exactly.
# LM_STUDIO_MODEL = "qwen2.5-3b-instruct"      # alternate generation model
# LM_STUDIO_MODEL = "qwen2.5-7b-instruct"      # heavier generation model
MAX_NEW_TOKENS = 800
TEMPERATURE = 0.2

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
# Which model answers the judge prompts. Keep this independent from the
# generation model to avoid using the same LLM for both answer generation
# and evaluation. If you do not have a separate judge model loaded in LM
# Studio, set this to the same model name temporarily; otherwise point it at
# a stronger model you have available.
JUDGE_MODEL = "qwen2.5-7b-instruct"
JUDGE_MAX_NEW_TOKENS = 400
JUDGE_TEMPERATURE = 0.0  # deterministic scoring, not creative generation

# --- Router ---
# Classifies each incoming question into a route BEFORE any retrieval
# happens. Keep this independent from the generation model so routing is
# not tied to the same model used to draft final answers. If you only have
# one local model loaded, set this to that model name temporarily.
ROUTER_MODEL = "qwen2.5-3b-instruct"
ROUTER_MAX_NEW_TOKENS = 300
ROUTER_TEMPERATURE = 0.0
VALID_ROUTES = {"simple_llm_direct", "rag", "advanced_rag"}
VALID_TECHNIQUES = {
    "rewriting", "multi_query", "decomposition", "hyde", "self_query",
    "reranking", "contextual_compression", "crag",
}

# --- Advanced RAG: query understanding & transformation ---
MULTI_QUERY_N = 3            # how many paraphrased query variants to generate
DECOMPOSITION_MAX_SUBQS = 4  # cap on sub-questions from decomposition
HYDE_MAX_NEW_TOKENS = 250    # hypothetical-answer length is meant to be short

# --- Advanced RAG: retrieval improvement ---
RERANK_CANDIDATE_POOL = 15   # how many fused hits go INTO the LLM reranker
RERANK_KEEP_TOP_K = 5        # how many the reranker keeps for generation
CRAG_MAX_CORRECTIONS = 1     # how many corrective re-retrieval attempts CRAG gets
CRAG_CORRECTED_POOL_MULTIPLIER = 2  # widen HYBRID_CANDIDATE_POOL by this factor on correction
