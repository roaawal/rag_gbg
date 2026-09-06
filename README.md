# Arabic Procedure Manuals RAG

Local Arabic document-question-answering system using Retrieval-Augmented Generation (RAG).
The project indexes internal procedure manuals, retrieves relevant passages, and generates
answers grounded in those passages. It supports both a command-line interface and a
Streamlit web interface.

## What the Project Does

The system answers questions about PDF procedure manuals covering:

- Central alarm and security procedures
- Fixed assets and warehouse operations
- Central mail and files procedures

The answer pipeline is designed to reduce hallucination by retrieving document evidence
before generation, showing source pages and chunk IDs, and explicitly reporting when the
retrieved context is insufficient.

## Architecture

```text
PDF files
   |
   v
PDF extraction and cleanup
   |
   v
Arabic-aware chunking
   |
   v
Embeddings stored in ChromaDB
   |
User question
   |
   v
Router: direct / rag / advanced_rag
   |
   +--> Direct generation
   |
   +--> Hybrid retrieval: dense search + BM25 + RRF
   |
   +--> Advanced retrieval: rewriting, decomposition, reranking,
        compression, and corrective retrieval
   |
   v
Grounded answer generation
   |
   v
Evaluation and JSONL logging
```

## Project Files

| File | Purpose |
|---|---|
| `chat.py` | Command-line and Streamlit user interface |
| `pipeline.py` | Main entry point and route orchestration |
| `config.py` | Central configuration for models, retrieval, and evaluation |
| `pdf_loader.py` | PDF extraction, Unicode normalization, cleanup, and section detection |
| `chunker.py` | Arabic-aware sentence splitting and chunk creation |
| `build_index.py` | Builds the persistent ChromaDB index |
| `rag.py` | Dense retrieval, BM25 retrieval, and Reciprocal Rank Fusion |
| `router.py` | Classifies questions and selects the processing route |
| `advanced_rag.py` | Runs the advanced RAG pipeline |
| `query_transform.py` | Rewriting, multi-query, decomposition, HyDE, and metadata filters |
| `retrieval_improve.py` | Reranking, contextual compression, and CRAG |
| `generation.py` | Local model calls, prompts, token tracking, and latency tracking |
| `evaluate.py` | LLM-as-judge evaluation |
| `reference_lookup.py` | Finds trusted answers for known evaluation questions |
| `logging_utils.py` | Reads and appends JSONL records |
| `reference_answers.json` | Reference answers for correctness evaluation |
| `data/` | Source PDF manuals |
| `chroma_store/` | Persistent local vector database |
| `logs/` | Per-question and evaluation logs |

## Requirements

- Python 3.10 or newer
- A project virtual environment
- LM Studio with its local server enabled
- Ollama with the configured router and judge models available
- Sufficient disk space for the embedding model and local language models

Python packages are listed in `requirements.txt`:

- PyMuPDF for PDF extraction
- sentence-transformers for embeddings
- ChromaDB for vector storage
- rank-bm25 for keyword retrieval
- requests for local model-server calls
- Streamlit for the web interface

## Installation

From PowerShell in the project directory:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation for the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

## Local Model Servers

The application calls OpenAI-compatible local HTTP endpoints.

### LM Studio

LM Studio serves the main generation model at:

```text
http://localhost:1234/v1
```

The current configuration expects:

```text
qwen2.5-3b-instruct
```

This model is used for final generation and the advanced-RAG helper stages by default:

- Rewriting
- Multi-query generation
- Question decomposition
- HyDE
- Reranking
- Contextual compression
- CRAG grading

Load the model in LM Studio and start its local server before asking questions.

### Ollama

Ollama is configured at:

```text
http://localhost:11434/v1
```

The router and evaluation judge use:

```powershell
ollama pull qwen2.5:3b
ollama pull qwen2.5:7b
ollama serve
```

The exact model IDs must match the names in `config.py`.

## Build the Index

Place PDF files in `data/`, then run:

```powershell
python build_index.py
```

Index construction performs these steps:

1. Extract text from every PDF page with PyMuPDF.
2. Normalize Arabic Unicode presentation forms.
3. Remove repeated headers and footers.
4. Detect possible source-text corruption and print warnings.
5. Detect section headings from tables of contents.
6. Split text into overlapping chunks.
7. Skip table-of-contents-like chunks.
8. Generate normalized `BAAI/bge-m3` embeddings.
9. Store text, embeddings, and metadata in ChromaDB.

The current chunk configuration is:

```text
Chunk size: 800 characters
Overlap: 150 characters
Collection: arabic_pdf_docs
```

Run the index build again whenever the PDFs change. The existing collection is replaced
when the index is rebuilt.

## Run the Application

### Interactive command line

```powershell
python chat.py
```

Then enter a question, for example:

```text
ما هي طريقة تدوير المخزون؟
```

Exit with `exit`, `quit`, `q`, or `Ctrl+C`.

### One question from the command line

```powershell
python chat.py "ما هي طريقة تدوير المخزون؟"
```

For reliable Arabic input in PowerShell, save the command or script as UTF-8. If Arabic
appears as question marks, the terminal encoding or the pasted command is not being
handled as UTF-8; the application itself configures standard output for UTF-8 where
supported.

### Streamlit interface

```powershell
streamlit run chat.py
```

The interface displays the answer, source passages, page numbers, route details, model
path, token usage, latency, cost, and evaluation values.

## Query Routes

### Direct route

Used for general or conversational questions that do not require the indexed manuals.
No document retrieval occurs.

### Basic RAG route

Used for a clear document question with one information need:

```text
router -> hybrid retrieval -> grounded generation -> evaluation
```

### Advanced RAG route

Used for vague, multi-part, conceptual, or metadata-constrained questions:

```text
router
  -> optional metadata filtering
  -> optional query rewriting
  -> decomposition or multi-query expansion
  -> optional HyDE query
  -> hybrid retrieval
  -> LLM reranking
  -> contextual compression
  -> CRAG grading and one corrective retrieval
  -> grounded generation
  -> evaluation
```

Generated rewrites, hypothetical passages, and search queries are used only to improve
retrieval. They are not passed to the final answer as document evidence.

## Retrieval Details

The normal retrieval mode combines two methods.

### Dense retrieval

The question and document chunks are represented as embeddings using `BAAI/bge-m3`.
This helps match paraphrases, synonyms, and semantically similar Arabic wording.

### BM25 keyword retrieval

BM25 finds exact word overlap. It is useful for names, codes, form titles, section headings,
dates, and exact Arabic terms.

### Reciprocal Rank Fusion

Dense and BM25 ranked lists are combined using Reciprocal Rank Fusion (RRF):

```text
RRF score = sum(1 / (k + rank + 1))
```

The current RRF constant is `60`. A result receives a stronger combined rank when it
appears near the top in both retrieval methods.

## Advanced Techniques

- **Query rewriting:** turns vague wording into a standalone search question.
- **Multi-query:** creates several phrasings of the same information need.
- **Decomposition:** splits a multi-part question into independent sub-questions.
- **HyDE:** creates a hypothetical passage to improve embedding-based search. It is never
  treated as evidence.
- **Self-query:** converts page, source, or section constraints into Chroma metadata filters.
- **Reranking:** asks a local model to score retrieved passages for direct relevance.
- **Contextual compression:** keeps only relevant original sentences from each passage.
- **CRAG:** grades retrieved context and retries retrieval once with a wider candidate pool
  when the evidence is ambiguous or incorrect.

## Grounding and Answer Behavior

The final prompt instructs the model to:

- Use only the supplied retrieved context for document claims.
- Answer the actual user question.
- Avoid inventing unsupported facts.
- State clearly when the context is insufficient.
- Preserve all important procedure steps.
- Include responsible roles, forms, records, and timing when available.
- Answer Arabic questions in Modern Standard Arabic.
- Ignore repeated administrative headers and footers.
- Mention source chunk IDs and page numbers.

This reduces hallucination but cannot guarantee perfect answers. The model can still
misunderstand corrupted source text or make an incorrect interpretation.

## Evaluation

Each answer is evaluated using four scores from 0 to 5:

| Metric | Meaning |
|---|---|
| Context relevance | Whether the retrieved passages contain useful evidence |
| Faithfulness | Whether answer claims are supported by the retrieved context |
| Answer relevance | Whether the answer directly addresses the question |
| Correctness | Whether the answer matches a trusted reference answer |

Correctness is scored only when the question matches an entry in `reference_answers.json`.
Other questions receive `None` for correctness because no trusted answer is available.

The evaluator is also a local LLM, so scores are useful for comparison and regression
detection but should not be treated as absolute ground truth.

To display aggregate historical results:

```powershell
python evaluate.py
```

## Logging

Records are stored as JSON Lines files:

- `logs/eval_runs.jsonl`: pipeline answers and evaluation results
- `logs/basic_rag_runs.jsonl`: baseline/basic-RAG records

Each record may include:

- Question and answer
- Selected route
- Router decision
- Retrieval chunks and scores
- Source file, page, and section
- Executed LLM stages
- Input and output tokens
- Estimated cost
- Latency
- Evaluation scores

The configured local cost is zero. Cost fields can still be used for comparison if paid
API pricing is configured later.

## Troubleshooting

### Index collection not found

Run:

```powershell
python build_index.py
```

### Cannot reach a local model server

Check that:

1. LM Studio is open and its local server is started.
2. The configured LM Studio model is loaded.
3. Ollama is running if router or judge calls use Ollama.
4. The model IDs in `config.py` exactly match the server model IDs.

### No PDFs found

Place at least one `.pdf` file inside `data/` and rebuild the index.

### Arabic text is corrupted

The loader can warn about several corruption patterns, but it cannot repair a broken
font mapping embedded in the original PDF. Obtain a cleaner PDF export or an OCR-based
version when possible.

### Answers are irrelevant

Check the following in order:

1. The source PDF contains extractable text.
2. The index was rebuilt after changing the PDFs.
3. The retrieved source pages are relevant.
4. The question is specific enough.
5. The router selected the correct route.
6. The model servers are using the intended models.

## Limitations

- The system depends on the quality of the source PDF text.
- Scanned image-only PDFs require OCR before useful indexing.
- Local model quality and hardware affect latency and answer quality.
- LLM-based routing, reranking, compression, and judging are not deterministic ground truth.
- Historical log records may have been produced with older model configurations.
- Rebuilding the index replaces the existing Chroma collection.
- CRAG retries local retrieval only; it does not search the internet.

## Quick Presentation Summary

This project is a local Arabic RAG system for querying internal procedure manuals. It cleans
and chunks PDF text, embeds the chunks with `BAAI/bge-m3`, and stores them in ChromaDB. A
router chooses between direct answering, basic hybrid RAG, and advanced RAG. Hybrid retrieval
combines semantic embeddings with BM25 keyword search through Reciprocal Rank Fusion. The
advanced pipeline can rewrite, decompose, rerank, compress, and correct retrieval results.
The final model answers only from retrieved document passages and returns source information.
Every request is evaluated and logged with retrieval, quality, token, cost, and latency data.
