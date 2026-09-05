"""
Run this once (and again any time your PDFs change) to build the vector index:

    python build_index.py

It reads every PDF in data/, chunks the text, embeds each chunk with
BAAI/bge-m3, and stores everything in a local persistent Chroma collection.

pdf_loader.load_pdfs_from_folder() now also strips repeated header/footer
boilerplate per document and prints a warning if a document's source text
looks corrupted (see pdf_loader.py's module docstring) -- both happen
automatically, nothing else here needs to change to benefit from them.
"""
import chromadb
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import config
from pdf_loader import load_pdfs_from_folder, extract_toc_sections, section_for_page
from chunker import chunk_records


def main():
    print(f"Loading PDFs from {config.DATA_DIR} ...")
    page_records = load_pdfs_from_folder(config.DATA_DIR)
    print(f"Loaded {len(page_records)} pages total.\n")

    print("Detecting section headings from each document's table of contents ...")
    records_by_source = defaultdict(list)
    for rec in page_records:
        records_by_source[rec["source"]].append(rec)

    tagged_records = []
    for source, recs in records_by_source.items():
        toc_entries = extract_toc_sections(recs)
        if toc_entries:
            print(f"  {source}: found {len(toc_entries)} section heading(s)")
        # extract_toc_sections already prints its own warning when it finds
        # zero entries for a document -- no separate handling needed here.
        for rec in recs:
            rec = dict(rec)
            rec["section"] = section_for_page(toc_entries, rec["page"])
            tagged_records.append(rec)
    print()

    print("Chunking text ...")
    chunks = chunk_records(tagged_records, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    print(f"Produced {len(chunks)} chunks.\n")

    if not chunks:
        print("No chunks produced — check that your PDFs contain extractable text "
              "(scanned/image-only PDFs need OCR first).")
        return

    print(f"Loading embedding model '{config.EMBED_MODEL_NAME}' (first run downloads it, ~2.2GB)...")
    model = SentenceTransformer(config.EMBED_MODEL_NAME)

    print("Embedding chunks ...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    print("\nWriting to Chroma ...")
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [f"{c['source']}::p{c['page']}::c{c['chunk_id']}" for c in chunks]
    metadatas = [
        {"source": c["source"], "page": c["page"], "section": c["section"] or ""}
        for c in chunks
    ]

    BATCH = 200
    for i in tqdm(range(0, len(ids), BATCH)):
        collection.add(
            ids=ids[i:i + BATCH],
            embeddings=embeddings[i:i + BATCH].tolist(),
            documents=texts[i:i + BATCH],
            metadatas=metadatas[i:i + BATCH],
        )

    print(f"\nDone. Indexed {len(ids)} chunks into '{config.COLLECTION_NAME}' at {config.CHROMA_DIR}")
    print("You can now run: python chat.py")


if __name__ == "__main__":
    main()
