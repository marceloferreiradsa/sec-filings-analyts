"""
index.py — Build the FAISS vector index from embedded chunks

Reads:  data/processed/embedded_chunks.json   (from embed.py)
        data/processed/embeddings.npy           (from embed.py)
Writes: data/index/index.faiss                  (FAISS binary — the search structure)
        data/index/chunks.json                   (chunk metadata — aligned with index)

WHAT THIS STEP DOES
  Takes the 6,396 embedding vectors produced by embed.py and organises
  them into a structure that answers the question "which vectors are most
  similar to this query vector?" in milliseconds.

  Without an index, finding the nearest neighbours in a set of N vectors
  requires comparing the query against every single vector — O(N) work.
  At 6,396 vectors this is already fast, but the index structure prepares
  the system to scale, and it makes the retrieval logic explicit.

INDEX TYPE: IndexFlatIP
  "Flat" = the index stores all vectors in a flat contiguous array and
  computes exact similarity. No approximation, no training needed.

  "IP" = inner product (dot product) as the similarity metric.

  WHY inner product rather than L2 distance:
    Our vectors have unit norm (confirmed in embed.py: mean norm = 1.0000).
    For unit-norm vectors, cosine similarity equals inner product:

        cosine_sim(a, b) = (a · b) / (|a| × |b|)
        when |a| = |b| = 1:  = a · b

    Cosine similarity measures the angle between vectors regardless of
    magnitude — it asks "do these two texts point in the same semantic
    direction?" A short passage and a long passage about the same topic
    score equally. L2 distance would penalise the shorter one.

  WHY exact (flat) rather than approximate (HNSW, IVF):
    Approximate methods trade accuracy for speed. The payoff is only
    meaningful at hundreds of thousands of vectors. At 6,396 vectors,
    IndexFlatIP returns results in under 5ms. No reason to accept
    approximation errors at this scale.

SAVE FORMAT
  We save the raw FAISS index binary (index.faiss) and the chunk metadata
  (chunks.json) as two aligned files — chunks.json[i] describes the
  vector at row i of the FAISS index. retrieve.py loads both together.
  This format is transparent: you can inspect chunks.json directly and
  understand exactly what is stored at each index position.

TEST QUERY
  After building the index, we run a live end-to-end test:
  question → embed → search → retrieve → display.
  This is the first moment you can see semantic retrieval working.
  One small API call (one question embedding).

Usage:
    python index.py
    python index.py --no-test    # skip the test query
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import faiss
from openai import OpenAI


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHUNKS_PATH  = Path("./data/processed/embedded_chunks.json")
VECTORS_PATH = Path("./data/processed/embeddings.npy")
INDEX_DIR    = Path("./data/index")

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM   = 1536

TEST_QUERY = "What drove NVIDIA's revenue growth?"
TOP_K_TEST = 5


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_inputs(
    chunks_path: Path,
    vectors_path: Path,
) -> tuple[list[dict], np.ndarray]:

    print(f"\n[LOAD] Reading chunks: {chunks_path.resolve()}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Not found: {chunks_path} — run embed.py first.")

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    print(f"[LOAD] {len(chunks):,} chunks loaded.")

    print(f"\n[LOAD] Reading vectors: {vectors_path.resolve()}")
    if not vectors_path.exists():
        raise FileNotFoundError(f"Not found: {vectors_path} — run embed.py first.")

    vectors = np.load(vectors_path)
    print(f"[LOAD] Vectors shape: {vectors.shape}  dtype: {vectors.dtype}")

    return chunks, vectors


def verify_alignment(chunks: list[dict], vectors: np.ndarray) -> None:
    """
    Confirm that chunks[i] corresponds to vectors[i].
    embed.py establishes this contract; we verify it here.
    """
    print(f"\n[VERIFY] Checking chunk-vector alignment...")

    if len(chunks) != vectors.shape[0]:
        raise ValueError(
            f"Misalignment detected:\n"
            f"  {len(chunks):,} chunks in embedded_chunks.json\n"
            f"  {vectors.shape[0]:,} rows in embeddings.npy\n"
            f"Re-run embed.py to regenerate both files from the same source."
        )

    if vectors.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Wrong embedding dimension: expected {EMBEDDING_DIM}, "
            f"got {vectors.shape[1]}."
        )

    norms = np.linalg.norm(vectors, axis=1)
    max_deviation = float(np.abs(norms - 1.0).max())

    print(f"  [OK] {len(chunks):,} chunks aligned with {vectors.shape[0]:,} vectors.")
    print(f"  [OK] Embedding dimension: {EMBEDDING_DIM}")
    print(
        f"  [OK] Norms — "
        f"min: {norms.min():.6f}  "
        f"max: {norms.max():.6f}  "
        f"mean: {norms.mean():.6f}  "
        f"max deviation from 1.0: {max_deviation:.6f}"
    )

    if max_deviation > 0.01:
        print(
            f"  [WARNING] Norms deviate from 1.0 by more than 0.01.\n"
            f"  Inner product will not equal cosine similarity precisely."
        )


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------

def build_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build an IndexFlatIP from the embedding matrix.

    FAISS requires a C-contiguous float32 array. Our vectors are already
    float32 (set in embed.py), but we ensure contiguity with ascontiguousarray
    before adding — passing a non-contiguous view would silently produce
    wrong results.
    """
    print(f"\n[INDEX] Building FAISS IndexFlatIP...")
    print(f"        {vectors.shape[0]:,} vectors × {vectors.shape[1]} dimensions")
    print(f"        Metric: inner product (= cosine on unit-norm vectors)")
    print(f"        Type: exact search, no training, no approximation")

    t_start = time.time()

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    elapsed = time.time() - t_start

    print(f"\n[INDEX] Build complete in {elapsed:.3f}s")
    print(f"[INDEX] Vectors stored: {index.ntotal:,}")
    print(f"[INDEX] is_trained: {index.is_trained}  "
          f"(always True for IndexFlat — no training needed)")

    return index


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_index(
    index: faiss.IndexFlatIP,
    chunks: list[dict],
    index_dir: Path,
) -> None:
    """
    Save the FAISS index binary and the aligned chunk metadata.

    index.faiss  —  raw FAISS binary, read back with faiss.read_index()
    chunks.json  —  list of chunk dicts; chunks.json[i] describes index row i

    The two files must always be saved and loaded together. Saving them
    in the same directory and using them in the same step prevents
    accidental misalignment (e.g., re-embedding without re-indexing).
    """
    index_dir.mkdir(parents=True, exist_ok=True)

    faiss_path  = index_dir / "index.faiss"
    chunks_path = index_dir / "chunks.json"

    print(f"\n[SAVE] Writing FAISS index → {faiss_path}")
    faiss.write_index(index, str(faiss_path))
    print(f"       {faiss_path.stat().st_size / 1024 / 1024:.1f} MB")

    print(f"[SAVE] Writing chunk metadata → {chunks_path}")
    chunks_path.write_text(
        json.dumps(chunks, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"       {chunks_path.stat().st_size / 1024 / 1024:.1f} MB")


# ---------------------------------------------------------------------------
# Index statistics
# ---------------------------------------------------------------------------

def print_index_statistics(chunks: list[dict]) -> None:
    from collections import Counter

    by_company = Counter(c["metadata"]["company"] for c in chunks)
    by_source  = Counter(c["metadata"]["source_type"] for c in chunks)
    by_section = Counter(c["metadata"].get("section_name", "") for c in chunks)

    print(f"\n{'='*60}")
    print(f"INDEX STATISTICS")
    print(f"{'='*60}")
    print(f"\n  Total documents indexed: {len(chunks):,}")

    print(f"\n  By company (bar = share of index):")
    for company, count in sorted(by_company.items()):
        pct = 100 * count / len(chunks)
        bar = "█" * int(pct / 2)
        print(f"    {company:<8} {count:>5}  ({pct:>5.1f}%)  {bar}")

    print(f"\n  By source type:")
    for source, count in sorted(by_source.items()):
        print(f"    {source:<22} {count:>5}")

    print(f"\n  By section (top 5):")
    for section, count in by_section.most_common(5):
        print(f"    {section:<34} {count:>5}")

    print(f"\n  Filterable metadata fields:")
    filterable = [
        "company", "filing_type", "source_type",
        "section_name", "fiscal_year", "fiscal_period",
    ]
    for field in filterable:
        values = sorted(set(
            str(c["metadata"].get(field, ""))
            for c in chunks
            if c["metadata"].get(field) is not None
        ))
        if len(values) <= 10:
            print(f"    {field:<18} {values}")
        else:
            print(f"    {field:<18} {values[:4]} ... ({len(values)} unique values)")


# ---------------------------------------------------------------------------
# Test query
# ---------------------------------------------------------------------------

def run_test_query(
    index: faiss.IndexFlatIP,
    chunks: list[dict],
    client: OpenAI,
    query: str,
    top_k: int,
) -> None:
    """
    Embed a question and retrieve the top-k most similar chunks.

    This is the first live end-to-end test: question → embed → search.
    It reveals what the retriever will actually hand to the LLM before
    we build the retriever module.

    HOW FAISS SEARCH WORKS
      index.search(query_vector, k) returns two arrays of shape (1, k):
        scores    — inner product similarity values, highest = most similar
        positions — integer row positions in the index (0 to N-1)

      We use positions to look up the corresponding chunk in chunks[i].
      The score range for unit-norm vectors is [-1, 1]:
        1.0  = identical direction (same meaning)
        0.0  = orthogonal (unrelated)
       -1.0  = opposite direction (rare in practice)

    WHAT TO LOOK FOR IN THE RESULTS
      - Do the top results come from the right company given the query?
      - Are both financial_data and narrative chunks represented, or does
        one source type dominate? (Tells us about relative chunk density.)
      - Are scores clustered tightly or spread out? Tight clustering means
        the query is near many similar chunks; a large gap after rank 1-2
        means one result is much better than the rest.
    """
    print(f"\n{'='*60}")
    print(f"TEST QUERY")
    print(f"{'='*60}")
    print(f"\n  Query: \"{query}\"")
    print(f"  Top-k: {top_k}")

    print(f"\n[TEST] Embedding query...")
    t_start  = time.time()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query],
    )
    q_vector = np.array(
        response.data[0].embedding,
        dtype=np.float32,
    ).reshape(1, -1)   # FAISS expects shape (n_queries, dim)
    embed_elapsed = time.time() - t_start

    print(f"[TEST] Embedded in {embed_elapsed:.3f}s  "
          f"(norm: {np.linalg.norm(q_vector):.6f})")

    print(f"\n[TEST] Searching {index.ntotal:,} vectors...")
    t_search = time.time()
    scores, positions = index.search(q_vector, top_k)
    search_elapsed = time.time() - t_search

    print(f"[TEST] Search completed in {search_elapsed * 1000:.3f}ms\n")
    print(f"  Rank   Score   Company   Filing   Year  Period  "
          f"Section                   Content preview")
    print(f"  {'─'*100}")

    for rank, (score, pos) in enumerate(
        zip(scores[0], positions[0]), start=1
    ):
        chunk = chunks[int(pos)]
        meta  = chunk["metadata"]
        preview = chunk["page_content"][:90].replace("\n", " ").strip()

        print(
            f"  {rank:<5}  {score:.4f}  "
            f"{meta.get('company',''):<8}  "
            f"{meta.get('filing_type',''):<6}  "
            f"{str(meta.get('fiscal_year','')):<5} "
            f"{str(meta.get('fiscal_period','')):<5}  "
            f"{meta.get('section_name','')[:24]:<24}  "
            f"{preview}..."
        )

    print(f"\n  Score spread: "
          f"highest={scores[0][0]:.4f}  "
          f"lowest={scores[0][-1]:.4f}  "
          f"gap={scores[0][0]-scores[0][-1]:.4f}")
    print(
        f"\n  Note: scores are inner products on unit-norm vectors.\n"
        f"  They equal cosine similarity. Range is [-1, 1]; typical\n"
        f"  relevant results score 0.80-0.95 for this model."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index.")
    parser.add_argument(
        "--no-test",
        action="store_true",
        dest="no_test",
        help="Skip the test query at the end.",
    )
    args = parser.parse_args()

    t_start = time.time()

    print("\n" + "="*60)
    print("INDEXING PIPELINE")
    print("="*60)
    print(f"\n  Chunks:  {CHUNKS_PATH}")
    print(f"  Vectors: {VECTORS_PATH}")
    print(f"  Output:  {INDEX_DIR}/")
    print(f"  Index:   IndexFlatIP  (exact inner product, unit-norm vectors)")

    # ── Load ──────────────────────────────────────────────────────────
    chunks, vectors = load_inputs(CHUNKS_PATH, VECTORS_PATH)

    # ── Verify alignment ──────────────────────────────────────────────
    verify_alignment(chunks, vectors)

    # ── Build ─────────────────────────────────────────────────────────
    index = build_index(vectors)

    # ── Save ──────────────────────────────────────────────────────────
    save_index(index, chunks, INDEX_DIR)

    # ── Statistics ────────────────────────────────────────────────────
    print_index_statistics(chunks)

    # ── Test query ────────────────────────────────────────────────────
    if not args.no_test:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print(
                "\n[TEST] Skipping — OPENAI_API_KEY not set."
            )
        else:
            run_test_query(index, chunks, OpenAI(), TEST_QUERY, TOP_K_TEST)
    else:
        print(f"\n[TEST] Skipped (--no-test).")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Total elapsed: {elapsed:.1f}s")
    print(f"\n  Index ready at: {INDEX_DIR}/")
    print(f"  Load in retrieve.py with:")
    print(f"    index  = faiss.read_index('data/index/index.faiss')")
    print(f"    chunks = json.loads(open('data/index/chunks.json').read())")
    print(f"\n  Next step: run retrieve.py to build the retrieval module")
    print(f"  That module adds metadata filtering on top of vector search.")


if __name__ == "__main__":
    main()
