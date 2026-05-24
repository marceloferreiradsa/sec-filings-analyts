"""
embed.py — Generate embedding vectors for all chunks

Reads:  data/processed/chunks.json           (from chunk.py)
Writes: data/processed/embedded_chunks.json  (filtered chunks, one per row)
        data/processed/embeddings.npy         (float32 array, shape N × 1536)

The two output files are aligned: embedded_chunks.json[i] is described by
the vector at embeddings.npy[i]. The FAISS index step loads both together.

MODEL
  text-embedding-3-small
  Dimensions: 1,536   Max tokens: 8,191   Price: $0.020 / 1M tokens
  Our chunks average ~375 tokens — well within the model's context limit.
  We use the small model for the initial working version. The embed step
  is designed to be re-run with a different model (local BAAI/bge-large-en-v1.5
  later) by changing EMBEDDING_MODEL and EMBEDDING_DIM constants.

FILTERING
  Chunks under MIN_CHUNK_LENGTH characters are excluded before embedding.
  Identified in the chunk.py analysis: page-header fragments like
  "Apple Inc. | 2024 Form 10-K | 7" and section-ending sentence fragments
  carry too little semantic content to produce useful embeddings.
  Root cause fix (stripping running headers in html_parser.py) is a
  documented future improvement. The filter here is the pragmatic solution.

BATCHING
  We send BATCH_SIZE chunks per API request. OpenAI accepts up to 2,048
  inputs per request; we use a smaller batch so progress prints frequently
  and retries on failure are cheap (only one batch is lost, not thousands).

COST CONFIRMATION
  The script prints the estimated cost before making any API calls and asks
  for confirmation. The estimate uses 4 chars per token as a rough ratio.
  Actual cost is reported after completion using the API's usage field.

API KEY
  Reads OPENAI_API_KEY from the environment (set via Windows environment
  variables). The OpenAI client picks it up automatically — no .env file
  needed. The script aborts with a clear message if the key is not set.

Usage:
    python embed.py
    python embed.py --yes        # skip confirmation prompt
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from openai import OpenAI, RateLimitError, APIError

# ---------------------------------------------------------------------------
# Configuration — change EMBEDDING_MODEL and EMBEDDING_DIM here when
# switching to the local model in a later experiment.
# ---------------------------------------------------------------------------

INPUT_PATH   = Path("./data/processed/chunks.json")
CHUNKS_OUT   = Path("./data/processed/embedded_chunks.json")
VECTORS_OUT  = Path("./data/processed/embeddings.npy")

EMBEDDING_MODEL     = "text-embedding-3-small"
EMBEDDING_DIM       = 1536           # fixed for text-embedding-3-small
PRICE_PER_M_TOKENS  = 0.020          # USD per 1 million tokens (OpenAI, 2025)

BATCH_SIZE          = 200            # chunks per API request (max 2048)
MIN_CHUNK_LENGTH    = 200            # chars — shorter chunks are excluded

# Retry on transient API errors (rate limits, timeouts)
MAX_RETRIES         = 3
INITIAL_RETRY_DELAY = 2.0            # seconds; doubles on each retry


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_chunks(path: Path) -> list[dict]:
    print(f"\n[LOAD] Reading {path.resolve()}...")

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
            f"Run chunk.py first to generate it."
        )

    chunks = json.loads(path.read_text(encoding="utf-8"))
    print(f"[LOAD] {len(chunks):,} chunks loaded.")
    return chunks


def filter_chunks(
    chunks: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    Split chunks into (keep, exclude) based on minimum length.

    Excluded chunks are logged for transparency but not embedded.
    They are also not written to the output files, so the embedded_chunks
    and embeddings arrays stay in sync with each other automatically.
    """
    keep    = [c for c in chunks if len(c["page_content"]) >= MIN_CHUNK_LENGTH]
    exclude = [c for c in chunks if len(c["page_content"]) <  MIN_CHUNK_LENGTH]
    return keep, exclude


def print_filter_summary(
    keep: list[dict],
    exclude: list[dict],
) -> None:
    from collections import Counter
    total = len(keep) + len(exclude)

    print(f"\n[FILTER] Minimum chunk length: {MIN_CHUNK_LENGTH} chars")
    print(
        f"[FILTER] {len(exclude):,} / {total:,} chunks excluded "
        f"({100*len(exclude)/total:.1f}%)"
    )
    print(
        f"[FILTER] {len(keep):,} chunks will be embedded."
    )

    if exclude:
        # Show what was excluded by company and source type
        by_company = Counter(
            c["metadata"]["company"] for c in exclude
        )
        print(f"\n[FILTER] Excluded breakdown by company:")
        for company, count in sorted(by_company.items()):
            print(f"  {company:<8} {count:>4} short chunks excluded")

        print(f"\n[FILTER] Sample excluded content (first 3):")
        for c in exclude[:3]:
            meta = c["metadata"]
            print(
                f"  [{meta['company']} {meta['filing_type']} "
                f"{meta.get('section_name','')[:20]}]  "
                f"({len(c['page_content'])} chars)  "
                f"'{c['page_content'][:70].strip()}'"
            )


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

def estimate_cost(chunks: list[dict]) -> tuple[int, float]:
    """
    Estimate embedding cost before calling the API.

    Uses 4 chars per token as a rough approximation for English prose.
    Actual token counts from the API may differ by ±10-15%.
    The exact cost is reported after completion using response.usage.
    """
    total_chars  = sum(len(c["page_content"]) for c in chunks)
    est_tokens   = total_chars // 4
    est_cost_usd = (est_tokens / 1_000_000) * PRICE_PER_M_TOKENS
    return est_tokens, est_cost_usd


def print_cost_estimate(
    chunks: list[dict],
    est_tokens: int,
    est_cost_usd: float,
) -> None:
    n_batches    = -(-len(chunks) // BATCH_SIZE)   # ceiling division
    total_chars  = sum(len(c["page_content"]) for c in chunks)

    print(f"\n{'='*60}")
    print(f"EMBEDDING PLAN")
    print(f"{'='*60}")
    print(f"\n  Model:          {EMBEDDING_MODEL}")
    print(f"  Dimensions:     {EMBEDDING_DIM:,}")
    print(f"  Chunks:         {len(chunks):,}")
    print(f"  Total chars:    {total_chars:,}")
    print(f"  Est. tokens:    ~{est_tokens:,}  (at 4 chars/token)")
    print(f"  Est. cost:      ~${est_cost_usd:.4f} USD")
    print(f"  Batches:        {n_batches}  (batch size {BATCH_SIZE})")
    print(f"\n  Output → vectors: {VECTORS_OUT}")
    print(f"  Output → chunks:  {CHUNKS_OUT}")


def confirm_embedding(auto_yes: bool) -> bool:
    """Ask the user to confirm before spending money on the API."""
    if auto_yes:
        print("\n[CONFIRM] Auto-confirmed (--yes flag).")
        return True

    print()
    answer = input("  Proceed with embedding? [y/N]: ").strip().lower()
    if answer in ("y", "yes"):
        print("[CONFIRM] Proceeding.\n")
        return True
    else:
        print("[CONFIRM] Aborted. No API calls made.")
        return False


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_batch_with_retry(
    client: OpenAI,
    texts: list[str],
    batch_num: int,
    total_batches: int,
) -> tuple[list[list[float]], int]:
    """
    Embed a single batch of texts with retry on transient errors.

    Returns the list of embedding vectors and the actual token count
    reported by the API.

    WHY we print the token count per batch:
      The API's usage.total_tokens gives the exact count (including any
      BPE overhead). Summing these gives a more accurate cost than our
      character-based estimate.
    """
    delay = INITIAL_RETRY_DELAY

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts,
            )

            vectors = [item.embedding for item in response.data]
            tokens  = response.usage.total_tokens

            print(
                f"  [BATCH {batch_num:>4}/{total_batches}] "
                f"{len(texts):>3} chunks  "
                f"{tokens:>6,} tokens"
            )
            return vectors, tokens

        except RateLimitError as e:
            if attempt == MAX_RETRIES:
                raise
            print(
                f"  [BATCH {batch_num}] Rate limit hit "
                f"(attempt {attempt}/{MAX_RETRIES}). "
                f"Waiting {delay:.0f}s..."
            )
            time.sleep(delay)
            delay *= 2

        except APIError as e:
            if attempt == MAX_RETRIES:
                raise
            print(
                f"  [BATCH {batch_num}] API error: {e}. "
                f"Retrying in {delay:.0f}s..."
            )
            time.sleep(delay)
            delay *= 2

    # Should not reach here
    raise RuntimeError(f"Batch {batch_num} failed after {MAX_RETRIES} attempts.")


def embed_all_chunks(
    client: OpenAI,
    chunks: list[dict],
) -> tuple[np.ndarray, int]:
    """
    Embed all chunks in batches.

    Returns:
      vectors      numpy float32 array, shape (N, EMBEDDING_DIM)
      actual_tokens  exact token count from API usage fields

    WHY float32 instead of float64:
      OpenAI returns 64-bit floats, but cosine similarity in FAISS is
      computed on float32. Storing as float32 halves the file size with
      no meaningful precision loss for similarity search.
    """
    n_chunks       = len(chunks)
    n_batches      = -(-n_chunks // BATCH_SIZE)
    all_vectors    : list[list[float]] = []
    actual_tokens  = 0

    print(f"[EMBED] Starting {n_batches} batches...\n")
    t_start = time.time()

    for batch_num, start in enumerate(range(0, n_chunks, BATCH_SIZE), start=1):
        batch_chunks = chunks[start : start + BATCH_SIZE]
        batch_texts  = [c["page_content"] for c in batch_chunks]

        vectors, tokens = embed_batch_with_retry(
            client, batch_texts, batch_num, n_batches
        )

        all_vectors.extend(vectors)
        actual_tokens += tokens

        # Print a running total every 10 batches
        if batch_num % 10 == 0 or batch_num == n_batches:
            elapsed  = time.time() - t_start
            done_pct = 100 * len(all_vectors) / n_chunks
            rate     = len(all_vectors) / elapsed if elapsed > 0 else 0
            eta_s    = (n_chunks - len(all_vectors)) / rate if rate > 0 else 0
            print(
                f"\n  Progress: {len(all_vectors):,}/{n_chunks:,} "
                f"({done_pct:.0f}%)  |  "
                f"elapsed {elapsed:.0f}s  |  "
                f"ETA ~{eta_s:.0f}s\n"
            )

    vectors_array = np.array(all_vectors, dtype=np.float32)
    return vectors_array, actual_tokens


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_embeddings(
    vectors: np.ndarray,
    chunks: list[dict],
) -> int:
    """
    Sanity-check the embedding output. Returns the number of errors found.

    Checks:
      - Shape matches expected (N chunks × EMBEDDING_DIM dimensions)
      - No NaN or infinite values (would silently corrupt similarity search)
      - Vector norms are in a reasonable range (malformed requests sometimes
        produce near-zero vectors)
    """
    print(f"\n[VALIDATE] Checking embedding output...")
    errors = 0

    # Shape
    expected_shape = (len(chunks), EMBEDDING_DIM)
    if vectors.shape != expected_shape:
        print(
            f"  [ERROR] Shape mismatch: expected {expected_shape}, "
            f"got {vectors.shape}"
        )
        errors += 1
    else:
        print(f"  [OK] Shape correct: {vectors.shape}")

    # NaN / inf
    n_nan = int(np.isnan(vectors).sum())
    n_inf = int(np.isinf(vectors).sum())
    if n_nan > 0:
        print(f"  [ERROR] {n_nan} NaN values found in vectors.")
        errors += 1
    elif n_inf > 0:
        print(f"  [ERROR] {n_inf} infinite values found in vectors.")
        errors += 1
    else:
        print(f"  [OK] No NaN or infinite values.")

    # Vector norms
    norms       = np.linalg.norm(vectors, axis=1)
    near_zero   = int((norms < 0.01).sum())
    print(
        f"  [OK] Vector norms — "
        f"min: {norms.min():.4f}  "
        f"max: {norms.max():.4f}  "
        f"mean: {norms.mean():.4f}"
    )
    if near_zero > 0:
        print(
            f"  [WARNING] {near_zero} vectors have near-zero norm. "
            f"These chunks may have been empty or contain only whitespace."
        )

    return errors


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_results(
    chunks: list[dict],
    vectors: np.ndarray,
    chunks_path: Path,
    vectors_path: Path,
) -> None:
    """
    Save the two aligned output files.

    embedded_chunks.json  —  the filtered chunk dicts (JSON)
    embeddings.npy        —  float32 vector array

    The files are aligned: embedded_chunks.json[i] describes
    the vector at row i of embeddings.npy. The index.py step
    relies on this alignment.
    """
    chunks_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n[SAVE] Writing {len(chunks):,} chunk records → {chunks_path}...")
    chunks_path.write_text(
        json.dumps(chunks, indent=2, default=str),
        encoding="utf-8",
    )
    chunks_size_mb = chunks_path.stat().st_size / (1024 * 1024)
    print(f"       {chunks_size_mb:.1f} MB")

    print(f"[SAVE] Writing {vectors.shape} float32 array → {vectors_path}...")
    np.save(vectors_path, vectors)
    vectors_size_mb = vectors_path.stat().st_size / (1024 * 1024)
    print(f"       {vectors_size_mb:.1f} MB")

    print(
        f"\n[SAVE] Combined output: "
        f"{chunks_size_mb + vectors_size_mb:.1f} MB"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Embed chunks with OpenAI.")
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip the cost-confirmation prompt and proceed automatically.",
    )
    args = parser.parse_args()

    t_pipeline_start = time.time()

    print("\n" + "="*60)
    print("EMBEDDING PIPELINE")
    print("="*60)
    print(f"\n  Input:  {INPUT_PATH}")
    print(f"  Model:  {EMBEDDING_MODEL}  ({EMBEDDING_DIM} dimensions)")
    print(f"  Key:    OPENAI_API_KEY environment variable")

    # ── Verify API key ────────────────────────────────────────────────
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "\n[ERROR] OPENAI_API_KEY environment variable is not set.\n"
            "  On Windows, set it via:\n"
            "    System Properties → Environment Variables → New\n"
            "  Then restart your terminal and re-run this script."
        )
        return

    key_preview = api_key[:8] + "..." + api_key[-4:]
    print(f"  Key found: {key_preview}")

    client = OpenAI()   # picks up OPENAI_API_KEY automatically

    # ── Load and filter ───────────────────────────────────────────────
    chunks    = load_chunks(INPUT_PATH)
    keep, excl = filter_chunks(chunks)
    print_filter_summary(keep, excl)

    if not keep:
        print("[ERROR] No chunks remain after filtering. Aborting.")
        return

    # ── Cost estimate and confirmation ────────────────────────────────
    est_tokens, est_cost = estimate_cost(keep)
    print_cost_estimate(keep, est_tokens, est_cost)

    if not confirm_embedding(args.yes):
        return

    # ── Embed ─────────────────────────────────────────────────────────
    vectors, actual_tokens = embed_all_chunks(client, keep)
    actual_cost = (actual_tokens / 1_000_000) * PRICE_PER_M_TOKENS

    print(f"\n{'='*60}")
    print(f"EMBEDDING COMPLETE")
    print(f"{'='*60}")
    print(f"\n  Chunks embedded:  {len(keep):,}")
    print(f"  Actual tokens:    {actual_tokens:,}  (from API usage field)")
    print(f"  Actual cost:      ${actual_cost:.4f} USD")
    print(f"  Estimate error:   {abs(actual_tokens - est_tokens):,} tokens "
          f"({abs(actual_tokens-est_tokens)/actual_tokens*100:.1f}%)")

    # ── Validate ──────────────────────────────────────────────────────
    errors = validate_embeddings(vectors, keep)
    if errors:
        print(f"\n[ERROR] {errors} validation error(s). Output not saved.")
        return

    # ── Save ──────────────────────────────────────────────────────────
    save_results(keep, vectors, CHUNKS_OUT, VECTORS_OUT)

    elapsed = time.time() - t_pipeline_start
    print(f"\n  Total elapsed:  {elapsed:.1f}s")
    print(f"\n  Next step: run index.py to build the FAISS vector index")
    print(f"  That step loads both output files together and creates")
    print(f"  the searchable index used by the retrieval module.")


if __name__ == "__main__":
    main()
