"""
ingest.py — Run the full document ingestion pipeline

Execute this once after downloading filings with sec-edgar-downloader.
Produces a list of LangChain Documents saved to data/processed/documents.json
for use by the chunking and embedding modules.

Usage:
    python ingest.py
"""

import json
from pathlib import Path

from document_builder import build_all_documents

RAW_DATA_PATH = Path("./data/raw")
OUTPUT_PATH   = Path("./data/processed/documents.json")


def main():
    print("Starting document ingestion pipeline...")
    print(f"Raw data path: {RAW_DATA_PATH.resolve()}")

    documents = build_all_documents(RAW_DATA_PATH)

    # Serialize to JSON for inspection and reuse
    # (avoids re-running the slow HTML parser on every dev iteration)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    serialized = [
        {
            "page_content": doc.page_content,
            "metadata":     doc.metadata,
        }
        for doc in documents
    ]

    OUTPUT_PATH.write_text(json.dumps(serialized, indent=2, default=str))

    print(f"\nSaved {len(documents)} documents to {OUTPUT_PATH}")
    print("\nNext step: run chunk.py to split documents into indexed chunks")


if __name__ == "__main__":
    main()
