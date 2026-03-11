"""
ServiceNow RAG on MongoDB Atlas — Runbook
==========================================
This file walks through every phase of the pipeline end to end:

  Phase 1  — Connect to MongoDB Atlas
  Phase 2  — Simulate ServiceNow record ingestion
  Phase 3  — Preprocess and chunk source text
  Phase 4  — Generate embeddings
  Phase 5  — Store chunks and vectors in Atlas
  Phase 6  — Create (or verify) the Atlas Vector Search index
  Phase 7  — Embed a user query
  Phase 8  — Run Atlas Vector Search with metadata filtering
  Phase 9  — Build a constrained prompt from retrieved context
  Phase 10 — Generate a grounded answer via LLM
  Phase 11 — Log the query, retrieved chunks, and answer to Atlas
  Phase 12 — Print a full audit trail

Run with:
    ```
    python app.py
    ```

Required environment variables (.env or shell):
    ```
    ATLAS_URI          MongoDB Atlas connection string
    OPENAI_API_KEY     OpenAI API key
    OPENAI_EMBED_MODEL Embedding model name  (default: text-embedding-3-small)
    OPENAI_CHAT_MODEL  Chat model name       (default: gpt-4o)
    DB_NAME            Atlas database name   (default: servicenow_rag)
    COLLECTION_CHUNKS  Chunk collection name (default: chunks)
    COLLECTION_AUDIT   Audit collection name (default: audit_log)
    ```
"""

import os
import re
import uuid
import json
import time
import textwrap
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from pymongo.operations import SearchIndexModel
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ATLAS_URI = os.getenv("ATLAS_URI", "mongodb+srv://<user>:<pass>@cluster.mongodb.net/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
DB_NAME = os.getenv("DB_NAME", "servicenow_rag")
COL_CHUNKS = os.getenv("COLLECTION_CHUNKS", "chunks")
COL_AUDIT = os.getenv("COLLECTION_AUDIT", "audit_log")
VECTOR_INDEX_NAME = "servicenow_vector_index"
EMBED_DIMS = 1536  # text-embedding-3-small output dimensions
TOP_K = 5

openai_client = OpenAI(api_key=OPENAI_API_KEY)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def banner(title: str) -> None:
    width = 70
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def step(n: int, label: str) -> None:
    print(f"\n[Phase {n}] {label}")
    print("-" * 50)


# ---------------------------------------------------------------------------
# Phase 1 — Connect to MongoDB Atlas
# ---------------------------------------------------------------------------

def connect_atlas() -> tuple[MongoClient, Any, Any]:
    step(1, "Connecting to MongoDB Atlas")
    client = MongoClient(ATLAS_URI)
    db = client[DB_NAME]
    # Ping to confirm connectivity
    client.admin.command("ping")
    print(f"  Connected to Atlas — database: {DB_NAME}")
    chunks_col = db[COL_CHUNKS]
    audit_col = db[COL_AUDIT]
    print(f"  Collections: {COL_CHUNKS}, {COL_AUDIT}")
    return client, chunks_col, audit_col


# ---------------------------------------------------------------------------
# Phase 2 — Simulate ServiceNow Records
# ---------------------------------------------------------------------------

# In production this would call the ServiceNow Table API:
#   GET /api/now/table/incident?sysparm_query=...&sysparm_fields=...
# and iterate paginated results. Here we use representative fixtures so the
# runbook is self-contained without a live ServiceNow instance.

SAMPLE_RECORDS: list[dict] = [
    {
        "sys_id": "abc001",
        "number": "INC0012345",
        "source_type": "incident",
        "short_description": "VPN MFA authentication failures for remote users",
        "description": (
            "Multiple users in the EMEA region are reporting that the VPN client "
            "is rejecting their MFA tokens. The issue started after the MFA policy "
            "was updated to enforce TOTP for all remote access. Users receive error "
            "code ERR-5021: token validation failed. Affected client version: 4.2.1."
        ),
        "close_notes": (
            "Root cause identified as a clock skew between the TOTP authenticator "
            "apps and the RADIUS server. Resolution: NTP sync enforced on RADIUS. "
            "Users instructed to re-sync authenticator app time settings."
        ),
        "metadata": {
            "product": "VPN",
            "category": "Network",
            "assignment_group": "Network Operations",
            "priority": "2 - High",
            "state": "Resolved",
            "tenant": "acme-corp",
            "trust_level": "verified",
        },
    },
    {
        "sys_id": "abc002",
        "number": "KB0056789",
        "source_type": "knowledge_base",
        "short_description": "How to resolve MFA clock skew errors on VPN",
        "description": (
            "Clock skew is the most common cause of TOTP-based MFA failures. "
            "TOTP tokens are time-sensitive and valid only within a ±30 second "
            "window. If the authenticator device clock drifts, tokens become invalid. "
            "\n\nResolution steps:\n"
            "1. Confirm RADIUS server is synced to a reliable NTP source.\n"
            "2. Ask the user to open their authenticator app settings and trigger "
            "a time correction / sync.\n"
            "3. If the issue persists, re-enroll the user's MFA device.\n"
            "4. Escalate to IAM team if re-enrollment fails."
        ),
        "close_notes": "",
        "metadata": {
            "product": "VPN",
            "category": "Identity & Access",
            "assignment_group": "IAM",
            "priority": None,
            "state": "Published",
            "tenant": "acme-corp",
            "trust_level": "verified",
        },
    },
    {
        "sys_id": "abc003",
        "number": "INC0098231",
        "source_type": "incident",
        "short_description": "Laptop battery not charging after BIOS update",
        "description": (
            "Several Dell Latitude 5540 laptops are not charging after the BIOS "
            "update to version 1.18.0 pushed via SCCM. Power LED blinks amber "
            "3 times. Affects approx 40 devices in the Chicago office."
        ),
        "close_notes": (
            "Dell engineering confirmed a regression in BIOS 1.18.0 affecting "
            "battery management firmware. Rollback to BIOS 1.17.2 resolves the "
            "issue. SCCM deployment corrected. Affected devices remediated."
        ),
        "metadata": {
            "product": "Laptop Hardware",
            "category": "Hardware",
            "assignment_group": "Desktop Support",
            "priority": "3 - Moderate",
            "state": "Resolved",
            "tenant": "acme-corp",
            "trust_level": "verified",
        },
    },
]


def fetch_servicenow_records() -> list[dict]:
    step(2, "Fetching ServiceNow Records")
    print(f"  Loaded {len(SAMPLE_RECORDS)} sample records (incident x2, knowledge_base x1)")
    for r in SAMPLE_RECORDS:
        print(f"    {r['number']} [{r['source_type']}] — {r['short_description'][:60]}")
    return SAMPLE_RECORDS


# ---------------------------------------------------------------------------
# Phase 3 — Preprocess and Chunk
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Strip HTML tags, collapse whitespace, normalise line breaks."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_text(text: str, max_chars: int = 600, overlap: int = 80) -> list[str]:
    """
    Simple sliding-window chunker.
    In production, replace with a semantic splitter (e.g. langchain RecursiveCharacterTextSplitter).
    """
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end].strip())
        start += max_chars - overlap
    return chunks


def preprocess_record(record: dict) -> list[dict]:
    """
    Combine relevant text fields, clean, and chunk into storable units.
    Returns a list of chunk dicts ready for embedding.
    """
    raw_parts = [
        record.get("short_description", ""),
        record.get("description", ""),
        record.get("close_notes", ""),
    ]
    combined = " ".join(clean_text(p) for p in raw_parts if p)
    text_chunks = chunk_text(combined)

    chunks = []
    for i, chunk in enumerate(text_chunks):
        chunks.append({
            "source_id": record["number"],
            "source_sys_id": record["sys_id"],
            "source_type": record["source_type"],
            "source_url": (
                f"https://instance.service-now.com/nav_to.do?"
                f"uri={record['source_type']}.do?sys_id={record['sys_id']}"
            ),
            "chunk_index": i,
            "chunk_text": chunk,
            "metadata": {
                **record["metadata"],
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
                "freshness_score": 1.0,
            },
        })
    return chunks


def preprocess_all(records: list[dict]) -> list[dict]:
    step(3, "Preprocessing and Chunking Records")
    all_chunks = []
    for record in records:
        chunks = preprocess_record(record)
        all_chunks.extend(chunks)
        print(f"  {record['number']} → {len(chunks)} chunk(s)")
    print(f"\n  Total chunks: {len(all_chunks)}")
    return all_chunks


# ---------------------------------------------------------------------------
# Phase 4 — Generate Embeddings
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch embed a list of strings using the configured OpenAI model."""
    response = openai_client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def embed_chunks(chunks: list[dict]) -> list[dict]:
    step(4, "Generating Embeddings")
    texts = [c["chunk_text"] for c in chunks]
    print(f"  Embedding {len(texts)} chunk(s) with model: {EMBED_MODEL}")
    vectors = embed_texts(texts)
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    print(f"  Done — each vector has {len(vectors[0])} dimensions")
    return chunks


# ---------------------------------------------------------------------------
# Phase 5 — Store in MongoDB Atlas
# ---------------------------------------------------------------------------

def store_chunks(chunks_col: Any, chunks: list[dict]) -> None:
    step(5, "Storing Chunks and Vectors in Atlas")
    ops = [
        UpdateOne(
            {"source_id": c["source_id"], "chunk_index": c["chunk_index"]},
            {"$set": c},
            upsert=True,
        )
        for c in chunks
    ]
    result = chunks_col.bulk_write(ops)
    print(f"  Upserted:  {result.upserted_count}")
    print(f"  Modified:  {result.modified_count}")
    print(f"  Total docs in collection: {chunks_col.count_documents({})}")


# ---------------------------------------------------------------------------
# Phase 6 — Atlas Vector Search Index
# ---------------------------------------------------------------------------

VECTOR_INDEX_DEFINITION = {
    "fields": [
        {
            "type": "vector",
            "path": "embedding",
            "numDimensions": EMBED_DIMS,
            "similarity": "cosine",
        },
        {"type": "filter", "path": "metadata.source_type"},
        {"type": "filter", "path": "metadata.product"},
        {"type": "filter", "path": "metadata.tenant"},
        {"type": "filter", "path": "metadata.trust_level"},
        {"type": "filter", "path": "metadata.state"},
    ]
}


def ensure_vector_index(chunks_col: Any) -> None:
    step(6, "Verifying Atlas Vector Search Index")
    existing = list(chunks_col.list_search_indexes())
    existing_names = [idx["name"] for idx in existing]

    if VECTOR_INDEX_NAME in existing_names:
        print(f"  Index '{VECTOR_INDEX_NAME}' already exists — skipping creation")
    else:
        print(f"  Creating index '{VECTOR_INDEX_NAME}' ...")
        index_model = SearchIndexModel(
            definition=VECTOR_INDEX_DEFINITION,
            name=VECTOR_INDEX_NAME,
            type="vectorSearch",
        )
        chunks_col.create_search_index(index_model)
        print(f"  Index creation initiated (may take 1-2 min to become READY on Atlas)")

    print(f"\n  Index definition:\n{json.dumps(VECTOR_INDEX_DEFINITION, indent=4)}")


# ---------------------------------------------------------------------------
# Phase 7 — Embed User Query
# ---------------------------------------------------------------------------

USER_QUERY = "Why are VPN users getting MFA errors and how do we fix it?"


def embed_query(query: str) -> list[float]:
    step(7, "Embedding User Query")
    print(f"  Query: \"{query}\"")
    vector = embed_texts([query])[0]
    print(f"  Query vector length: {len(vector)}")
    return vector


# ---------------------------------------------------------------------------
# Phase 8 — Atlas Vector Search with Metadata Filtering
# ---------------------------------------------------------------------------

def vector_search(
    chunks_col: Any,
    query_vector: list[float],
    tenant: str = "acme-corp",
    top_k: int = TOP_K,
) -> list[dict]:
    step(8, "Running Atlas Vector Search with Metadata Filters")

    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": top_k * 10,
                "limit": top_k,
                "filter": {
                    "metadata.tenant": {"$eq": tenant},
                    "metadata.trust_level": {"$eq": "verified"},
                },
            }
        },
        {
            "$project": {
                "_id": 0,
                "source_id": 1,
                "source_type": 1,
                "source_url": 1,
                "chunk_index": 1,
                "chunk_text": 1,
                "metadata.product": 1,
                "metadata.state": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    results = list(chunks_col.aggregate(pipeline))
    print(f"  Filters applied: tenant={tenant}, trust_level=verified")
    print(f"  Retrieved {len(results)} chunk(s):\n")
    for r in results:
        print(
            f"    [{r['source_id']}#{r['chunk_index']}]  "
            f"score={r['score']:.4f}  "
            f"type={r['source_type']}\n"
            f"    {textwrap.shorten(r['chunk_text'], width=80)}\n"
        )
    return results


# ---------------------------------------------------------------------------
# Phase 9 — Build Constrained Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a support assistant. Answer ONLY using the context passages provided below.
    Do not use any external knowledge or make assumptions beyond the context.
    For every factual claim, cite the source ID in square brackets, e.g. [INC0012345] or [KB0056789].
    If the context does not contain sufficient information to answer the question, respond with:
    "I don't have enough information in the available records to answer this."
""")


def build_prompt(query: str, chunks: list[dict]) -> tuple[str, str]:
    step(9, "Building Constrained Prompt")

    context_blocks = []
    for c in chunks:
        block = (
            f"--- Source: {c['source_id']} (chunk {c['chunk_index']}) "
            f"| type: {c['source_type']} | score: {c['score']:.4f} ---\n"
            f"{c['chunk_text']}"
        )
        context_blocks.append(block)

    context_str = "\n\n".join(context_blocks)
    user_message = f"Context:\n{context_str}\n\nQuestion:\n{query}"

    print("  System prompt (truncated):")
    print(textwrap.indent(SYSTEM_PROMPT[:200] + "...", "    "))
    print(f"\n  Context passages: {len(chunks)}")
    print(f"  Total context characters: {len(context_str)}")
    return SYSTEM_PROMPT, user_message


# ---------------------------------------------------------------------------
# Phase 10 — Generate Grounded Answer
# ---------------------------------------------------------------------------

def generate_answer(system_prompt: str, user_message: str) -> str:
    step(10, "Generating Grounded Answer via LLM")
    print(f"  Model: {CHAT_MODEL}")
    print("  Calling OpenAI Chat Completions API ...")

    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,  # deterministic — grounding over creativity
        max_tokens=512,
    )
    answer = response.choices[0].message.content
    print(f"\n  Answer:\n{textwrap.indent(answer, '    ')}")
    return answer


# ---------------------------------------------------------------------------
# Phase 11 — Audit Log
# ---------------------------------------------------------------------------

def log_to_atlas(
    audit_col: Any,
    query: str,
    retrieved_chunks: list[dict],
    answer: str,
    latency_ms: int,
) -> str:
    step(11, "Logging Query, Chunks, and Answer to Atlas Audit Log")

    log_entry = {
        "query_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc),
        "user_query": query,
        "retrieved_chunks": [
            {
                "source_id": c["source_id"],
                "chunk_index": c["chunk_index"],
                "score": c["score"],
            }
            for c in retrieved_chunks
        ],
        "answer": answer,
        "latency_ms": latency_ms,
        "model": CHAT_MODEL,
        "embed_model": EMBED_MODEL,
        "feedback": None,
    }

    result = audit_col.insert_one(log_entry)
    print(f"  Audit log written — _id: {result.inserted_id}")
    print(f"  query_id: {log_entry['query_id']}")
    return log_entry["query_id"]


# ---------------------------------------------------------------------------
# Phase 12 — Full Audit Trail Summary
# ---------------------------------------------------------------------------

def print_audit_trail(
    query: str,
    retrieved_chunks: list[dict],
    answer: str,
    query_id: str,
    latency_ms: int,
) -> None:
    step(12, "Full Audit Trail")
    print(f"  query_id    : {query_id}")
    print(f"  latency     : {latency_ms}ms")
    print(f"  query       : {query}")
    print(f"  sources used:")
    for c in retrieved_chunks:
        print(f"    - {c['source_id']} chunk#{c['chunk_index']}  score={c['score']:.4f}")
    print(f"\n  final answer:\n{textwrap.indent(answer, '    ')}")


# ---------------------------------------------------------------------------
# Main Runbook
# ---------------------------------------------------------------------------

def main() -> None:
    banner("ServiceNow RAG on MongoDB Atlas — End-to-End Runbook")

    start_total = time.time()

    # Phase 1 — Connect
    client, chunks_col, audit_col = connect_atlas()

    # Phase 2 — Ingest ServiceNow records
    records = fetch_servicenow_records()

    # Phase 3 — Preprocess and chunk
    chunks = preprocess_all(records)

    # Phase 4 — Embed chunks
    chunks = embed_chunks(chunks)

    # Phase 5 — Store in Atlas
    store_chunks(chunks_col, chunks)

    # Phase 6 — Ensure vector search index exists
    ensure_vector_index(chunks_col)

    # Phase 7 — Embed user query
    query_vector = embed_query(USER_QUERY)

    # Phase 8 — Vector search
    t0 = time.time()
    retrieved = vector_search(chunks_col, query_vector)
    retrieval_ms = int((time.time() - t0) * 1000)

    # Phase 9 — Build constrained prompt
    system_prompt, user_message = build_prompt(USER_QUERY, retrieved)

    # Phase 10 — Generate answer
    t1 = time.time()
    answer = generate_answer(system_prompt, user_message)
    generation_ms = int((time.time() - t1) * 1000)

    total_ms = int((time.time() - start_total) * 1000)

    # Phase 11 — Audit log
    query_id = log_to_atlas(audit_col, USER_QUERY, retrieved, answer, total_ms)

    # Phase 12 — Summary
    print_audit_trail(USER_QUERY, retrieved, answer, query_id, total_ms)

    banner(f"Runbook Complete  |  retrieval={retrieval_ms}ms  generation={generation_ms}ms  total={total_ms}ms")

    client.close()


if __name__ == "__main__":
    main()
