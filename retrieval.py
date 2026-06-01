"""Module 8 — Applied Lab: Vector Retrieval.

Implement BM25, dense, and hybrid retrievers against a Weaviate index of the
CQADupStack + Stack Exchange technical-Q&A corpus, then evaluate all three on
the bundled 60-pair labeled set.

Methodology (canonical — autograder enforces):
- recall@k: gold_doc_id in top_k_returned_ids; mean over all queries.
- MRR: 1-indexed position of gold_doc_id in returned list of length 10;
  1/rank if found, 0 if not; mean over all queries.
- Hybrid alpha: 0.5 for the base assignment.
- Top-k for retrieval calls during evaluation: k=10; recall@5 is the top-5
  slice of those 10. One retrieval call per query.
"""

import json
from typing import Callable

import weaviate

CLASS_NAME = "Post"


def create_schema(client: weaviate.Client) -> None:
    """Create the Post class in Weaviate.

    Properties:
      - doc_id (text, filterable): globally-unique "{subset}:{post_id}"
      - subset (text, filterable): one of "programmers" / "webmasters" / "android"
      - title (text, BM25-indexed)
      - question_text (text, BM25-indexed)
      - answer_text (text, BM25-indexed)
      - text (text, stored — NOT BM25-indexed; double-counts otherwise)

    Class-level config:
      - vectorizer: "none" (we supply vectors externally)
      - vectorIndexConfig: {"distance": "cosine"}

    If the class already exists, delete it first (so re-running create_schema
    on an existing index is idempotent).

    The BM25 retrieval surface is the three BM25-indexed properties; `text`
    exists as the unified dense-embedding source and a backward-compat
    "full doc" view but does not participate in BM25.
    """
    # if the class exists, delete it
    if client.schema.exists("Post"):
        client.schema.delete_class("Post")

    # build the class definition dict with the 6 properties above
    # client.schema.create_class(class_def)
    
    client.schema.create_class({
        "class": "Post",
        "vectorizer": "none",                   
        "vectorIndexConfig": {"distance": "cosine"}, 
        "properties": [
            {
                "name": "doc_id",
                "dataType": ["text"],
                "indexSearchable": False, 
                "indexFilterable": True,   
                "tokenization": "field",
            },
            {
                "name": "subset",
                "dataType": ["text"],
                "indexSearchable": False,
                "indexFilterable": True,
                "tokenization": "field",
            },
            {
                "name": "title",
                "dataType": ["text"],
                "indexSearchable": True,   
                "indexFilterable": False,
                "tokenization": "word",
            },
            {
                "name": "question_text",
                "dataType": ["text"],
                "indexSearchable": True,   
                "indexFilterable": False,
                "tokenization": "word",
            },
            {
                "name": "answer_text",
                "dataType": ["text"],
                "indexSearchable": True,   
                "indexFilterable": False,
                "tokenization": "word",
            },
            {
                "name": "text",
                "dataType": ["text"],
                "indexSearchable": False,  
                "indexFilterable": False,
                "tokenization": "word",
            },
        ],
    })


    


def index_corpus(client: weaviate.Client, corpus_path: str, embedder) -> int:
    """Embed and ingest the corpus into the Post class.

    For each line in `corpus_path` (JSONL, one document per line):
      - Embed `row["text"]` with `embedder.encode(...)` (returns a numpy array)
      - Add a Weaviate object with vector=qv.tolist() and all 6 properties
        populated from the row.

    Use `client.batch` for efficiency. Call `client.batch.flush()` (or use
    a `with client.batch as batch:` context) so the final batch commits.

    Returns the count of ingested objects (verify via Aggregate query, or
    simply track count as you ingest).
    """
    # TODO: load the corpus from corpus_path (JSONL)
    # TODO: batch-embed the texts (model.encode(texts, batch_size=64) for speed)
    # TODO: ingest each row with vector + all 6 properties
    # TODO: flush the batch and return the count
    raise NotImplementedError("index_corpus is not yet implemented")


def bm25_search(client: weaviate.Client, query: str, k: int) -> list[str]:
    """BM25 retrieval. Return ordered list of doc_id strings, length <= k.

    Use:
        client.query.get("Post", ["doc_id"]).with_bm25(query=query).with_limit(k).do()
    """
    # TODO: run the BM25 query; extract doc_id values from response
    raise NotImplementedError("bm25_search is not yet implemented")


def dense_search(client: weaviate.Client, query: str, k: int, embedder) -> list[str]:
    """Dense retrieval. Embed the query with the same embedder used at ingest.

    Use:
        client.query.get("Post", ["doc_id"]).with_near_vector({"vector": qv}).with_limit(k).do()
    """
    # TODO: embed the query (qv = embedder.encode(query).tolist())
    # TODO: run the near_vector query; extract doc_id values
    raise NotImplementedError("dense_search is not yet implemented")


def hybrid_search(client: weaviate.Client, query: str, k: int, embedder, alpha: float = 0.5) -> list[str]:
    """Hybrid retrieval. alpha=0.5 is the canonical mix for the base assignment.

    Use:
        client.query.get("Post", ["doc_id"]).with_hybrid(query=query, vector=qv, alpha=alpha).with_limit(k).do()
    """
    # TODO: embed the query, run hybrid, extract doc_id values
    raise NotImplementedError("hybrid_search is not yet implemented")


def evaluate_retriever(eval_path: str, search_fn: Callable, k_values=(5, 10)) -> dict:
    """Evaluate a retriever against the labeled set.

    For each (query, gold_doc_id, query_type) row:
      - Call search_fn(query, k=max(k_values))  # one call per query
      - Compute hit@5 (gold in top-5) and hit@10 (gold in top-10)
      - Compute MRR contribution: 1/rank (1-indexed) if gold in top-10, else 0

    Return:
        {
          "recall@5": <mean hit@5>,
          "recall@10": <mean hit@10>,
          "mrr": <mean MRR>,
          "by_type": {  # REQUIRED — used in the comparison brief
            "factoid": {"recall@5": ..., "recall@10": ..., "mrr": ...},
            "paraphrastic": {"recall@5": ..., "recall@10": ..., "mrr": ...}
          }
        }
    """
    # TODO: load eval_path (JSONL); iterate rows
    # TODO: for each row, call search_fn(query, k=max(k_values)); compute hit@5/hit@10/MRR
    # TODO: aggregate across all rows; also split by query_type into by_type
    raise NotImplementedError("evaluate_retriever is not yet implemented")
