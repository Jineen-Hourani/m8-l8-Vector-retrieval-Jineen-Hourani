# Comparison Brief — Module 8 Lab

## Metrics Table

| Retriever | recall@5 | recall@10 | MRR | factoid recall@5 | paraphrastic recall@5 |
|---|---|---|---|---|---|
| BM25 | 0.57 | 0.63 | 0.55 | 1.00 | 0.13 |
| Dense | 0.90 | 0.93 | 0.67 | 0.83 | 0.97 |
| Hybrid (α=0.5) | 0.85 | 0.98 | 0.70 | 1.00 | 0.70 |

## Where BM25 Wins

BM25 achieves perfect recall on factoid queries (recall@5 = 1.0) because these
queries contain exact tokens that appear verbatim in the indexed fields.

1. Factoid queries with exact technical identifiers — e.g. a query naming a
   specific error code, library name, or version string. BM25 matches the rare
   token directly against `title` and `question_text`, ranking the gold doc
   first. Dense embeddings smooth over rare tokens and may miss the exact match.

2. Queries that quote a distinctive phrase from the original post — BM25 rewards
   high term frequency of an uncommon word sequence, while dense retrieval
   collapses synonyms and dilutes the signal from the specific token.

## Where Dense Wins

Dense retrieval dominates paraphrastic queries (recall@5 = 0.97 vs. BM25's 0.13),
where the user's wording differs from the document's wording.

1. A query like "best practices for structuring a large codebase" can match a
   post titled "How do I refactor a long function?" — same concept, zero shared
   tokens. BM25 scores this near zero; dense retrieval places both in the same
   vector neighbourhood.

2. Cross-vocabulary questions where the user writes in plain English and the
   post uses technical jargon (or vice versa). The embedding model maps both
   surface forms to nearby vectors, recovering the gold doc that BM25 misses
   entirely.

## Alpha Recommendation

Recommended alpha: **0.5**, with an acceptable range of 0.4–0.6.

The labeled set is split between factoid and paraphrastic queries. BM25 handles
factoid queries perfectly (recall@5 = 1.0) while dense handles paraphrastic
queries almost perfectly (recall@5 = 0.97). A balanced alpha of 0.5 lets each
mode contribute where it is strongest, producing the best overall recall@10
(0.98) and MRR (0.70) of the three retrievers. Pushing alpha below 0.4 would
weaken the BM25 signal on factoid queries; pushing above 0.6 would weaken the
semantic signal on paraphrastic queries.

## Schema Choice — Cosine vs. Dot Product

The schema uses `"distance": "cosine"`. The `all-MiniLM-L6-v2` model outputs
L2-normalized vectors, which means cosine similarity and dot product produce
identical rankings on normalized embeddings. However, cosine distance is the
safer default: if vectors are ever re-generated with a model that does not
normalize outputs, cosine continues to measure angular similarity correctly
while dot product rankings would degrade. Cosine also generalizes more safely
across embedding models during future upgrades to the index.