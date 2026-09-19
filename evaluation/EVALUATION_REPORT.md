# RAG Pipeline Evaluation Report

**Date:** 2026-09-19 14:36:30
**Pipeline:** Hybrid Search (Vector + FTS5 BM25) with Reciprocal Rank Fusion (RRF)

## 1. Executive Summary

| Metric | Target | Measured | Status |
| :--- | :--- | :--- | :--- |
| **HitRate@4** | $\ge 90.0\%$ | **100.0%** (7/7) | **PASS** |
| **Mean Reciprocal Rank (MRR)** | $\ge 0.80$ | **1.0000** | **PASS** |
| **Average Query Latency** | $< 100\text{ ms}$ | **10.21 ms** | **PASS** |

## 2. Per-Query Breakdown

| ID | Document | Target Page | Hit? | Rank | Reciprocal Rank | Latency (ms) | Top Source | Top Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `rag-eval-01` | `company_policy.pdf` | 3 | ✅ Yes | 1 | 1.0000 | 12.08 | `hybrid` | 0.0328 |
| `rag-eval-02` | `company_policy.pdf` | 1 | ✅ Yes | 1 | 1.0000 | 9.97 | `hybrid` | 0.0328 |
| `rag-eval-03` | `company_policy.pdf` | 2 | ✅ Yes | 1 | 1.0000 | 10.11 | `hybrid` | 0.0328 |
| `rag-eval-04` | `security_guidelines.md` | 1 | ✅ Yes | 1 | 1.0000 | 10.54 | `hybrid` | 0.0328 |
| `rag-eval-05` | `security_guidelines.md` | 1 | ✅ Yes | 1 | 1.0000 | 9.03 | `hybrid` | 0.0328 |
| `rag-eval-06` | `vacation_rules.docx` | 1 | ✅ Yes | 1 | 1.0000 | 9.55 | `hybrid` | 0.0328 |
| `rag-eval-07` | `vacation_rules.docx` | 1 | ✅ Yes | 1 | 1.0000 | 10.19 | `hybrid` | 0.0328 |

## 3. Methodology & Evaluation Criteria
- **Ingestion:** Page-aware recursive chunking (chunk_size=350, chunk_overlap=50) preserving 1-based page numbers.
- **Retrieval:** Multi-tenant hybrid retrieval querying SQLite dense embeddings and FTS5 inverted index.
- **Fusion:** Reciprocal Rank Fusion score $RRF(d) = \sum_{m \in \{vec, fts\}} \frac{1}{60 + rank_m(d)}$.
- **Hit Definition:** Ground truth text phrase matches retrieved chunk, matching filename and page number within Top-K.
