# Persian News RAG Project

A **Retrieval-Augmented Generation (RAG)** system built for answering questions based on a dataset of Persian news articles from major Iranian news agencies (FarsNews, MehrNews, ISNA, and others).

## Overview

This project implements a baseline RAG pipeline to retrieve relevant news documents and generate accurate, context-based answers in Persian. It combines **lexical** and **semantic** retrieval methods with a large language model (LLM) for response generation.

The system processes a large Persian news dataset, cleans and chunks the articles, indexes them for fast retrieval, and uses an LLM to produce concise Persian answers grounded in the retrieved context.

Key features:
- Persian text normalization (handling Arabic-Persian character differences, diacritics, digits).
- Chunking of long articles for efficient embedding.
- Three retrieval strategies: BM25 (lexical), dense embeddings (semantic), and hybrid.
- Answer generation via Groq API (Llama-3.1-8B).
- Evaluation on 15 hand-crafted questions with ground-truth references.

## Dataset

- **Source**: [Persian News Dataset on Kaggle](https://www.kaggle.com/datasets/... ) (approximately 390k articles from 2021–2023).
- **Subset**: Randomly sampled 20,000 articles for feasible experimentation → saved as `news_subset.csv`.
- Articles include title, body, abstract, tags, category, date, and agency.

## Project Structure

```
.
├── rag-project-notebook-1.ipynb      # Data preprocessing & subset creation
├── rag-project-notebook-2.ipynb      # RAG pipeline, retrieval, generation & evaluation
├── news_subset.csv                   # Processed 20k article subset
├── test_results.json                 # Evaluation results on 15 questions
└── README.md                         # This file
```

## Dependencies

- Python 3.x
- pandas
- numpy
- sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2)
- faiss-cpu
- rank-bm25
- groq (for LLM inference)

Install with:

```bash
pip install pandas numpy sentence-transformers faiss-cpu rank-bm25 groq
```

## How It Works

1. **Preprocessing** (`notebook-1`):
   - Load raw CSV.
   - Drop rows without body text.
   - Normalize Persian text (character mapping, diacritics removal, digit conversion, zero-width chars).
   - Save cleaned subset.

2. **Chunking**:
   - Combine title + body.
   - Split into chunks of ~800 characters (with overlap handling for long texts).

3. **Indexing**:
   - **Lexical**: BM25 on tokenized chunks.
   - **Semantic**: Embeddings via `paraphrase-multilingual-MiniLM-L12-v2` + FAISS inner-product index.

4. **Retrieval**:
   - `lexical`: Pure BM25.
   - `semantic`: Dense vector search.
   - `hybrid`: Top-20 BM25 → rerank with cosine similarity (top-5).

5. **Generation**:
   - Retrieved chunks concatenated as context.
   - Prompted to Groq's Llama-3.1-8B with a system prompt enforcing Persian, concise, and faithful answers.

## Evaluation Results

Tested on 15 questions with known ground-truth document IDs.

| Method   | Successful Reference Retrieval (out of 15) |
|----------|--------------------------------------------|
| Lexical (BM25)     | 15                                         |
| Semantic           | 10                                         |
| Hybrid             | 13                                         |

BM25 performed best on this keyword-rich Persian news dataset, while hybrid offered a strong balance.

**Note**: The example query in notebook-2 ("سرمربی تیم بارسلونا کیه؟") returns "Ronald Koeman" because the retrieved articles are from 2021. In reality (as of January 2026), the current Barcelona head coach is **Hansi Flick**.

## Usage

1. Run `rag-project-notebook-1.ipynb` to generate `news_subset.csv` (if starting from full dataset).
2. Run `rag-project-notebook-2.ipynb` to build indexes and test queries.
3. Modify the `answer()` function with your own Groq API key (store via Kaggle Secrets or environment variable).

Example query:

```python
response, doc_ids, chunks = answer("سوال شما به زبان پارسی", 'hybrid')
print(response)
```

## Future Improvements

- Use Persian-specific embedding models (e.g., fine-tuned on Persian data).
- Advanced reranking (e.g., cross-encoder).
- Larger evaluation set with automated metrics (BLEU, ROUGE, BERTScore).
- Deployment as a web API (e.g., with FastAPI or Gradio).
- Support for longer context LLMs.

## License

MIT License – feel free to use, modify, and share.

---

Built as an exploration of RAG techniques on non-English (Persian) news data. Contributions welcome!
