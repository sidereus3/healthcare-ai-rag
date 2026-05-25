# Healthcare AI RAG Pipeline

An enterprise-grade Retrieval-Augmented Generation (RAG) pipeline designed for parsing, indexing, and executing fully grounded semantic queries over clinical transcripts and medical histories. 

This system pairs a high-performance local **FAISS** vector database with **Azure OpenAI (GPT-4o & Text-Embedding-3-Large)** via LangChain Expression Language (LCEL) to provide context-constrained clinical analysis without data hallucination.

---

## System Architecture

The workflow isolates data ingestion from the query execution loop to save on compute resources and minimize API costs:

1. **Ingestion & Preprocessing (`pipeline.py`)**: Parses raw clinical datasets (`mtsamples.csv`), breaks records into semantic chunks using a `RecursiveCharacterTextSplitter`, converts chunks into dense vector embeddings via Azure's `text-embedding-3-large`, and builds/caches a local FAISS index.
2. **Retrieval & Generation (`query.py`)**: Takes user-facing clinical questions, performs a localized similarity search across the vector matrix, extracts the top $k$ relevant matching clinical contexts, and injects them into a strict, defensively guarded system prompt executed by Azure `gpt-4o`.

---

## Features

- **Strict Clinical Grounding**: The LLM system layer is strictly locked down. If the required clinical context is missing from the underlying dataset, the engine explicitly declines to answer rather than extrapolating or hallucinating medical hypotheses.
- **Data Privacy & Optimization Built-In**: Local vector database binaries (`faiss_index/`) are kept entirely out of Git history via targeted `.gitignore` patterns to prevent leakage of medical text or index serialization bloat.
- **Modern Package Ecosystem**: Leverages standalone partner packages (`langchain-community-faiss`) utilizing unified namespaces for robust cross-version support.

---

## Installation & Setup

### Prerequisites
- Python 3.11 or higher
- [Poetry](https://python-poetry.org/) for Python dependency management
- An active Microsoft Azure Subscription with Azure OpenAI endpoints provisioned in a region containing valid model quotas (e.g., `eastus2`).

### 1. Clone and Install Dependencies
Clone the repository and spin up the isolated virtual environment using Poetry:

```bash
git clone [https://github.com/sidereus3/healthcare-ai-rag.git](https://github.com/sidereus3/healthcare-ai-rag.git)
cd healthcare-ai-rag
poetry install
