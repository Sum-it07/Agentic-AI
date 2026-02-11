# RAG Assignment - EduRAG: Smart Education using RAG

## Problem Statement

Students often struggle to find accurate, textbook-grounded answers to their academic queries. Internet searches introduce unverified or irrelevant content. **EduRAG** solves this by building a Retrieval-Augmented Generation (RAG) pipeline over NCERT textbook PDFs, ensuring every answer is fact-grounded and sourced directly from the syllabus material.

## Dataset / Knowledge Source

- **Type of data:** PDF (NCERT textbook chapters)
- **Data source:** Public - NCERT textbooks freely available at [ncert.nic.in](https://ncert.nic.in/)
- Documents are placed in the `data/` directory and processed via `ingest.py`

## RAG Architecture

```
+------------+    +--------------+    +--------------------+    +-----------+
|  NCERT     |--->|  PyPDFLoader |--->| RecursiveCharacter  |--->|  Chroma   |
|  PDFs      |    | (LangChain)  |    |  TextSplitter       |    | VectorDB  |
+------------+    +--------------+    +--------------------+    +-----+-----+
                                                                      |
                    +-------------------------------------------------+
                    v
+------------+    +--------------+    +-----------------+    +------------+
|  User      |--->|  Query       |--->| Hybrid Search   |--->|  Gemini    |
|  Query     |    | Preprocessing|    | (Semantic +     |    |  2.5 Flash |
+------------+    | + Synonym    |    |  Keyword)       |    |  (LLM)     |
                  |  Expansion   |    +-----------------+    +-----+------+
                  +--------------+                                 |
                                                                   v
                                                            +------------+
                                                            |  Answer +  |
                                                            |  Sources   |
                                                            +------------+
```

**Pipeline steps:**
1. **Data Ingestion** (`ingest.py`) - Load PDFs -> chunk text -> embed with HuggingFace -> store in Chroma
2. **Query Processing** (`backend.py`) - Preprocess query with synonym expansion -> hybrid search (semantic + keyword) -> retrieve top-k documents
3. **Answer Generation** - Feed retrieved context + conversation history into Gemini 2.5 Flash via LangChain `LLMChain` -> return answer with source references

## Text Chunking Strategy

- **Chunk size:** 1000 characters
- **Chunk overlap:** 100 characters
- **Splitter:** `RecursiveCharacterTextSplitter` (LangChain)
- **Reason:** A 1000-char chunk balances retaining enough context per chunk (full paragraphs/concepts) while staying within embedding model token limits. The 100-char overlap prevents information loss at chunk boundaries. `RecursiveCharacterTextSplitter` is chosen because it splits on natural boundaries (paragraphs -> sentences -> words) preserving semantic coherence.

## Embedding Details

- **Embedding model:** `all-MiniLM-L6-v2` (via `HuggingFaceEmbeddings` from `sentence-transformers`)
- **Reason:** Lightweight (80 MB), fast inference on CPU, produces 384-dimensional embeddings, and ranks among the top models on the MTEB benchmark for its size. Ideal for educational RAG where low latency and local execution (no API cost) are important.

## Vector Database

- **Vector store:** **ChromaDB** (`langchain_community.vectorstores.Chroma`)
- Persistent storage at `chroma_db/` directory
- Supports similarity search with scores, used for both semantic retrieval and the hybrid search implementation

## Notebook Implementation

> The project is implemented as modular Python scripts rather than a single notebook. The step-wise flow is:

| Step | File | Description |
|------|------|-------------|
| 1. Data Loading & Chunking | `ingest.py` | Loads PDFs from `data/`, splits into chunks, embeds, and persists to ChromaDB |
| 2. Retrieval | `backend.py` | Hybrid search (semantic + keyword), query preprocessing with synonym expansion |
| 3. Generation | `backend.py` | LangChain `LLMChain` with Gemini 2.5 Flash, conversational prompt with history |
| 4. UI / Interaction | `streamlit_app.py` | Streamlit chat interface with text, voice, and image input |

**Sample test queries (domain: NCERT Physics):**
1. *"What are the laws of motion?"*
2. *"Explain the concept of conservation of energy"*
3. *"What is the difference between electricity and magnetism?"*

## Future Improvements

- **Better chunking** - Use semantic chunking or section-aware splitting based on headings/chapters
- **Reranking / hybrid search** - Add a cross-encoder reranker (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) on top of the current hybrid retrieval
- **Metadata filtering** - Filter by chapter, subject, or page number before retrieval for more targeted results
- **UI integration** - The Streamlit UI is already built; future work could add user authentication, bookmarks, and study progress tracking

## README / Report

### Project Overview
EduRAG is an AI-powered educational assistant that uses RAG to answer student questions grounded in NCERT textbook content. It features a conversational Streamlit UI with text, voice, and image input support, plus a multilingual variant.

### Tools & Libraries Used
| Category | Tools |
|----------|-------|
| Framework | LangChain |
| LLM | Google Gemini 2.5 Flash (`langchain-google-genai`) |
| Embeddings | `all-MiniLM-L6-v2` (`sentence-transformers`) |
| Vector Store | ChromaDB |
| PDF Parsing | `PyPDFLoader` (LangChain) |
| Frontend | Streamlit |
| Voice Input | `SpeechRecognition`, `PyAudio` |
| Image Input | Pillow |
| Translation | `deep-translator` |
| Environment | `python-dotenv` |

### Instructions to Run

```bash
# 1. Clone the repository and navigate to the project
cd EduRAG

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create a .env file with your API key
echo GOOGLE_API_KEY="your_key_here" > .env

# 5. Place NCERT PDFs in the data/ folder, then run ingestion
python ingest.py

# 6. Launch the Streamlit app
streamlit run streamlit_app.py
# Or for the multilingual version:
streamlit run streamlit_app_multilingual.py
```

## Bonus (Optional)

- **Streamlit UI** - Full chat-based interface implemented in `streamlit_app.py` with:
  - Conversational memory (multi-turn dialogue)
  - Voice input via `SpeechRecognition`
  - Image upload support
  - Source citation with page numbers
- **Multilingual support** - `streamlit_app_multilingual.py` adds language translation via `deep-translator`
- **Flask API** - `app.py` exposes a REST endpoint (`POST /ask`) for programmatic access
