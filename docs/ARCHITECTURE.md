# CompliAI - System Architecture Documentation

## Overview

CompliAI (Policy & Compliance Assistant) is a streamlined RAG-based application with a Streamlit interface for easy interaction. It uses LangChain for orchestration, ChromaDB for vector storage, and OpenAI GPT-4 for answer generation.

## Component Architecture

### 1. User Interface (Streamlit)

**Technology**: Streamlit 1.29.0

**Features**:

- Chat interface for natural language queries
- Document upload with metadata (department, policy type, effective date)
- Real-time query history display
- Citation and confidence score visualization
- Export functionality for chat history
- CompliAI system statistics dashboard

### 2. RAG Pipeline (LangChain)

#### 2.1 Document Ingestion

```python
Document → Loader → Text Splitter → Embeddings → Vector Store
```

**Chunking Strategy**:

```python
RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)
```

**Embedding Model**: OpenAI `text-embedding-3-small`

#### 2.2 Vector Store (ChromaDB)

**Technology**: ChromaDB 0.4.22

**Storage Configuration**:

- Persistent storage in `./data/vector_store`
- Collection name: `compliai_policies`
- Distance metric: Cosine similarity
- Local file-based storage (DuckDB + Parquet)

#### Retrieval Pipeline

**Hybrid Search**:

- Semantic search (vector similarity)
- Optional Cohere reranking
- Context compression

#### Answer Generation

**LLM**: GPT-4-turbo with PDO prompting

### 3. Observability Layer

#### LangSmith Integration (Optional)

**Project Name**: `compliai-assistant`

**Traced Operations**:

- Document loading
- Embedding generation
- Vector search
- LLM calls
- Token usage

#### Local Logging

**Log Locations**:

- Application logs: `./logs/compliai.log`
- Query history: Session state (exportable)
- Performance metrics: Displayed in UI

## Data Flow

```
┌─────────────────┐
│  User Question  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  CompliAI UI    │
│  (Streamlit)    │
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│  Query Engine       │
│  (LangChain)       │
└────────┬──────────┘
         │
         ▼
┌─────────────────────┐
│  Retriever          │
│  (Hybrid Search)   │
└────────┬──────────┘
         │
         ▼
┌─────────────────────┐
│  Vector Store       │
│  (ChromaDB)        │
└────────┬──────────┘
         │
         ▼
┌─────────────────────┐
│  Generator          │
│  (LLM: GPT-4-turbo)│
└────────┬──────────┘
         │
         ▼
┌─────────────────────┐
│  Structured Answer  │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Display in UI      │
└─────────────────────┘
```

## File Structure

```
compliai/
├── app.py                          # CompliAI Streamlit application
├── src/
│   ├── ingestion/
│   │   └── document_processor.py   # Document loading and chunking
│   ├── rag/
│   │   ├── query_engine.py         # Main orchestrator
│   │   ├── retriever.py            # Hybrid retrieval logic
│   │   └── generator.py            # Answer generation with PDO
│   ├── prompts/
│   │   └── templates.py            # PDO prompt templates
│   ├── evaluation/
│   │   └── evaluator.py            # RAG evaluation pipeline
│   └── utils/
│       ├── vector_store.py         # ChromaDB wrapper
│       └── embeddings.py           # Embedding model config
├── data/
│   ├── uploads/                    # Uploaded policy documents
│   └── vector_store/               # ChromaDB persistent storage
├── tests/
│   ├── unit/                       # Unit tests
│   ├── integration/                # Integration tests
│   └── test_queries.json           # Test dataset
├── scripts/
│   ├── evaluate_rag.py             # Evaluation runner
│   └── batch_ingest.py             # Batch document processor
└── docs/
    ├── ARCHITECTURE.md             # This file
    └── EVALUATION.md               # Evaluation methodology
```

## Performance Characteristics

- Query Response: <3 seconds (p95)
- Document Ingestion: <5 minutes per 100-page PDF
- Concurrent Users: 50+ (Streamlit limitation)
- Queries per Minute: Unlimited (local processing)

## Security Considerations

**Data Privacy**:

- All CompliAI processing happens locally (except API calls)
- Policy documents stored on local disk
- No data sent to third parties (except LLM APIs)

## Deployment Options

### Local Development

```bash
streamlit run app.py
```

### Production Deployment

**Option 1: Streamlit Community Cloud**

- Free hosting for public apps
- Automatic HTTPS
- Easy GitHub integration
- Deploy CompliAI with one click

**Option 2: Streamlit for Teams**

- Private app sharing
- Team management
- Usage analytics
- Deploy on your infrastructure

**Option 3: Cloud VM**

- Deploy CompliAI to AWS EC2, GCP Compute, or Azure VM
- Use nginx as reverse proxy
- Enable HTTPS with Let's Encrypt

## Future Enhancements

1. **Multi-user Support**: Add authentication and user-specific history to CompliAI
2. **Advanced Analytics**: Dashboard for query patterns and trends
3. **API Endpoint**: REST API for programmatic access to CompliAI
4. **Fine-tuning**: Custom embedding model for domain-specific terms
5. **Multi-modal**: Support for images and tables in policies
6. **Automated Updates**: Monitor policy changes and re-index
7. **Slack/Teams Integration**: CompliAI chat bot interface
8. **Mobile App**: Native iOS/Android CompliAI applications
