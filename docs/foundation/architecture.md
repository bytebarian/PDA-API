# PDA Foundation Architecture

The Foundation phase delivers a working local PDA shell: existing React frontend connected to a FastAPI backend, PostgreSQL with pgvector, local file storage, a replaceable processing runner, OCR/text extraction, chunking, embeddings, retrieval, chat with citations, report generation, persisted settings, and smoke tests.

```mermaid
flowchart LR
  FE[pda-personal-documen React shell]
  API[PDA-API FastAPI]
  DB[(PostgreSQL + pgvector)]
  FS[(Local document storage)]
  JOB[Processing runner]
  OCR[OCR / extraction adapter]
  LLM[Local model adapter]

  FE -->|REST JSON| API
  API --> DB
  API --> FS
  API --> JOB
  JOB --> DB
  JOB --> FS
  JOB --> OCR
  JOB --> LLM
```
