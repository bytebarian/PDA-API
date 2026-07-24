# PDA-API Technical Documentation

## 1. Executive summary

PDA-API is a privacy-first FastAPI backend for a Personal Documents Assistant. It stores uploaded documents locally, records metadata in a relational database, runs an asynchronous document-processing pipeline, indexes chunks for retrieval, and exposes APIs for document management, operational health, semantic/hybrid search, grounded chat, grounded report generation, and citation normalization.

The current implementation is a monolithic Python API service with clear internal layers:

- **HTTP layer:** FastAPI routers in `app/api/routers/`.
- **Schema layer:** Pydantic request/response contracts in `app/schemas/`.
- **Service layer:** business workflow orchestration in `app/services/`.
- **Adapter layer:** replaceable OCR, embedding, summarization, categorization, and LLM providers in `app/adapters/`.
- **Persistence layer:** SQLAlchemy async ORM models and repositories backed by SQLite for local/test use or PostgreSQL + pgvector for Docker/local production-like development.
- **Migration layer:** Alembic revisions in `alembic/versions/`.

The application is intentionally local-first. Default AI integrations target local providers such as Tesseract and Ollama. Docker Compose starts PostgreSQL with pgvector and the API container; no external cloud dependency is required for local development.

## 2. Technologies and frameworks used

### 2.1 Runtime and application framework

| Area | Technology | Where it is used | Purpose |
|---|---|---|---|
| Language | Python `>=3.11` | `pyproject.toml` | Main application language. |
| Web framework | FastAPI `>=0.111.0` | `app/main.py`, `app/api/routers/*` | ASGI HTTP API framework, OpenAPI generation, dependency injection. |
| ASGI server | Uvicorn standard extras `>=0.29.0` | `pyproject.toml`, `Dockerfile` | Runs the FastAPI app in containers/local development. |
| Multipart parsing | `python-multipart >=0.0.27` | Upload endpoint | Parses multipart file uploads for `POST /documents/upload`. |
| Validation/settings | Pydantic and `pydantic-settings >=2.3.0` | `app/core/config.py`, `app/schemas/*` | Runtime configuration and request/response models. |
| HTTP client | HTTPX `>=0.27.0` | Ollama adapters and tests | Calls local Ollama HTTP APIs and supports API tests. |

### 2.2 Persistence, migrations, and database extensions

| Area | Technology | Where it is used | Purpose |
|---|---|---|---|
| ORM | SQLAlchemy asyncio `>=2.0.0` | `app/models/*`, `app/db/session.py` | Async DB engine/session and typed ORM models. |
| Migrations | Alembic `>=1.13.0` | `alembic/` | Versioned database schema upgrades/downgrades. |
| Test/local DB driver | AioSQLite `>=0.20.0` | settings/tests | SQLite async database support. |
| PostgreSQL async driver | AsyncPG `>=0.29.0` | Docker/local PostgreSQL | Async PostgreSQL connectivity. |
| Vector extension | pgvector Python package `>=0.4.2` and Docker image `pgvector/pgvector:pg16` | `app/models/document_chunk.py`, `docker-compose.yml` | Stores and queries embeddings using PostgreSQL vector capabilities. |
| Full-text search | PostgreSQL GIN indexes and `tsvector` SQL in migrations/repositories | Alembic revisions, search repositories | Hybrid retrieval text matching. |

### 2.3 AI/local-processing providers

| Capability | Provider abstraction | Implementations | Purpose |
|---|---|---|---|
| OCR | `OCRProvider` | fake, Tesseract | Extracts text from image-like documents. |
| Embeddings | `EmbeddingProvider` | fake, Ollama | Converts query/chunk text to vectors for semantic search. |
| Chat/LLM | `ChatModelProvider` | mock, Ollama | Generates answers and reports from grounded context. |
| Summarization | `SummarizationProvider` | mock, Ollama | Generates document summaries during processing. |
| Categorization | `CategorizationProvider` | rules, mock, Ollama | Assigns document categories. Rules provider is the default. |

### 2.4 Tooling and quality gates

| Tool | Command | Purpose |
|---|---|---|
| Ruff | `make lint` | Static linting over `app` and `tests`. |
| MyPy | `make typecheck` | Static type checking over `app` and `tests`. |
| Pytest | `make test` | Unit/integration tests. |
| Alembic | `make migrate` | Apply DB migrations to the configured database. |
| Docker Compose | `docker compose up --build` | Starts PostgreSQL + pgvector and the API. |

## 3. System scope and business capabilities

### 3.1 Implemented capabilities

1. **Operational API checks**
   - Root endpoint returns API identity/status.
   - Liveness checks process health.
   - Readiness checks DB connectivity with `SELECT 1`.

2. **Document management**
   - Upload PDF, text, PNG, JPEG/JPG files.
   - Validate file presence, size, MIME type, and filename safety.
   - Store original files under configured local storage.
   - Create document records and processing job records.
   - List documents with pagination, filtering, text filename query, and sorting.
   - Fetch detailed document metadata including latest processing job.
   - Patch safe metadata fields and protect manual categorization metadata.
   - Request reprocessing by creating a new job and resetting workflow status.
   - Download original file.
   - Delete document, chunks, jobs, and locally stored file when safely resolvable.

3. **Document processing**
   - Validate job/document state.
   - Run ordered pipeline stages: queued, upload received, OCR, text extraction, normalization, chunking, embeddings/indexing, summarization, categorization, completion.
   - Maintain processing status, attempts, error details, and stage history.
   - Mark both job and document failed when a stage raises.

4. **Retrieval and search**
   - Semantic vector search over indexed chunks.
   - Hybrid search combining vector similarity and PostgreSQL full-text search through Reciprocal Rank Fusion (RRF).
   - Metadata filters at document level.
   - Default search scope prefers ready documents.

5. **Grounded generation**
   - Chat asks one-shot questions over retrieved chunks.
   - Report generation builds markdown from retrieved document context.
   - Both flows require indexed context and return insufficient-context responses when no useful context is retrieved.
   - Citations are extracted from `[S1]`-style model markers or synthesized from included sources when missing.

6. **Citation utility**
   - Normalizes source inputs into citation objects.
   - Parses answer source markers.
   - Returns diagnostics for citation construction.

### 3.2 Explicit non-goals and boundaries

- The service is not a multi-service distributed system.
- File storage is local filesystem storage, not cloud object storage.
- Default model integrations are local; cloud model APIs are not part of the default architecture.
- Background worker execution is represented by services and placeholder worker package, but no external queue broker is required by the current repository.

## 4. Repository structure

```text
app/
  api/                 FastAPI router aggregation and HTTP endpoints
  adapters/            OCR, embedding, LLM, summarization, categorization providers
  core/                Runtime settings and validation
  db/                  SQLAlchemy declarative base, async engine/session dependency
  domain/              Shared status/stage vocabulary
  models/              SQLAlchemy ORM persistence models
  repositories/        Search-oriented DB access helpers
  schemas/             Pydantic request/response contracts
  services/            Application workflows and business logic
  workers/             Placeholder package for future background worker entrypoints
alembic/               Alembic environment and schema revisions
docker/initdb/         PostgreSQL initialization SQL, including pgvector extension
docs/                  Project documentation
tests/                 Pytest coverage for API, services, models, migrations, settings
```

## 5. Runtime architecture

```mermaid
flowchart TB
    Client[API client / UI] --> FastAPI[FastAPI app\napp/main.py]
    FastAPI --> Router[Top-level APIRouter\napp/api/router.py]
    Router --> Root[Root router]
    Router --> Health[Health router]
    Router --> Documents[Documents router]
    Router --> Jobs[Jobs router]
    Router --> Search[Search router]
    Router --> Chat[Chat router]
    Router --> Reports[Reports router]
    Router --> Citations[Citations router]

    Documents --> Schemas[Pydantic schemas]
    Search --> Schemas
    Chat --> Schemas
    Reports --> Schemas
    Citations --> Schemas

    Documents --> Services[Service layer]
    Search --> Services
    Chat --> Services
    Reports --> Services
    Jobs --> DBSession[Async SQLAlchemy session]
    Health --> DBEngine[Async SQLAlchemy engine]

    Services --> Adapters[Provider adapters\nTesseract/Ollama/rules/mock/fake]
    Services --> Repositories[Repositories]
    Services --> DBSession
    Repositories --> DBSession
    DBSession --> Database[(SQLite or PostgreSQL + pgvector)]
    DBEngine --> Database
    Services --> Storage[(Local filesystem storage)]
```

### 5.1 Application startup and shutdown

```mermaid
sequenceDiagram
    participant Runtime as ASGI runtime / import
    participant Main as app.main
    participant Config as app.core.config
    participant API as FastAPI app
    participant SearchRouter as app.api.routers.search

    Runtime->>Main: import app
    Main->>Config: validate_settings()
    Config->>Config: load PDA_* env vars and .env
    Config-->>Main: Settings or RuntimeError
    Main->>API: FastAPI(title, version, docs_url, openapi_url, lifespan)
    Main->>API: include api_router with configured prefix
    Runtime->>API: serve requests
    Runtime->>API: shutdown lifespan
    API->>SearchRouter: close_search_providers()
    SearchRouter->>SearchRouter: close cached embedding providers and clear cache
```

## 6. HTTP API surface

| Endpoint | Method | Router | Main flow |
|---|---:|---|---|
| `/` | GET | root | Return app title and status. |
| `/health/live` | GET | health | Return process liveness and version. |
| `/health/ready` | GET | health | Open DB connection and execute `SELECT 1`; return 503 if unavailable. |
| `/documents/upload` | POST | documents | Read multipart file with size guard, validate, store, create document/job. |
| `/documents` | GET | documents | List documents with page, page size, status/category/file type/query filters, and sort. |
| `/documents/{document_id}` | GET | documents | Return document detail and latest processing job summary. |
| `/documents/{document_id}` | PATCH | documents | Update safe metadata and category override metadata. |
| `/documents/{document_id}/reprocess` | POST | documents | Create awaiting processing job and reset document status to awaiting. |
| `/documents/{document_id}` | DELETE | documents | Delete jobs, chunks, document row, and stored file if safe. |
| `/documents/{document_id}/download` | GET | documents | Stream original stored file. |
| `/jobs/{job_id}` | GET | jobs | Return processing job status/detail. |
| `/search/semantic` | POST | search | Embed query and run vector search over chunks. |
| `/search/hybrid` | POST | search | Combine vector and full-text retrieval with RRF. |
| `/chat/ask` | POST | chat | Retrieve context, call chat model, return answer with citations. |
| `/reports/generate` | POST | reports | Retrieve context, call model with report prompt, return markdown with citations. |
| `/citations/build` | POST | citations | Build normalized citations from sources and optional answer text. |

## 7. Domain and persistence model

### 7.1 Entity relationship diagram

```mermaid
erDiagram
    DOCUMENTS ||--o{ DOCUMENT_CHUNKS : contains
    DOCUMENTS ||--o{ PROCESSING_JOBS : has
    APP_SETTINGS ||--|| APP_SETTINGS : singleton_like_configuration

    DOCUMENTS {
        uuid id PK
        string filename
        string category
        string file_type
        string mime_type
        string status
        string path
        int size
        string checksum_sha256
        json metadata_jsonb
        text extracted_text
        text summary
        string summary_model
        datetime summary_generated_at
        string summary_status
        text summary_error
        string category_source
        float category_confidence
        text category_reason
        string category_model
        datetime category_generated_at
        string category_status
        text category_error
        int chunk_count
        string embedding_model
        datetime last_indexed_at
        datetime created_at
        datetime updated_at
    }

    DOCUMENT_CHUNKS {
        uuid id PK
        uuid document_id FK
        int chunk_index
        text content
        int token_count
        int page_number
        int source_start_offset
        int source_end_offset
        json metadata_jsonb
        vector embedding
        string embedding_model
        string embedding_provider
        int embedding_dimension
        datetime embedding_created_at
        datetime created_at
        datetime updated_at
    }

    PROCESSING_JOBS {
        uuid id PK
        uuid document_id FK
        string status
        string stage
        int attempt_count
        int max_attempts
        text error_message
        json error_details_jsonb
        json stage_history_jsonb
        datetime started_at
        datetime completed_at
        datetime created_at
        datetime updated_at
    }

    APP_SETTINGS {
        uuid id PK
        string storage_path
        int max_file_size_bytes
        json allowed_file_types_jsonb
        bool ocr_enabled
        string ocr_provider
        string ocr_language
        int ocr_dpi
        int chunk_size
        int chunk_overlap
        string embedding_provider
        string embedding_model
        int embedding_dimensions
        string llm_provider
        string llm_model
        bool privacy_local_only
        bool telemetry_enabled
        json extra_settings_jsonb
        datetime created_at
        datetime updated_at
    }
```

### 7.2 Status vocabulary

```mermaid
stateDiagram-v2
    [*] --> awaiting
    awaiting --> processing: processing starts
    processing --> ready: all required stages complete
    processing --> failed: stage raises error
    failed --> awaiting: reprocess requested / retry job created
    ready --> awaiting: reprocess requested
```

`DocumentStatus` and `ProcessingJobStatus` both use `awaiting`, `processing`, `ready`, and `failed`. `ProcessingJobStage` adds fine-grained pipeline stages: `queued`, `upload_received`, `ocr`, `text_extraction`, `normalize_text`, `chunking`, `embedding`, `indexing`, `summary_generation`, `category_assignment`, `completed`, and `failed`.

## 8. Use case model

```mermaid
flowchart LR
    User((Document user))
    Operator((Operator / health checker))
    UI((Frontend / API client))

    subgraph PDA[PDA-API]
      UC1[Upload document]
      UC2[List/filter documents]
      UC3[View document detail]
      UC4[Update metadata/category]
      UC5[Download document]
      UC6[Delete document]
      UC7[Request reprocessing]
      UC8[Check job status]
      UC9[Search documents]
      UC10[Ask grounded question]
      UC11[Generate grounded report]
      UC12[Build citations]
      UC13[Check liveness/readiness]
    end

    User --> UI
    UI --> UC1
    UI --> UC2
    UI --> UC3
    UI --> UC4
    UI --> UC5
    UI --> UC6
    UI --> UC7
    UI --> UC8
    UI --> UC9
    UI --> UC10
    UI --> UC11
    UI --> UC12
    Operator --> UC13
```

### 8.1 Use case details

#### Upload document

- **Actor:** API client or frontend user.
- **Preconditions:** API is running; DB is reachable; upload file is present; MIME type is allowed; file is within configured maximum bytes.
- **Main success path:** API reads upload in chunks, rejects over-limit payloads early, validates MIME type, saves the file locally, creates `Document(status=awaiting)`, creates `ProcessingJob(status=awaiting, stage=upload_received)`, commits, and returns IDs/statuses.
- **Failure paths:** Empty file returns 400; too large returns 413; unsupported MIME type returns 415; DB/storage failures propagate as server errors.

#### Process document

- **Actor:** Internal service/worker invocation.
- **Preconditions:** Job exists, is awaiting, is in `queued` or `upload_received`, document is awaiting or failed, and retry attempts are not exhausted.
- **Main success path:** Mark job/document processing, run all pipeline stages, persist stage history entries, mark job/document ready, set job stage completed.
- **Failure path:** On any stage exception, mark document failed, job failed, stage failed, error message/details persisted, and the exception is re-raised.

#### Search documents

- **Actor:** API client or higher-level chat/report service.
- **Preconditions:** Chunks exist and have embeddings for vector search; embedding provider is configured and available for query embedding.
- **Main success path:** Validate request, embed query, apply filters, retrieve ranked chunks, return result diagnostics.
- **Failure paths:** Missing configuration returns service configuration error; unavailable provider maps to 503; unexpected search error maps to 500.

#### Ask grounded question

- **Actor:** API client or frontend user.
- **Preconditions:** Documents have been processed and indexed; model provider is configured.
- **Main success path:** Retrieve chunks using hybrid or semantic strategy, build bounded context, call chat model, parse or synthesize citations, return answer, citations, retrieval diagnostics, model diagnostics, and usage estimates.
- **Fallback path:** If no context is retrieved or all context is excluded, return a fixed insufficient-context answer without calling the model.

#### Generate grounded report

- **Actor:** API client or frontend user.
- **Preconditions:** Same as grounded chat.
- **Main success path:** Retrieve chunks by topic, build report-style grouped context, call model, parse or synthesize citations, return markdown report and diagnostics.
- **Fallback path:** If no context is available, return an insufficient-context markdown report.

## 9. Detailed process and flow diagrams

### 9.1 Upload flow

```mermaid
sequenceDiagram
    participant Client
    participant Router as documents.upload_document
    participant Reader as read_upload_limited
    participant Ingestion as ingest_upload
    participant Storage as local file storage
    participant DB as AsyncSession

    Client->>Router: POST /documents/upload multipart file
    Router->>Reader: read chunks up to max_file_size_bytes
    alt exceeds max bytes
        Reader-->>Client: HTTP 413
    else within limit
        Reader-->>Router: bytes
    end
    Router->>Ingestion: filename, content_type, bytes, settings
    Ingestion->>Ingestion: reject empty bytes with 400
    Ingestion->>Ingestion: normalize and validate MIME type
    alt MIME unsupported
        Ingestion-->>Client: HTTP 415
    else MIME allowed
        Ingestion->>Storage: sanitize filename and save bytes
        Storage-->>Ingestion: stored path and SHA-256 checksum
        Ingestion->>DB: add Document(awaiting)
        DB-->>Ingestion: document.id after flush
        Ingestion->>DB: add ProcessingJob(awaiting, upload_received)
        Ingestion->>DB: commit and refresh rows
        Ingestion-->>Router: document, job
        Router-->>Client: 201 UploadResponse
    end
```

### 9.2 Document processing orchestration flow

```mermaid
flowchart TD
    Start([process_job called]) --> Load[Load ProcessingJob]
    Load --> Exists{Job exists?}
    Exists -- No --> NotFound[Raise ProcessingJobNotFoundError]
    Exists -- Yes --> Doc[Get associated Document]
    Doc --> Validate{Processable?}
    Validate -- No --> StateError[Raise ProcessingOrchestratorStateError]
    Validate -- Yes --> MarkProcessing[Set job/document processing, increment attempt, clear errors/history]
    MarkProcessing --> Queued[queued stage]
    Queued --> UploadReceived[upload_received stage]
    UploadReceived --> OCR{Document requires OCR?}
    OCR -- Yes --> RunOCR[OCRService.extract_text_for_document]
    OCR -- No --> SkipOCR[Record OCR skipped]
    RunOCR --> Extract[Text extraction stage]
    SkipOCR --> Extract
    Extract --> Normalize[Normalize extracted text]
    Normalize --> Chunk[Chunk normalized/extracted text]
    Chunk --> Embed[Generate embeddings]
    Embed --> Index[Indexing bookkeeping]
    Index --> Summary[Generate summary]
    Summary --> Category[Assign category]
    Category --> Ready[Set document/job ready and stage completed]
    Ready --> End([Commit success])

    Queued -. any exception .-> Fail
    UploadReceived -. any exception .-> Fail
    RunOCR -. any exception .-> Fail
    Extract -. any exception .-> Fail
    Normalize -. any exception .-> Fail
    Chunk -. any exception .-> Fail
    Embed -. any exception .-> Fail
    Index -. any exception .-> Fail
    Summary -. any exception .-> Fail
    Category -. any exception .-> Fail
    Fail[Set document failed, job failed, stage failed, persist error details] --> Raise[Re-raise original exception]
```

### 9.3 Processing job stage state diagram

```mermaid
stateDiagram-v2
    [*] --> queued
    queued --> upload_received: upload accepted or queued job begins
    upload_received --> ocr: job runner advances
    ocr --> text_extraction: OCR completed or skipped
    text_extraction --> normalize_text: text loaded/extracted
    normalize_text --> chunking: normalization completed
    chunking --> embedding: chunks persisted
    embedding --> indexing: vectors persisted
    indexing --> summary_generation: indexing bookkeeping done
    summary_generation --> category_assignment: summary ready/skipped/failed per service policy
    category_assignment --> completed: category ready/skipped/fallback
    completed --> [*]

    queued --> failed: stage error
    upload_received --> failed: stage error
    ocr --> failed: OCR error
    text_extraction --> failed: extraction error
    normalize_text --> failed: normalization error
    chunking --> failed: chunking error
    embedding --> failed: embedding/provider/vector error
    indexing --> failed: indexing error
    summary_generation --> failed: unexpected summary stage error
    category_assignment --> failed: unexpected category stage error
```

### 9.4 OCR and text extraction decision flow

```mermaid
flowchart TD
    Start[Document enters OCR stage] --> NeedOCR{document_requires_ocr?}
    NeedOCR -- Yes --> OCRProvider[Resolve OCR provider]
    OCRProvider --> OCRRun[Run OCR against stored file]
    OCRRun --> PersistText[Persist extracted_text and OCR metadata]
    PersistText --> ExtractionStage[Enter text_extraction stage]
    NeedOCR -- No --> Skip[Append OCR skipped stage history]
    Skip --> ExtractionStage
    ExtractionStage --> ExistingText{extracted_text present?}
    ExistingText -- Yes --> Done[Record char_count]
    ExistingText -- No --> ResolvePath[Resolve stored file under storage root]
    ResolvePath --> Extractor[extract_text_from_file by MIME/filename]
    Extractor --> SaveText[Set document.extracted_text]
    SaveText --> Done
```

### 9.5 Text normalization flow

```mermaid
flowchart TD
    Start[normalize_text stage] --> Enabled{PDA_TEXT_NORMALIZATION_ENABLED?}
    Enabled -- No --> Skipped[Record skipped=true]
    Enabled -- Yes --> HasText{extracted_text is present and non-empty?}
    HasText -- No --> Strict{fail_on_empty_output?}
    Strict -- Yes --> EmptyError[Raise normalization empty input/output error]
    Strict -- No --> SkipEmpty[Record skipped reason]
    HasText -- Yes --> Options[Build TextNormalizationOptions]
    Options --> Normalize[Normalize Unicode, line endings, control chars, spacing, dehyphenation, blank lines]
    Normalize --> EmptyOutput{output empty?}
    EmptyOutput -- Yes --> StrictOut{fail_on_empty_output?}
    StrictOut -- Yes --> EmptyOutError[Raise TextNormalizationEmptyOutputError]
    StrictOut -- No --> PersistEmpty[Persist empty normalized text]
    EmptyOutput -- No --> Persist[Replace document.extracted_text with normalized text]
    Persist --> Metadata[Store normalization metadata/warnings]
    PersistEmpty --> Metadata
    Metadata --> Done[Record completed details]
```

### 9.6 Chunking flow

```mermaid
flowchart TD
    Start[chunk_document] --> LoadDoc[Load Document]
    LoadDoc --> Text{extracted_text has non-whitespace text?}
    Text -- No --> Empty[Raise ChunkingEmptyTextError]
    Text -- Yes --> Settings[Load AppSettings or defaults]
    Settings --> Validate[Validate chunk_size > 0, overlap >= 0, overlap < size]
    Validate --> Split[Normalize line endings and split into overlapping chunks]
    Split --> Boundaries[Prefer natural boundaries near chunk size]
    Boundaries --> Offsets[Calculate source offsets/page metadata]
    Offsets --> DeleteOld[Delete existing chunks for document]
    DeleteOld --> Insert[Insert DocumentChunk rows ordered by chunk_index]
    Insert --> UpdateDoc[Set chunk_count]
    UpdateDoc --> Done[Return chunking result]
```

### 9.7 Embedding and indexing flow

```mermaid
flowchart TD
    Start[generate_embeddings_for_document] --> Load[Load document and optional job]
    Load --> Chunks[Load ordered chunks]
    Chunks --> HasChunks{Any non-empty chunks?}
    HasChunks -- No --> NoChunks[Raise NoChunksToEmbedError]
    HasChunks -- Yes --> Runtime[Resolve provider, model, dimensions, batch size, truncate]
    Runtime --> Provider{Provider exists?}
    Provider -- No --> Unknown[Raise UnknownEmbeddingProviderError]
    Provider -- Yes --> Batch[For each chunk batch]
    Batch --> Embed[provider.embed_texts]
    Embed --> CountCheck{result count and indices match batch?}
    CountCheck -- No --> ProviderError[Raise provider response error]
    CountCheck -- Yes --> Dimension{dimensions match?}
    Dimension -- No --> DimError[Raise dimension mismatch]
    Dimension -- Yes --> ValidateVectors[Validate finite numeric vector values]
    ValidateVectors --> Persist[Persist vector/model/provider/dimension timestamps on chunks]
    Persist --> DocMeta[Set document embedding_model, chunk_count, last_indexed_at]
    DocMeta --> Done[Return embedding generation result]
```

### 9.8 Semantic search flow

```mermaid
sequenceDiagram
    participant Client
    participant Router as /search/semantic
    participant Service as SearchService
    participant Provider as EmbeddingProvider
    participant Repo as VectorSearchRepository
    participant DB as Database

    Client->>Router: SemanticSearchRequest(query, top_k, filters)
    Router->>Service: semantic_search(request)
    Service->>Provider: embed query
    alt provider unavailable
        Provider-->>Service: unavailable/error
        Service-->>Router: EmbeddingProviderNotAvailableError
        Router-->>Client: HTTP 503
    else provider ok
        Provider-->>Service: query vector
        Service->>Repo: search_similar_chunks(vector, filters, limit)
        Repo->>DB: vector distance query or generic fallback
        DB-->>Repo: candidate chunks joined to documents
        Repo-->>Service: ranked results
        Service-->>Router: SemanticSearchResponse
        Router-->>Client: 200 results + diagnostics
    end
```

### 9.9 Hybrid search flow

```mermaid
flowchart TD
    Start[POST /search/hybrid] --> Validate[Validate request and filters]
    Validate --> Embed[Embed query for vector path]
    Embed --> VectorPath[Run vector similarity search]
    Validate --> TextPath[Run PostgreSQL full-text search path]
    VectorPath --> Fuse[Reciprocal Rank Fusion]
    TextPath --> Fuse
    Fuse --> Deduplicate[Merge duplicate chunk hits]
    Deduplicate --> Rank[Sort by fused score and deterministic tie-breakers]
    Rank --> Limit[Apply top_k]
    Limit --> Response[Return HybridSearchResponse with diagnostics]
```

### 9.10 Grounded chat flow

```mermaid
sequenceDiagram
    participant Client
    participant Router as /chat/ask
    participant Chat as ChatService
    participant Search as Search/HybridSearchService
    participant Context as ContextBuilderService
    participant LLM as ChatModelProvider
    participant Citations as CitationBuilder

    Client->>Router: ChatAskRequest(question, retrieval_strategy, filters)
    Router->>Chat: ask_question(request)
    Chat->>Search: retrieve relevant chunks
    Search-->>Chat: ranked chunk results
    Chat->>Context: build_context(results, query, max_tokens, raw style)
    Context-->>Chat: context text + source map
    alt no included context
        Chat-->>Router: insufficient-context response
        Router-->>Client: 200 answer without model call
    else context available
        Chat->>LLM: generate(system/user messages, model, temperature, max_tokens)
        alt model unavailable
            LLM-->>Chat: unavailable error
            Chat-->>Router: provider unavailable
            Router-->>Client: HTTP 503
        else model returns answer
            LLM-->>Chat: answer text
            Chat->>Citations: extract markers and build citations
            Citations-->>Chat: citations + diagnostics
            Chat-->>Router: ChatAskResponse
            Router-->>Client: 200 answer, citations, diagnostics
        end
    end
```

### 9.11 Grounded report flow

```mermaid
flowchart TD
    Start[POST /reports/generate] --> Retrieve[Retrieve chunks by topic using semantic or hybrid strategy]
    Retrieve --> Context[Build report-style context grouped by document]
    Context --> HasContext{Included chunks > 0?}
    HasContext -- No --> Insufficient[Return insufficient-context markdown]
    HasContext -- Yes --> Prompt[Build report system/user messages]
    Prompt --> Generate[Call configured chat/model provider]
    Generate --> CitationMarkers[Extract source markers]
    CitationMarkers --> BuildCitations[Build citations from context sources and retrieval results]
    BuildCitations --> Missing{No citations but sources exist?}
    Missing -- Yes --> FallbackCites[Return top included sources with warning]
    Missing -- No --> Response[Return markdown, citations, retrieval/model/usage diagnostics]
    FallbackCites --> Response
```

### 9.12 Citation builder flow

```mermaid
flowchart TD
    Start[POST /citations/build] --> Convert[Convert CitationSourceInput to ContextSource]
    Convert --> Lookup[Create retrieval-result compatible objects]
    Lookup --> Markers{answer_text provided?}
    Markers -- Yes --> Parse[Parse source markers like S1]
    Markers -- No --> IncludePolicy[Use include_uncited_sources policy]
    Parse --> Normalize[Build normalized citation objects]
    IncludePolicy --> Normalize
    Normalize --> Excerpts[Resolve excerpt from included text range, text, or explicit excerpt]
    Excerpts --> Diagnostics[Attach diagnostics]
    Diagnostics --> Response[CitationBuildResponse]
```

### 9.13 Delete flow

```mermaid
sequenceDiagram
    participant Client
    participant Router as DELETE /documents/{id}
    participant DB as AsyncSession
    participant Storage as Local storage

    Client->>Router: DELETE document id
    Router->>DB: get Document
    alt not found
        Router-->>Client: HTTP 404
    else found
        Router->>Storage: resolve path safely under storage root
        Router->>DB: delete ProcessingJob rows
        Router->>DB: delete DocumentChunk rows
        Router->>DB: delete Document row
        Router->>DB: commit
        alt stored file exists and path safe
            Router->>Storage: unlink file
        end
        Router-->>Client: HTTP 204
    end
```

## 10. Component/class overview

```mermaid
classDiagram
    class FastAPIApp {
      +create_app() FastAPI
      +lifespan closes providers
    }
    class Settings {
      +app_name
      +api_prefix
      +database_url
      +storage_path
      +allowed_file_types
      +embedding_provider
      +summarization_provider
      +categorization_provider
      +text_normalization_enabled
    }
    class Document {
      +UUID id
      +filename
      +status
      +extracted_text
      +summary
      +category
      +chunk_count
    }
    class DocumentChunk {
      +UUID id
      +document_id
      +chunk_index
      +content
      +embedding
    }
    class ProcessingJob {
      +UUID id
      +document_id
      +status
      +stage
      +attempt_count
      +stage_history_jsonb
    }
    class SearchService {
      +semantic_search()
    }
    class HybridSearchService {
      +hybrid_search()
    }
    class ChatService {
      +ask_question()
    }
    class ReportService {
      +generate_report()
    }
    class ProcessingOrchestrator {
      +process_job()
    }
    class ProviderAdapters {
      +OCRProvider
      +EmbeddingProvider
      +ChatModelProvider
      +SummarizationProvider
      +CategorizationProvider
    }

    FastAPIApp --> Settings
    FastAPIApp --> SearchService : closes cached providers
    Document "1" --> "many" DocumentChunk
    Document "1" --> "many" ProcessingJob
    ProcessingOrchestrator --> Document
    ProcessingOrchestrator --> ProcessingJob
    ProcessingOrchestrator --> ProviderAdapters
    SearchService --> ProviderAdapters
    HybridSearchService --> SearchService
    ChatService --> HybridSearchService
    ChatService --> SearchService
    ChatService --> ProviderAdapters
    ReportService --> HybridSearchService
    ReportService --> SearchService
    ReportService --> ProviderAdapters
```

## 11. Configuration model

Runtime configuration is loaded from defaults, `.env`, and environment variables prefixed with `PDA_`. Important settings include:

| Setting | Default | Meaning |
|---|---:|---|
| `PDA_APP_NAME` | `PDA API` | FastAPI title. |
| `PDA_APP_VERSION` | `0.1.0` | API version and liveness response version. |
| `PDA_API_PREFIX` | `/` normalized to empty prefix | Prefix applied to all routers. |
| `PDA_DATABASE_URL` | `sqlite+aiosqlite:///./pda.db` | Async SQLAlchemy database URL. |
| `PDA_STORAGE_PATH` | `./storage` | Local file storage root. |
| `PDA_ALLOWED_FILE_TYPES` | PDF, text, PNG, JPEG/JPG | MIME allow-list. |
| `PDA_MAX_FILE_SIZE_BYTES` | 10 MiB | Upload maximum. |
| `PDA_OCR_PROVIDER` | `tesseract` | OCR provider name. |
| `PDA_EMBEDDING_PROVIDER` | `ollama` | Embedding provider name. |
| `PDA_EMBEDDING_MODEL` | `all-minilm` | Embedding model name. |
| `PDA_EMBEDDING_DIMENSIONS` | `1536` | Required vector dimension. |
| `PDA_MODEL_PROVIDER` | `local` | LLM provider selector. |
| `PDA_MODEL_NAME` | `llama3.1:8b-instruct` | Chat/report model. |
| `PDA_SUMMARIZATION_PROVIDER` | `ollama` | Summarization provider. |
| `PDA_CATEGORIZATION_PROVIDER` | `rules` | Categorization provider. |
| `PDA_TEXT_NORMALIZATION_ENABLED` | `true` | Enables normalization pipeline stage. |
| `PDA_OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama base URL. |

Configuration validators normalize comma-separated lists, enforce positive numeric limits, normalize API prefix formatting, and currently allow only cosine embedding distance.

## 12. Database and migration design

The schema is managed through Alembic. The migration history creates:

1. An initial empty schema baseline.
2. `documents` table.
3. `processing_jobs` table.
4. `document_chunks` table.
5. App settings table.
6. Document/chunk search, vector, full-text, category, and summary metadata extensions.

SQLite compatibility is preserved for tests by using JSON vector fallback where needed. PostgreSQL deployments use pgvector and GIN indexing for retrieval-oriented workflows.

## 13. Error handling and reliability behavior

- FastAPI routers translate known provider/configuration/service exceptions into HTTP 503 or 500 responses.
- Upload validation uses explicit HTTP status codes: 400 empty file, 413 oversized, 415 unsupported MIME type.
- Document/job lookups return 404 when missing.
- Processing orchestration atomically records stage history and failure details.
- Embedding generation validates provider response counts, batch indices, model consistency, vector dimensions, and vector numeric validity before mutating chunks.
- Search provider instances are cached by relevant settings and closed during app shutdown.

## 14. Privacy and local-first considerations

- Uploaded documents are stored on local disk under the configured storage path.
- Database content is stored in local SQLite by default or local Docker PostgreSQL in Compose.
- Default OCR, embeddings, summarization, chat, and categorization integrations are local or mock/rules-based.
- Services avoid logging raw document text. Logs focus on IDs, counts, statuses, durations, provider names, and sanitized error reasons.
- The persisted `AppSettings` model contains `privacy_local_only` defaulting to true and `telemetry_enabled` defaulting to false.

## 15. Development and validation workflow

Canonical commands:

```bash
make setup
make lint
make typecheck
make test
make migrate
```

Docker workflow:

```bash
cp .env.example .env
docker compose up --build
```

PostgreSQL-only workflow:

```bash
docker compose up db
PDA_DATABASE_URL=postgresql+asyncpg://pda:pda_dev@localhost:5432/pda uvicorn app.main:app --reload
```

## 16. Logical end-to-end scenarios

### 16.1 Upload to searchable document

1. Client uploads a supported file.
2. API stores original bytes and creates an awaiting document/job.
3. Job runner invokes processing orchestrator.
4. OCR runs only when the document type requires OCR.
5. Text extraction ensures text exists for non-OCR documents.
6. Normalization rewrites extracted text into deterministic clean text.
7. Chunking replaces stale chunks with new ordered chunks.
8. Embeddings are generated batch-by-batch and persisted.
9. Summary generation writes summary fields on the document.
10. Categorization writes category metadata unless a manual category is protected.
11. Job/document become ready.
12. Search, chat, and report flows can retrieve the document chunks.

### 16.2 Manual category override

1. Client patches a document with `category`.
2. API validates the category against the allowed vocabulary.
3. Document category source becomes `manual`.
4. Category status becomes `ready`.
5. Provider-generated confidence/reason/model/error metadata is cleared.
6. Later automatic categorization skips this document unless forced.

### 16.3 Reprocessing

1. Client posts to `/documents/{id}/reprocess` with optional force/reason.
2. API creates a new awaiting job at queued stage.
3. Document status resets to awaiting.
4. Existing storage metadata is preserved.
5. A worker/service can process the new job and replace derived state such as normalized text/chunks/embeddings.

### 16.4 Grounded answer with citations

1. Client asks a question.
2. Chat service runs semantic or hybrid retrieval.
3. Context builder selects unique chunks under a token/character budget and assigns source IDs.
4. Model receives instructions to answer from the provided sources.
5. Citation builder parses source markers or falls back to top sources.
6. Response includes answer, citations, retrieval diagnostics, model diagnostics, and usage estimates.

## 17. Future extension points

- Add a real worker process or queue runner under `app/workers/`.
- Add new adapters by implementing provider protocols in `app/adapters/*/base.py`.
- Add new domain models and migrations only when required by a feature task.
- Extend file storage behind the existing storage service boundary if non-local storage is explicitly required.
- Add smoke tests when operational smoke coverage exists; `make smoke` is intentionally reserved until then.
