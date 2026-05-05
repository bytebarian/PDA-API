# PDA — Personal Documents Assistant

PDA (Personal Documents Assistant) is a privacy-first, locally hosted document management and AI-assisted analysis solution for an individual, household, or shared home workspace.

Its main goal is to securely store important personal documents, make them easy to retrieve, and enable intelligent interaction with their contents through AI-powered search, Q&A, and reporting.

## Project goals

- securely store critical personal documents
- prevent loss or damage of important records
- enable fast retrieval by filename, metadata, or natural-language description
- support AI-based questions and answers grounded in document content
- provide deeper analysis and reporting across documents
- preserve privacy by prioritizing local processing and offline-capable AI models

## Main capabilities

### Document management
- single-file upload
- support for PDF, TXT, MD, DOC, DOCX, JPEG, PNG
- local file storage on disk or network share
- metadata capture
- rename and delete operations
- document re-embedding with different chunking or embedding settings

### Search and retrieval
- exact filename search
- partial filename search
- metadata search
- contextual AI-assisted search over stored documents

### AI features
- chat over documents
- concise answers with citations
- document summarization
- automatic categorization
- research and reporting
- comparative analysis between documents

## High-level architecture

PDA is designed as a modular solution consisting of:

- **Frontend application** — simple, modern, responsive UI
- **Backend application** — orchestration of document workflows and APIs
- **AI module** — chat, reporting, retrieval, and model orchestration
- **Documents database** — metadata and file references
- **Vector database** — embeddings and retrieval for RAG

## Processing flow

1. User uploads a document
2. System stores the original file
3. Metadata is captured and saved
4. Scheduler triggers OCR / text extraction
5. Extracted text is normalized and chunked
6. Embeddings are generated
7. Vectors are stored for retrieval
8. The system summarizes and categorizes the document
9. The document becomes available for search, chat, and reporting

## Privacy and security

PDA is designed as a privacy-first solution.

- intended to run on a home server
- available within a local home network
- should work offline with local AI models
- sensitive data must not be sent to public AI services unless confidential information has been removed first

## Example use cases

- find the latest contract with a supplier
- check the notice period in an employment contract
- compare two offers
- detect conflicts between agreements
- generate reports based on contracts and official documents

## Local development with Docker

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) with the Compose plugin (v2+)

### Start the stack

```bash
# Copy the example environment file (first time only)
cp .env.example .env

# Build and start PostgreSQL + API
docker compose up --build
```

The API will be available at <http://localhost:8000>.  
PostgreSQL will be reachable on `localhost:5432`.

### Stop the stack

```bash
docker compose down
```

To also remove the database volume (destructive):

```bash
docker compose down -v
```

### Verify pgvector is available

After the stack is running you can confirm the extension is present (it is enabled automatically on first start):

```bash
docker compose exec db sh -lc 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT extname FROM pg_extension WHERE extname = '\''vector'\'';"'
```

### Running only PostgreSQL (no API container)

If you prefer to run the API outside Docker (e.g. with `uvicorn` directly), you can start just the database:

```bash
docker compose up db
```

Then replace the `PDA_DATABASE_URL` line in your local `.env` with the PostgreSQL URL:

```
PDA_DATABASE_URL=postgresql+asyncpg://pda:pda_dev@localhost:5432/pda
```

> Replace `pda`, `pda_dev`, and the database name with the values you set for `POSTGRES_USER`, `POSTGRES_PASSWORD`, and `POSTGRES_DB` if you changed the defaults.

## Project status

This project is currently in the analysis and design phase.  
Business, technical, and scope documentation has already been prepared, and the next steps include detailed architecture, implementation planning, and MVP scoping.

## License

This project is licensed under the MIT License.
