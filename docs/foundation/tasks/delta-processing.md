# Delta – Processing pipeline, OCR/text extraction, and status orchestration

## Repository
bytebarian/PDA-API

## Objective
Deliver this Foundation slice as an independent Codex task with tests and deterministic verification.

## Implementation scope
- ProcessingOrchestrator
- JobRunner abstraction
- TXT/MD extraction
- PDF/DOCX extraction
- Image OCR adapter
- Text normalization
- Chunking using settings
- Embedding adapter
- Summary and category generation
- Status transitions and job history

## Files and directories that may be touched
- `app/services/processing`
- `app/adapters`
- `app/workers`
- `tests`

## Out of scope
- Do not redesign the existing UI unless this task explicitly says so.
- Do not add public cloud dependencies for tests.
- Do not remove privacy-first local execution assumptions.

## Commands Codex must run
- `make test`
- `make smoke`

## Definition of Done
- Uploaded text fixture becomes ready
- Chunks and embeddings are stored
- Failures mark document failed with reason
- Job endpoint exposes stage history

## Acceptance notes
Implementation is complete only when the code, tests, and documentation are committed together and the required commands pass in a clean local environment.
