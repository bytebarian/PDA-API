# Eta – Contract tests, smoke flow, and demonstration harness

## Repository
Both repositories

## Objective
Deliver this Foundation slice as an independent Codex task with tests and deterministic verification.

## Implementation scope
- Backend contract tests
- Fixture corpus
- Processing integration tests
- Retrieval/chat/report tests
- Playwright setup
- End-to-end smoke flow
- make smoke in both repos

## Files and directories that may be touched
- `tests`
- `e2e`
- `fixtures`
- `Makefile`
- `AGENTS.md`

## Out of scope
- Do not redesign the existing UI unless this task explicitly says so.
- Do not add public cloud dependencies for tests.
- Do not remove privacy-first local execution assumptions.

## Commands Codex must run
- `make smoke`
- `make test`
- `npm run smoke`

## Definition of Done
- Single command validates upload to report flow
- No external internet required
- Failures produce useful logs
- Codex can verify completion deterministically

## Acceptance notes
Implementation is complete only when the code, tests, and documentation are committed together and the required commands pass in a clean local environment.
