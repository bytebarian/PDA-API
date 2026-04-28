# Zeta – Frontend bridge with zero visual redesign

## Repository
bytebarian/pda-personal-documen

## Objective
Deliver this Foundation slice as an independent Codex task with tests and deterministic verification.

## Implementation scope
- Add API client
- Add React Query provider
- Replace upload mock
- Replace document local state
- Replace chat mock
- Replace report mock
- Persist settings through API
- Poll processing status

## Files and directories that may be touched
- `src/lib/api`
- `src/lib/hooks`
- `src/lib/mappers`
- `src/providers`
- `src/App.tsx`

## Out of scope
- Do not redesign the existing UI unless this task explicitly says so.
- Do not add public cloud dependencies for tests.
- Do not remove privacy-first local execution assumptions.

## Commands Codex must run
- `npm install`
- `npm run lint`
- `npm run typecheck`
- `npm test`

## Definition of Done
- Existing six views remain visually intact
- Upload calls backend
- Documents render live backend data
- Chat and reports call backend
- Settings persist

## Acceptance notes
Implementation is complete only when the code, tests, and documentation are committed together and the required commands pass in a clean local environment.
