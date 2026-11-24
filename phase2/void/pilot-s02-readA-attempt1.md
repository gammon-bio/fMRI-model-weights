# Void attempt: pilot-s02 read task, Opus 5.5, session_01Y4tqQu6vEMWSATjFN9bxPv (2026-11-04)

**Harness failure, not an agent failure.** The weight files (`*.npz`) were excluded by `.gitignore`,
so the agent received no weights. The agent noticed, pushed a labelled guess to
`claude/phase2-pilot-s02-readA-opus`, and flagged the episode invalid.
- Cost reported by the session: $0.52.
- Not ingested, not scored.

**Fix:** `.gitignore` exception for `phase2/handoff/**/*.npz`. The episode is re-run with a fresh
agent session on the same hand-off. No episode information reached the first agent beyond TASK.md
and ARCH.md.
