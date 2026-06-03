# Agent Rules

Elephant Never Forgets should be useful for agents that need repo-local memory.

Manual edits should be made with a focused patch. Test evidence should be written
down near the issue or handoff note. Generated cache state belongs in `.enf/` and
should not be committed.

When a search result cites this file, the user should see a repo-relative path and
line range in plain output. JSON output should include the same path, score
components, profile hash, mode, and level.

