# Agent Instructions

This project keeps persistent working context in `agent_context/memory.md`.

When working on this project:

1. Read `agent_context/memory.md` before making non-trivial changes.
2. Update `agent_context/memory.md` whenever something important changes, including:
   - project goals or scope
   - setup or run instructions
   - important decisions
   - completed milestones
   - known problems, blockers, or TODOs
   - file organization conventions
3. Keep memory entries concise and factual.
4. Prefer updating the existing sections instead of adding duplicate notes.
5. Do not store secrets, passwords, API keys, tokens, or private credentials in memory.
6. Long-running commands and pipeline stages must emit useful `logging.info` progress:
   - log start and finish of each CLI command;
   - log input/output paths and row counts for data/feature steps;
   - log CPD symbol/window progress periodically;
   - log training candidate, epoch, train loss, validation after-cost Sharpe, threshold, and selected best model;
   - log backtest checkpoint, test rows, metrics path, and positions path.
   Use `logging.warning` for recoverable fallbacks and `logging.error` only when a command is about to fail.

The goal is to make the repository itself carry enough context that work can continue across machines and future assistant sessions.
