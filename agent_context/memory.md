# Project Memory

## Purpose

This repository is intended to collect and organize materials related to CTA, deep learning, market timing, regime detection, momentum, and systematic portfolio research.

## Current State

- The project currently contains a `papers/` directory with research PDFs.
- The repository is intended to be synchronized through GitHub at `https://github.com/StephBay30/dl-for-cta.git`.
- Persistent assistant/project context is stored in this file.
- Cross-platform line endings are controlled by `.gitattributes`.
- In this Windows/Codex environment, `.git` may be blocked by ACL sandboxing; `.gitdata/` can be used as a local fallback git metadata directory if needed.

## Working Rules

- Keep important project context in `agent_context/memory.md`.
- Keep assistant-facing maintenance instructions in `agent_context/AGENTS.md`.
- Update this memory file when important project decisions, TODOs, milestones, or setup details change.
- Do not write secrets, passwords, API keys, tokens, or private credentials into this repository.

## Decisions

- Use GitHub as the source of truth for syncing the project across machines.
- Store project memory in Markdown files inside `agent_context/`.
- Use LF line endings for text files so the repository behaves consistently on Windows and Linux.
- Ignore local git metadata directories via `.gitignore`.

## TODO

- Finish initializing the local git repository and push it to GitHub.
- Consider adding a README that explains the project purpose and paper organization.
- Consider adding metadata for papers, such as title, topic, source, year, and notes.

## Notes

- A previous attempt initialized `.git`, but `git add .` was blocked by a Windows permission issue on `.git/index.lock`.
