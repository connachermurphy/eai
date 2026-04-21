# CLAUDE.md

## Tools

- Always use `uv` to run Python scripts (e.g., `uv run script.py`), never `python` or `python3` directly.

## Approach

- Do not take simplifying shortcuts (e.g., picking the first match, dropping data) when a more robust approach is available. If a problem requires symmetric treatment across data sources, apply the full solution to all sides upfront rather than handling one side and deferring the other.
