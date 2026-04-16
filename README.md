# eai
Combining various data sources related to the economics of AI

## Data sources

- Anthropic Economic Index: https://huggingface.co/datasets/Anthropic/EconomicIndex
  - Download: `uv run anthropic/download.py` (update `RELEASE` in the script when a new release drops).
  - Inspect: `uv run anthropic/clean.py` prints the head of each downloaded CSV.
- O*NET task statements: https://www.onetcenter.org/dictionary/20.1/excel/task_statements.html

## Development

### Linting

Check for lint issues:

```bash
uvx ruff check .
uvx ruff format --check .
```

Auto-fix lint issues:

```bash
uvx ruff check --fix .
uvx ruff format .
```

If `uvx` is unavailable, use `ruff` directly (after `pip install ruff`).
