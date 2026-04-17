# eai
Combining various data sources related to the economics of AI

## Data sources

- Anthropic Economic Index: https://huggingface.co/datasets/Anthropic/EconomicIndex
  - Reference code: https://huggingface.co/datasets/Anthropic/EconomicIndex/tree/main/release_2025_09_15
  - Download: `uv run anthropic/download.py` (update `RELEASE` in the script when a new release drops).
  - Clean: `uv run anthropic/clean.py` filters to GLOBAL + `onet_task::collaboration`, pivots wide (one column per collaboration type), and merges onto O*NET task statements. Outputs `data/<release>/aei_cleaned_claude_ai.csv`.
- O*NET task statements: https://www.onetcenter.org/dictionary/20.1/excel/task_statements.html
- OEWS (Occupational Employment and Wage Statistics): https://www.bls.gov/oes/tables.htm

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
