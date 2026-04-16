# eai
Combining various data sources related to the economics of AI

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
