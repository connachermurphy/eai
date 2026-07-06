# eai

Combining data sources related to the economics of AI, occupational tasks,
employment, wages, and usage/exposure measures.

## Repository Layout

- `pipeline/`: data download, conversion, and build scripts, run in the order
  shown below.
- `analysis/`: analysis scripts that consume the built panels.
- `eai/`: shared package code (AEI loading, plotting, logging, codebook
  writing).
- `input/`: raw source data. Large or re-downloadable files are gitignored.
- `output/`: generated data, figures, and reports. Ignored by default, with a
  small allowlist of committed review artifacts (see File Storage below).
- `docs/eai.md`: detailed data and measurement notes.

## Data Sources

- Anthropic Economic Index (AEI): task-level usage releases from Hugging Face.
  The repo currently handles September 2025, January 2026, and March 2026
  releases for both Claude.ai and first-party API.
- OpenAI Signals release: local CSVs under
  `input/oai_260616/data-download-csv`, including monthly O*NET IWA shares for
  all U.S. messages and work-related U.S. messages.
- O*NET 20.1 task statements: used for the AEI task-to-occupation pipeline.
- O*NET 30.2 database: used for Intermediate Work Activity (IWA), Detailed Work
  Activity (DWA), and task-to-DWA mappings.
- SOC 2010 to SOC 2018 crosswalk: used to build direct crosswalk edges and
  connected-component occupation groups.
- OEWS national files: converted CSVs under `output/oews`. The older
  occupational-characteristics outputs default to May 2022 OEWS; newer wage and
  OpenAI apportionment analyses use May 2024 OEWS.
- Eloundou et al. occupation-level exposure data from "GPTs are GPTs".

## Main Commands

Use `uv` for Python scripts. A full rebuild, in dependency order:

```bash
uv run pipeline/convert_onet.py
uv run pipeline/build_crosswalk.py
uv run pipeline/convert_oews.py
uv run pipeline/download_aei.py
uv run pipeline/download_eloundou.py
uv run pipeline/clean_aei.py
uv run pipeline/panel.py
```

Build the main occupational-characteristics panels:

```bash
uv run pipeline/occupational_characteristics.py
uv run pipeline/occupational_characteristics.py --oews-year 2024 --aei-only
```

Build the O*NET 30.2 IWA mapping and the OpenAI IWA/OEWS occupation measures:

```bash
uv run pipeline/build_iwa_onet_mapping.py
uv run pipeline/build_openai_iwa_oews.py
```

Run analysis scripts:

```bash
uv run analysis/net_automation_wages.py
uv run analysis/openai_iwa_wages.py
uv run analysis/cross_release_correlation.py
```

## Codebooks

Every output directory contains a `codebook.md` documenting the variables in
its generated files. The codebooks are written by the generating scripts
themselves (via `eai/codebook.py`) each time they run, so they stay in sync
with the code. To change a definition, edit the generating script, not the
codebook file.

## Key Outputs

### Occupational Characteristics (`output/`, see `output/codebook.md`)

- `output/occupations_eloundou_et_al.csv`: SOC 2018 panel with Eloundou et al.
  exposure scores and OEWS fields.
- `output/occupations_aei.csv`: SOC 2010 panel with AEI task, automation, and
  augmentation counts apportioned by equal and employment weights. This default
  output uses May 2022 OEWS.
- `output/occupations_aei_oews_2024.csv`: same AEI task panel rebuilt with May
  2024 OEWS for wage analysis.
- `output/occupations_aei_auto_aug_2025_03_27.csv`: SOC 2010 panel using the
  AEI occupation-level automation/augmentation release. The current unsuffixed
  file is from the default May 2022 OEWS run; use `--oews-year`/`--output-tag`
  to write year-tagged variants.

### O*NET IWA Mapping (`output/onet/iwa_mapping/`, see its `codebook.md`)

- `iwa_to_onet_soc_via_tasks.csv`: IWA to O*NET-SOC 2019 mapping via
  `IWA Reference -> DWA Reference -> Tasks to DWAs` (not committed; rebuild
  with `pipeline/build_iwa_onet_mapping.py`).
- `iwa_to_onet_soc_via_tasks_detail.csv`: task-DWA-IWA audit table (not
  committed).
- `iwa_occupation_links.csv`: slim IWA-occupation edge table with DWA counts,
  task counts, and IWA/occupation coverage counts (not committed).
- `iwa_occupation_counts.csv` and `occupation_iwa_counts.csv`: coverage
  summaries from both sides of the link table.
- `iwa_weight_*.csv` and `iwa_weight_*.png`: diagnostic summaries and
  correlations for possible within-IWA link weights.
- `iwa_onet_mapping_report.md`: mapping method, validation checks, coverage,
  and weight-correlation diagnostics.

### OpenAI IWA/OEWS Occupation Measures (`output/openai_iwa_oews/`, see its `codebook.md`)

- `iwa_soc2018_employment_links.csv`: static IWA by SOC 2018 link table with
  OEWS employment weights. O*NET-SOC 2019 codes are collapsed to six-digit SOC
  2018 for this merge.
- `openai_iwa_month.csv`: stacked OpenAI IWA/month data with mapping coverage
  fields.
- `openai_iwa_soc2018_month_panel.csv`: IWA by SOC 2018 by month link panel.
- `openai_soc2018_month_summary.csv`: SOC 2018 by month panel with
  employment-apportioned OpenAI IWA shares.
- `openai_soc2018_mean_summary.csv`: one row per SOC 2018 occupation, with mean
  all-message and work-related OpenAI IWA shares across all available months
  (committed).
- `openai_iwa_unmatched.csv`: unallocated OpenAI IWA rows, currently the
  `Other IWA` privacy bucket.
- `openai_iwa_oews_month_checks.csv` and `openai_iwa_oews_weight_checks.csv`:
  allocation validation tables.
- `iwa_openai_oews_report.md`: source coverage, allocation checks, zero-usage
  counts, and apportionment-sensitivity diagnostics.

### Wage And Cross-Release Analyses

- `output/net_automation_wages/` (committed, see its `codebook.md`):
  `occupation_net_automation_usage.csv`, `net_automation_wage_correlations.csv`,
  and `figures/` — wage correlations and figures for AEI net automation usage
  using the 2024 OEWS panel.
- `output/openai_iwa_wages/` (see its `codebook.md`):
  `openai_usage_wage_analysis_panel.csv`, `openai_usage_wage_correlations.csv`,
  `winsor_bounds.csv`, `openai_usage_wage_report.md`, and `figures/` — wage
  correlations and figures for OpenAI IWA occupation usage, including both raw
  apportioned share and per-million-worker variants.
- `output/cross_release_panel_task.csv` and
  `output/cross_release_panel_occupation.csv`: AEI cross-release panels
  (documented in `output/codebook.md`).
- `output/cross_release/task/` and `output/cross_release/occupation/`: task-
  and occupation-level cross-release comparison tables and figures (see
  `output/cross_release/codebook.md`).

## File Storage

Generated outputs are ignored by default; `output/.gitignore` allowlists a
small set of committed review artifacts (the occupation panels, crosswalk
tables, small IWA-mapping outputs, wage-analysis outputs, the OpenAI mean
summary, and every `codebook.md`) so they are available from a clean checkout
without running Python. Large regenerable tables — the multi-megabyte IWA
mapping link and audit CSVs — are not committed; rebuild them with
`uv run pipeline/build_iwa_onet_mapping.py`.

## Method Notes

- OEWS uses SOC 2018. AEI task outputs are built on O*NET-SOC 2010, so the repo
  allocates OEWS employment from SOC 2018 to SOC 2010 across direct crosswalk
  edges.
- O*NET 30.2 is O*NET-SOC 2019, which is based on SOC 2018. The OpenAI IWA
  pipeline collapses O*NET-SOC codes to six-digit SOC 2018 codes before merging
  OEWS. The OEWS merge uses exact SOC matches plus a simple trailing-zero
  broad-code fallback; it does not allocate other reported aggregate codes such
  as `25-2052` to SOC 2018 child occupations.
- OpenAI IWA shares are allocated to occupations by employment within each IWA.
  Link-count columns are retained as diagnostics, but no alternate OpenAI
  exposure outputs are produced.
- The OpenAI `Other IWA` privacy bucket is not allocated to occupations because
  it does not identify a specific O*NET IWA.
- Missing OEWS employment is imputed with the median across occupations before
  employment weighting; imputed rows are flagged with `oews_emp_was_imputed`.
- Scale caveat: `_pc` columns in the AEI occupation panels are per worker,
  while the OpenAI wage analysis reports `mean_share_per_million_workers` per
  million workers.

More detail is in `docs/eai.md`.

## Development

Lint and format:

```bash
uvx ruff check .
uvx ruff format --check .
```

Auto-format:

```bash
uvx ruff format .
```
