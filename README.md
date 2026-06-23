# eai

Combining data sources related to the economics of AI, occupational tasks,
employment, wages, and usage/exposure measures.

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

Raw downloaded inputs are local and mostly ignored under `input/`. Derived CSVs,
figures, and reports are written under `output/`.

## Main Commands

Use `uv` for Python scripts.

```bash
uv run convert_onet.py
uv run build_crosswalk.py
uv run convert_oews.py
uv run download_aei.py
uv run clean_aei.py
```

Build the main occupational-characteristics panels:

```bash
uv run occupational_characteristics.py
uv run occupational_characteristics.py --oews-year 2024 --aei-only
```

Build the O*NET 30.2 IWA mapping and the OpenAI IWA/OEWS occupation measures:

```bash
uv run build_iwa_onet_mapping.py
uv run build_openai_iwa_oews.py
```

Run analysis scripts:

```bash
uv run analysis/net_automation_wages.py
uv run analysis/openai_iwa_wages.py
uv run analysis/cross_release_correlation.py
```

## Key Outputs

### Occupational Characteristics

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

### O*NET IWA Mapping

- `output/onet/iwa_mapping/iwa_to_onet_soc_via_tasks.csv`: IWA to O*NET-SOC
  2019 mapping via `IWA Reference -> DWA Reference -> Tasks to DWAs`.
- `output/onet/iwa_mapping/iwa_to_onet_soc_via_tasks_detail.csv`: task-DWA-IWA
  audit table.
- `output/onet/iwa_mapping/iwa_occupation_links.csv`: slim IWA-occupation edge
  table with DWA counts, task counts, and IWA/occupation coverage counts.
- `output/onet/iwa_mapping/iwa_occupation_counts.csv` and
  `output/onet/iwa_mapping/occupation_iwa_counts.csv`: coverage summaries from
  both sides of the link table.
- `output/onet/iwa_mapping/iwa_weight_*.csv` and `iwa_weight_*.png`: diagnostic
  summaries and correlations for possible within-IWA link weights.
- `output/onet/iwa_mapping/iwa_onet_mapping_report.md`: mapping method,
  validation checks, coverage, and weight-correlation diagnostics.

### OpenAI IWA/OEWS Occupation Measures

- `output/openai_iwa_oews/iwa_soc2018_employment_links.csv`: static IWA by SOC
  2018 link table with OEWS employment weights. O*NET-SOC 2019 codes are
  collapsed to six-digit SOC 2018 for this merge.
- `output/openai_iwa_oews/openai_iwa_month.csv`: stacked OpenAI IWA/month data
  with mapping coverage fields.
- `output/openai_iwa_oews/openai_iwa_soc2018_month_panel.csv`: IWA by SOC 2018
  by month link panel.
- `output/openai_iwa_oews/openai_soc2018_month_summary.csv`: SOC 2018 by month
  panel with employment-apportioned OpenAI IWA shares.
- `output/openai_iwa_oews/openai_soc2018_mean_summary.csv`: one row per SOC
  2018 occupation, with mean all-message and work-related OpenAI IWA shares
  across all available months.
- `output/openai_iwa_oews/openai_iwa_unmatched.csv`: unallocated OpenAI IWA
  rows, currently the `Other IWA` privacy bucket.
- `output/openai_iwa_oews/openai_iwa_oews_month_checks.csv` and
  `output/openai_iwa_oews/openai_iwa_oews_weight_checks.csv`: allocation
  validation tables.
- `output/openai_iwa_oews/iwa_openai_oews_report.md`: source coverage,
  allocation checks, zero-usage counts, and apportionment-sensitivity
  diagnostics.

### Wage And Cross-Release Analyses

- `output/net_automation_wages/occupation_net_automation_usage.csv`,
  `net_automation_wage_correlations.csv`, and `figures/`: wage correlations and
  figures for AEI net automation usage using the 2024 OEWS panel.
- `output/openai_iwa_wages/openai_usage_wage_analysis_panel.csv`,
  `openai_usage_wage_correlations.csv`, `winsor_bounds.csv`,
  `openai_usage_wage_report.md`, and `figures/`: wage correlations and figures
  for OpenAI IWA occupation usage, including both raw apportioned share and
  per-million-worker variants.
- `output/cross_release_panel_task.csv` and
  `output/cross_release_panel_occupation.csv`: AEI cross-release panels.
- `output/cross_release/task/` and `output/cross_release/occupation/`: current
  task- and occupation-level cross-release comparison tables and figures. Older
  top-level `output/cross_release/tables/` and `figures/` files may still exist
  from earlier runs.

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
