# eai
Combining various data sources related to the economics of AI

## Data sources

- Anthropic Economic Index: https://huggingface.co/datasets/Anthropic/EconomicIndex
  - Reference code: https://huggingface.co/datasets/Anthropic/EconomicIndex/tree/main/release_2025_09_15
  - Download: `uv run anthropic/download.py` (update `RELEASE` in the script when a new release drops).
  - Clean: `uv run anthropic/clean.py` filters to GLOBAL + `onet_task::collaboration`, pivots wide (one column per collaboration type), and merges onto O*NET task statements. Outputs `data/<release>/aei_cleaned_claude_ai.csv`.
  - Method note: each AEI release-platform file is treated as a random sample of one million conversations.
- O*NET task statements: https://www.onetcenter.org/dictionary/20.1/excel/task_statements.html
- SOC 2010→2018 crosswalk: https://www.bls.gov/soc/2018/home.htm
- OEWS (Occupational Employment and Wage Statistics): https://www.bls.gov/oes/tables.htm
  - Current input file: 2022 national OEWS (`output/oews/national_M2022_dl.csv`)

## Occupational characteristics

`occupational_characteristics.py` builds three merged outputs:

- `output/occupations_eloundou_et_al.csv`
  - identifiers: `soc_2018`, `title_2018`, `group_id`
  - Eloundou columns: all aggregated numeric `dv_*` and `human_*` columns from `input/eloundou_occ_level.csv`
  - OEWS columns: `oews_occ_title`, `oews_tot_emp`, `oews_a_mean`, `oews_a_median`, `oews_broad_match`, `oews_soc_2018_broad`, `oews_tot_emp_adjusted`, `oews_tot_emp_imputed`
- `output/occupations_aei.csv`
  - identifiers: `soc_2010`, `title_2010`, `group_id`
  - OEWS columns: `oews_tot_emp_allocated`, `oews_tot_emp_imputed`, `oews_a_mean`
  - AEI columns: equal-weighted, employment-weighted, and employment-weighted per-capita counts for each platform and total across `task_count`, `automation_count`, and `augmentation_count`
- `output/occupations_aei_auto_aug_2025_03_27.csv`
  - identifiers: `soc_2010`, `title_2010`, `group_id`
  - OEWS columns: `oews_tot_emp_allocated`, `oews_tot_emp_imputed`, `oews_a_mean`
  - AEI auto/aug columns: `pct_occ_scaled`, `augmentation_weighted_ratio`, `automation_weighted_ratio`, `pct_occ_scaled_pc`

Crosswalk handling:

- `build_crosswalk.py` writes direct SOC 2010↔2018 edges to `output/onet/soc_crosswalk_edges.csv`
- it also writes connected-component lookups to `output/onet/soc_2010_to_group.csv` and `output/onet/soc_2018_to_group.csv`
- OEWS employment is moved from SOC 2018 to SOC 2010 by equal-splitting each SOC 2018 occupation across its direct SOC 2010 crosswalk links
- `group_id` is used for harmonized occupation grouping and diagnostics; employment allocation uses direct crosswalk edges, not connected-component totals
- `oews_tot_emp_allocated` is a modeled allocation to SOC 2010 occupations rather than a directly observed OEWS occupation count

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
