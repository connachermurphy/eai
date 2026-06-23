# EAI Data And Measurement Notes

This document describes the current data sources, build scripts, occupation
universes, exposure measures, and analysis outputs in the repo.

## Source Inventory

### Anthropic Economic Index

Source: Hugging Face dataset for the Anthropic Economic Index.

The repo handles three AEI releases:

- `release_2025_09_15`
- `release_2026_01_15`
- `release_2026_03_24`

Each release has Claude.ai and first-party API files. `download_aei.py`
downloads the registered releases into `input/`. `clean_aei.py` filters raw
files to `geo_id == "GLOBAL"`, pivots `onet_task::collaboration` values wide,
and writes one cleaned file per release/platform to `output/release_*/`.

The cleaned task-level AEI data use O*NET task text and O*NET-SOC 2010
occupation codes through the O*NET 20.1 task statements.

### OpenAI Signals

The local OpenAI Signals files live under
`input/oai_260616/data-download-csv`. The OpenAI IWA pipeline currently uses:

- `usa_share_of_messages_by_onet_iwa_month.csv`
- `usa_share_of_work_related_messages_by_onet_iwa_month.csv`

Both files contain monthly `share_of_messages` values. The local release README
defines these shares as differentially private weighted shares in `[0, 1]`.
The currently generated panel covers 21 months, from `2024-07-01` through
`2026-03-01`.

### O*NET 20.1

`convert_onet.py` converts the O*NET 20.1 task statements workbook to
`output/onet/task_statements.csv`. The AEI task-level pipeline uses these task
statements and truncates 8-digit O*NET-SOC 2010 codes to six-digit SOC 2010
codes when aggregating to occupations.

### O*NET 30.2

`build_iwa_onet_mapping.py` reads local O*NET 30.2 Excel files in
`input/db_30_2_excel/` and the local O*NET 30.2 data dictionary PDF. It builds
IWA-to-occupation mappings through:

`IWA Reference -> DWA Reference -> Tasks to DWAs -> O*NET-SOC Code`

O*NET 30.2 uses O*NET-SOC 2019, which is based on the 2018 SOC taxonomy. The
OpenAI IWA/OEWS pipeline therefore collapses O*NET-SOC codes to six-digit SOC
2018 codes before merging OEWS.

### OEWS

`convert_oews.py` converts local OEWS Excel files to lowercase CSVs under
`output/oews/`. The repo currently has national OEWS files for 2019 through
2024, plus the May 2024 all-data file.

Important year distinction:

- `occupational_characteristics.py` defaults to May 2022 OEWS.
- `occupational_characteristics.py --oews-year 2024 --aei-only` writes the
  2024-tagged AEI/OEWS panel used by the net automation wage analysis.
- `build_openai_iwa_oews.py` defaults to May 2024 OEWS.
- `analysis/openai_iwa_wages.py` and `analysis/net_automation_wages.py` use
  2024 OEWS-based inputs by default.

### SOC Crosswalk

`build_crosswalk.py` writes:

- `output/onet/soc_crosswalk_edges.csv`
- `output/onet/soc_2010_to_group.csv`
- `output/onet/soc_2018_to_group.csv`

The direct edge file is used for OEWS allocation between SOC 2018 and SOC 2010.
The group files are connected components of the bipartite SOC 2010/SOC 2018
crosswalk graph and are used as harmonized occupation-group identifiers and
diagnostics. Employment allocation uses direct edges, not connected-component
totals.

### Eloundou Et Al.

`occupational_characteristics.py` reads `input/eloundou_occ_level.csv`, collapses
numeric exposure columns to six-digit SOC 2018, and merges them onto the SOC
2018 universe.

## Occupation Universes

The current generated outputs use three related occupation universes:

- SOC 2018: 867 detailed occupations from `output/onet/soc_2018_to_group.csv`.
- SOC 2010: 840 detailed occupations from `output/onet/soc_2010_to_group.csv`.
- O*NET-SOC 30.2: 923 O*NET-SOC occupation identifiers in the O*NET 30.2 task
  statements.

Broad SOC codes ending in `0` are excluded from the SOC 2010 and SOC 2018
universes. When OEWS reports a broad code that corresponds to multiple detailed
SOC 2018 occupations, the scripts split employment equally across those detailed
occupations and assign the reported wage values to each split detailed row.

## Build Order

A typical rebuild order is:

```bash
uv run convert_onet.py
uv run build_crosswalk.py
uv run convert_oews.py
uv run download_aei.py
uv run clean_aei.py
uv run occupational_characteristics.py
uv run occupational_characteristics.py --oews-year 2024 --aei-only
uv run build_iwa_onet_mapping.py
uv run build_openai_iwa_oews.py
uv run analysis/net_automation_wages.py
uv run analysis/openai_iwa_wages.py
uv run analysis/cross_release_correlation.py
```

Some raw inputs are local and ignored by git. The build scripts expect those
inputs to already exist in `input/`.

## Occupational Characteristics Pipeline

Script: `occupational_characteristics.py`

Default command:

```bash
uv run occupational_characteristics.py
```

The script creates three main panels.

### Eloundou SOC 2018 Panel

Output: `output/occupations_eloundou_et_al.csv`

Current shape: 867 rows.

The script takes the SOC 2018 universe, merges Eloundou et al. numeric exposure
columns by six-digit SOC 2018, and merges OEWS employment and wage fields. OEWS
matching uses exact detailed SOC 2018 matches first and broad-code matching
second. Missing OEWS employment is imputed only for occupations with nonmissing
Eloundou exposure values.

### AEI Task SOC 2010 Panel

Outputs:

- `output/occupations_aei.csv`
- `output/occupations_aei_oews_2024.csv`

Current shape: 840 rows for each panel.

The script loads cleaned AEI task files across releases and platforms, computes
automation counts as `directive_count + feedback_loop_count`, computes
augmentation counts as `validation_count + task_iteration_count + learning_count`,
and sums task counts across registered releases.

AEI task counts are merged onto O*NET 20.1 task-occupation pairs. Missing AEI
tasks are filled with zero, which means an O*NET task with no observed AEI usage
is treated as zero usage rather than dropped.

Shared task counts are apportioned across SOC 2010 occupations in two ways:

- Equal weights across all occupations linked to the task.
- Employment weights using allocated-or-imputed OEWS employment.

OEWS starts at SOC 2018. For this SOC 2010 panel, OEWS employment is allocated
from SOC 2018 to SOC 2010 by equal-splitting each SOC 2018 occupation across its
direct crosswalk edges. The resulting `oews_tot_emp_allocated` is therefore a
modeled allocation, not a directly observed OEWS occupation count.

The script also adds employment-normalized versions of the employment-weighted
AEI counts. Zero employment is recoded to missing before per-capita division.

### AEI Occupation Automation/Augmentation Panel

Output: `output/occupations_aei_auto_aug_2025_03_27.csv`

Current shape: 840 rows.

The current unsuffixed output is generated by the default
`occupational_characteristics.py` run, so it uses May 2022 OEWS. Running the
script with a non-default OEWS year and without `--aei-only` would write a
year-tagged variant.

This panel reads `input/aei_occupation_automation_augmentation_data.csv`. Because
that file contains occupation-level ratios rather than additive task counts, the
script collapses 8-digit O*NET-SOC rows to six-digit SOC 2010 using
`pct_occ_scaled` as the weighting variable for the automation and augmentation
ratios. Occupations with O*NET tasks but no row in the AEI occupation release are
zero-recoded.

## O*NET 30.2 IWA Mapping

Script: `build_iwa_onet_mapping.py`

Command:

```bash
uv run build_iwa_onet_mapping.py
```

Core outputs:

- `output/onet/iwa_mapping/iwa_to_onet_soc_via_tasks.csv`
- `output/onet/iwa_mapping/iwa_to_onet_soc_via_tasks_detail.csv`
- `output/onet/iwa_mapping/iwa_occupation_links.csv`
- `output/onet/iwa_mapping/iwa_occupation_counts.csv`
- `output/onet/iwa_mapping/occupation_iwa_counts.csv`
- `output/onet/iwa_mapping/iwa_weight_summary.csv`
- `output/onet/iwa_mapping/iwa_weight_correlations.csv`
- `output/onet/iwa_mapping/iwa_weight_correlations_pearson_matrix.csv`
- `output/onet/iwa_mapping/iwa_weight_correlations_spearman_matrix.csv`
- `output/onet/iwa_mapping/iwa_weight_scatter.png`
- `output/onet/iwa_mapping/iwa_weight_correlation_heatmap.png`
- `output/onet/iwa_mapping/iwa_onet_mapping_report.md`

Current mapping counts:

- 332 IWAs.
- 2,087 DWAs.
- 23,850 task-to-DWA-to-IWA detail rows.
- 15,285 IWA/O*NET-SOC occupation pairs.
- 923 O*NET-SOC occupations.

The compact link table `iwa_occupation_links.csv` is one row per
IWA/O*NET-SOC occupation link. It includes:

- `link_dwa_count`: number of distinct DWAs connecting the IWA and occupation.
- `iwa_count_for_occupation`: number of distinct IWAs linked to the occupation.
- `occupation_count_for_iwa`: number of distinct occupations linked to the IWA.
- Task, DWA, core-task, supplemental-task, and unclassified-task diagnostics.

The script also writes normalized within-IWA weight columns based on task and
DWA link counts. These are diagnostics for future apportionment work. They are
not the active OpenAI exposure outputs.

## OpenAI IWA To OEWS Occupation Measures

Script: `build_openai_iwa_oews.py`

Command:

```bash
uv run build_openai_iwa_oews.py
```

The script combines:

- OpenAI monthly IWA shares.
- O*NET 30.2 IWA-to-occupation links.
- May 2024 OEWS employment and wage data.
- The SOC 2018 universe and crosswalk groups.

### Method

1. Collapse the O*NET 30.2 IWA mapping to one row per `iwa_id` by six-digit
   `soc_2018`.
2. Merge May 2024 OEWS to SOC 2018 using the same exact plus broad-code handling
   used elsewhere in the repo.
3. Impute missing linked SOC 2018 employment to the median employment among
   unique linked SOC 2018 occupations with OEWS employment. The current
   imputation value is 43,105.
4. Within each IWA, compute
   `employment_weight_within_iwa = oews_tot_emp_imputed / sum(oews_tot_emp_imputed)`.
5. Allocate each OpenAI IWA/month share to linked SOC 2018 occupations using
   that employment weight.
6. Sum across IWAs to create SOC 2018 monthly measures.
7. Average across months to create the SOC 2018 mean summary.

No alternate OpenAI exposure outputs are implemented. The report includes
sensitivity diagnostics comparing employment apportionment with equal allocation
and link-count allocations, but those diagnostics are not saved as alternate
exposure files.

### OpenAI Measures

The two measures are:

- `us_all_messages_iwa_share`: monthly IWA share among all U.S. consumer
  ChatGPT messages.
- `us_work_related_messages_iwa_share`: monthly IWA share among work-related
  U.S. consumer ChatGPT messages.

The mean summary output keeps these as wide columns:

- `mean_us_all_messages_iwa_share`
- `mean_us_work_related_messages_iwa_share`

### Coverage And Interpretation

Current generated outputs cover 867 SOC 2018 occupations and 21 months
(`2024-07-01` through `2026-03-01`).

OpenAI's `Other IWA` privacy bucket is intentionally left unallocated because it
does not identify a specific O*NET IWA. As a result, the summed occupation-level
mean shares are slightly below one:

- all U.S. messages: about 0.9826
- work-related U.S. messages: about 0.9842

Occupations with no allocated released-IWA usage are assigned zero in the monthly
and mean SOC 2018 summaries. These zeroes mean zero usage from the released,
mapped IWA categories after allocation.

The measurement is sensitive to the apportionment assumption. Current report
diagnostics show employment-vs-diagnostic Spearman correlations around 0.82 to
0.84, but Pearson correlations around 0.45 to 0.54. This suggests rank ordering
is fairly stable, while the level of the occupation exposure measure is
meaningfully affected by the allocation rule.

### Outputs

- `output/openai_iwa_oews/iwa_soc2018_employment_links.csv`: static IWA by SOC
  2018 link table with employment weights.
- `output/openai_iwa_oews/openai_iwa_month.csv`: stacked OpenAI IWA/month data
  with mapping coverage fields.
- `output/openai_iwa_oews/openai_iwa_soc2018_month_panel.csv`: IWA by SOC 2018
  by month link panel.
- `output/openai_iwa_oews/openai_soc2018_month_summary.csv`: SOC 2018 by month
  summary, 36,414 rows.
- `output/openai_iwa_oews/openai_soc2018_mean_summary.csv`: SOC 2018 mean
  summary, 867 rows.
- `output/openai_iwa_oews/openai_iwa_unmatched.csv`: currently the unallocated
  `Other IWA` rows.
- `output/openai_iwa_oews/openai_iwa_oews_month_checks.csv`: month-level source,
  mapped, unmatched, and apportioned share checks.
- `output/openai_iwa_oews/openai_iwa_oews_weight_checks.csv`: IWA/month weight
  and allocation checks.
- `output/openai_iwa_oews/iwa_openai_oews_report.md`: method, coverage, checks,
  zero counts, and apportionment-sensitivity diagnostics.

## Wage Analyses

### AEI Net Automation And Wages

Script: `analysis/net_automation_wages.py`

Default input: `output/occupations_aei_oews_2024.csv`

The script computes net automation usage as employment-weighted automation usage
minus employment-weighted augmentation usage, using the per-capita AEI columns.
It then compares net automation usage with May 2024 OEWS annual mean wages.

Outputs:

- `output/net_automation_wages/occupation_net_automation_usage.csv`
- `output/net_automation_wages/net_automation_wage_correlations.csv`
- `output/net_automation_wages/figures/`

Correlations are reported both on all occupations with wage/employment data and
on nonzero net-usage occupations. Usage columns are winsorized at p1/p99 for the
correlation and plotting outputs. Weighted correlations use
`oews_tot_emp_imputed`.

### OpenAI IWA Usage And Wages

Script: `analysis/openai_iwa_wages.py`

Default input: `output/openai_iwa_oews/openai_soc2018_mean_summary.csv`

The script reshapes the two OpenAI mean-share columns internally and compares
them with May 2024 OEWS annual mean wages. It reports:

- `Mean apportioned share`: the monthly IWA-to-SOC 2018 usage share averaged
  across months.
- `Mean apportioned share per million workers`: the mean share divided by
  imputed OEWS employment and multiplied by one million.

Both variants are useful but answer different questions. The raw apportioned
share reflects the occupation's share of allocated OpenAI IWA usage. The
per-million-worker measure asks how concentrated that allocated usage is relative
to occupation employment.

Outputs:

- `output/openai_iwa_wages/openai_usage_wage_analysis_panel.csv`
- `output/openai_iwa_wages/openai_usage_wage_correlations.csv`
- `output/openai_iwa_wages/winsor_bounds.csv`
- `output/openai_iwa_wages/openai_usage_wage_report.md`
- `output/openai_iwa_wages/figures/`

Zero employment is recoded to missing before per-worker division. Current
generated data have no zero imputed-employment occupations.

## AEI Cross-Release Analysis

Script: `analysis/cross_release_correlation.py`

Command:

```bash
uv run analysis/cross_release_correlation.py
```

The script builds task- and occupation-level panels comparing AEI task counts
across releases and platforms.

Task-level output:

- `output/cross_release_panel_task.csv`: 18,428 task rows.

Occupation-level output:

- `output/cross_release_panel_occupation.csv`: 974 O*NET-SOC occupation rows.

For occupation-level comparisons, task counts are allocated equally across all
O*NET-SOC occupations sharing the task. For each task, weights sum to one, so
total task counts are preserved when moving to occupations.

The current script writes extensive-margin tables, Pearson/Spearman
correlations, and scatter plots under:

- `output/cross_release/task/`
- `output/cross_release/occupation/`

Older top-level `output/cross_release/tables/` and
`output/cross_release/figures/` files may still exist from earlier runs and
should not be confused with the current task/occupation split. The current
comparisons are:

- January 2026 vs March 2026 within API and Claude.ai.
- API vs Claude.ai within each release.
- API vs Claude.ai pooled across releases.

## Output And Git Hygiene

The repo keeps raw source downloads out of version control where possible.
`input/.gitignore` excludes local data drops such as the O*NET 30.2 folder and
OpenAI Signals release folder. `output/.gitignore` ignores most generated
intermediate files by default while allowing selected occupation and O*NET
outputs to be tracked. Some analysis outputs have been force-added when they are
useful review artifacts.

When regenerating outputs, check the resulting diff before committing. Many
outputs contain dates, figures, or generated report text.

## Outstanding Measurement Questions

- The OpenAI `Other IWA` bucket is unallocated. This is preferable to assigning a
  privacy bucket to occupations without an identified IWA, but it means the
  occupation-level OpenAI shares sum to mapped released-IWA usage rather than the
  full OpenAI source total.
- Employment apportionment is the only active OpenAI IWA allocation rule. The
  sensitivity diagnostics should be revisited before treating levels as
  production measures.
- OEWS broad-code matching assigns the same reported wage values to detailed SOC
  occupations split from a broad OEWS row. This is a standard practical handling
  in the repo, but it is still an assumption.
- SOC 2010 and SOC 2018 panels are not directly interchangeable. Use `group_id`
  for harmonized grouping and diagnostics, but do not treat connected components
  as the employment-allocation rule.
