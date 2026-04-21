# Data Sources

## Anthropic Economic Index

Source: [Hugging Face](https://huggingface.co/datasets/Anthropic/EconomicIndex). Uses O\*NET-SOC 2010 codes (8-digit, e.g. `11-1011.00`), inherited from O\*NET 20.1 task statements.

## O\*NET Task Statements

Source: [O\*NET Center](https://www.onetcenter.org/dictionary/20.1/excel/task_statements.html), version 20.1. Uses O\*NET-SOC 2010 codes (8-digit, e.g. `11-1011.00`), which extend standard SOC codes with a `.XX` suffix for finer occupational granularity.

## OEWS (Occupational Employment and Wage Statistics)

Source: [Bureau of Labor Statistics](https://www.bls.gov/oes/tables.htm), 2022 national OEWS file. Uses SOC 2018 codes (6-digit, e.g. `11-1011`).

## Eloundou et al. (GPTs are GPTs)

Source: [GitHub](https://github.com/openai/GPTs-are-GPTs/blob/main/data/occ_level.csv). Uses O\*NET-SOC 2019 codes (8-digit, e.g. `11-1011.00`), built on the SOC 2018 taxonomy. Contains 923 occupations with human and GPT-4 exposure ratings at alpha, beta, and gamma thresholds.

## SOC 2010 <=> 2018 Crosswalk

Source: [Bureau of Labor Statistics](https://www.bls.gov/soc/2018/home.htm). The AEI and O\*NET use SOC 2010 while OEWS uses SOC 2018, so a crosswalk is required for joining. The mapping is many-to-many: some 2010 codes split into multiple 2018 codes, and some 2018 codes map to multiple 2010 codes.

I store two crosswalk artifacts:

- a direct edge list between SOC 2010 and SOC 2018 occupations
- connected components of the bipartite crosswalk graph, used to create a common occupation group identifier (`group_id`)

I use the direct edge list to allocate OEWS employment from SOC 2018 to SOC 2010. I use `group_id` for harmonized grouping and diagnostics.

# Occupational Characteristics Datasets

Exposure is defined and measured in a variety of ways. I try adopt the term 'occupational characteristics' to mitigate confusion about the various definitions of exposure in the literature. I construct some datasets that combine these sources in the `occupational_characteristics.py` script.

## Data Sources Based on 2018 SOC Codes

I use the 2018 half of the SOC 2010 <=> 2018 Crosswalk to form our universe of 2018 occupations.

I first merge the Eloundou et al. data onto the 2018 occupation universe. Since Eloundou et al. use 8-digit SOC codes, I first take the mean of all numeric Eloundou columns within 6-digit SOC code, including the `dv_*` and `human_*` measures. I treat missing values in the Eloundou data as true missing values.

I next merge the 2022 OEWS data onto the 2018 occupation universe, using 2018 SOC codes. I use the 2022 OEWS to represent pre-ChatGPT employment.

In some cases, the OEWS aggregates to broader SOC codes. In those cases, I apportion employment evenly across the matching detailed 6-digit SOC codes in the 2018 occupation universe and assume `a_mean` and `a_median` are distributed identically across those detailed occupations.

If we are missing OEWS employment data for an occupation, we replace it with the median employment value across all occupations with OEWS employment data.

## Data Sourced Based on 2010 SOC Codes

I use the 2010 half of the SOC 2010 <=> 2018 Crosswalk to form our universe of 2010 occupations.

I first aggregate the Anthropic Economic Index task-level usage counts for Claude.ai and the first party API for the September 2025, January 2026, and March 2026 releases. Each release-platform file categorizes a random sample of one million conversations, so summing counts across files gives each release-platform file equal weight.

I merge the task-level usage counts onto the O\*NET task statements. I fill in all tasks missing queries with zero usage.

We next need to apportion usage across occupations where multiple occupations share a task.

I keep OEWS at the SOC 2018 level, then apportion each SOC 2018 occupation's employment across its directly linked SOC 2010 occupations in the crosswalk. The default rule is an equal split across those direct links. The resulting SOC 2010 employment is therefore an allocated quantity, not directly observed OEWS employment. I then sum the allocated employment to the SOC 2010 level and compute an employment-weighted `a_mean` using the same allocation weights.

We apportion a task's usage counts with (a) equal weights and (b) employment weights across the occupations that share that task. The employment-weighted rule uses the allocated-or-imputed SOC 2010 employment, not directly observed OEWS occupation counts. If a task's linked occupations have zero total imputed employment, the employment-weighted rule falls back to equal weights.
