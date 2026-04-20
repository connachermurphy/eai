= Economics of AI

== Anthropic Economic Index

Source: #link("https://huggingface.co/datasets/Anthropic/EconomicIndex")[Hugging Face]. Uses O\*NET-SOC 2010 codes (8-digit, e.g. `11-1011.00`), inherited from O\*NET 20.1 task statements.

== O\*NET Task Statements

Source: #link("https://www.onetcenter.org/dictionary/20.1/excel/task_statements.html")[O\*NET Center], version 20.1. Uses O\*NET-SOC 2010 codes (8-digit, e.g. `11-1011.00`), which extend standard SOC codes with a `.XX` suffix for finer occupational granularity.

== OEWS (Occupational Employment and Wage Statistics)

Source: #link("https://www.bls.gov/oes/tables.htm")[Bureau of Labor Statistics], May 2024. Uses SOC 2018 codes (6-digit, e.g. `11-1011`).

== Eloundou et al. (GPTs are GPTs)

Source: #link("https://github.com/openai/GPTs-are-GPTs/blob/main/data/occ_level.csv")[GitHub]. Uses O\*NET-SOC 2019 codes (8-digit, e.g. `11-1011.00`), built on the SOC 2018 taxonomy. Contains 923 occupations with human and GPT-4 exposure ratings at alpha, beta, and gamma thresholds.

== SOC 2010→2018 Crosswalk

Source: #link("https://www.bls.gov/soc/2018/home.htm")[Bureau of Labor Statistics]. The AEI and O\*NET use SOC 2010 while OEWS uses SOC 2018, so a crosswalk is required for joining. The mapping is many-to-many: some 2010 codes split into multiple 2018 codes, and some merge.
