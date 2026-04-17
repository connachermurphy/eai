= Economics of AI

== Anthropic Economic Index

Source: #link("https://huggingface.co/datasets/Anthropic/EconomicIndex")[Hugging Face]. Uses O\*NET-SOC codes (8-digit, e.g. `11-1011.00`), inherited from O\*NET task statements.

== O\*NET Task Statements

Source: #link("https://www.onetcenter.org/dictionary/20.1/excel/task_statements.html")[O\*NET Center]. Uses O\*NET-SOC codes (8-digit, e.g. `11-1011.00`), which extend standard SOC codes with a `.XX` suffix for finer occupational granularity.

== OEWS (Occupational Employment and Wage Statistics)

Source: #link("https://www.bls.gov/oes/tables.htm")[Bureau of Labor Statistics]. Uses standard SOC codes (6-digit, e.g. `11-1011`). The first 6 digits of O\*NET-SOC codes map to SOC codes, enabling joins across sources.
