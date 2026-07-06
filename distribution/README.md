# Distribution Files

This directory contains the CSV files intended for distribution from a clean
checkout. They are copied from generated outputs in `output/` so `output/` can
remain a generated working directory.

Included files:

- `occupations_aei.csv`
- `occupations_eloundou_et_al.csv`
- `occupations_aei_auto_aug_2025_03_27.csv`

After regenerating the source outputs, update these copies with:

```bash
uv run python pipeline/sync_distribution.py
```

To verify the committed copies are current:

```bash
uv run python pipeline/sync_distribution.py --check
```
