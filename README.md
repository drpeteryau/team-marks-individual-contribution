# team-marks-individual-contribution

Python code for the analysis in:

> Yau, P. C. (2026). *Do team marks reflect individual contribution? Empirical evidence from an undergraduate computing module.* Preprint.

The script computes score dispersion, effective weights, tracking of solo performance, Brown–Forsythe and Pitman–Morgan tests, bootstrap intervals, rank-based quartile summaries and the reweighting simulation for three cohorts of one module.

## Data

Student marks are **not** included and must not be committed (they are not public). Supply one `.xlsx` file per cohort:

- column headers on row 3 (change with `--header-row`);
- one row per student with a sequential anonymous id (no names, matriculation numbers or team identifiers);
- the component columns named in `SPEC` at the top of `analyse_team_marks.py`.

Add `*.xlsx` and `results/` to `.gitignore`.

## Run

```bash
pip install -r requirements.txt
python analyse_team_marks.py --ay2324 AY2324.xlsx --ay2425 AY2425.xlsx --ay2526 AY2526.xlsx --out results/
```

Outputs: `results.json` (all statistics reported in the paper), `Figure1_rank_scatter.png`, `Figure2_reweighting.png`. The random seed is fixed (2026), so bootstrap intervals are reproducible.

## Adapting to another module

Edit three blocks near the top of the script: `Y` (file loading, column names, published weights), `R` (observed solo or group-type classification by the module leader) and `DES` (designed classification), plus the component pairs for the Pitman–Morgan tests.

## Tested with

Python 3.13, numpy 2.5, pandas 3.0, scipy 1.18, matplotlib 3.11.

## Licence

MIT (add a `LICENSE` file when you create the repository).
