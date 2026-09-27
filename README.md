# Data

## Source

**Wine Quality** dataset, UCI Machine Learning Repository
(Cortez et al., 2009) —
[`winequality-red.csv`](winequality-red.csv) is the red-wine variant,
downloaded from
<https://archive.ics.uci.edu/dataset/53/wine+quality>.

Please cite the creators if you reuse it:
P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
"Modeling wine preferences by data mining from physicochemical properties."
*Decision Support Systems*, 47(4):547-553, 2009.

## Format

- 1,599 rows, semicolon-separated (`;`), header row included.
- 11 input features + `quality` target (integer sensory score, 3–8).

## Data dictionary

| Column | Description |
|--------|-------------|
| `fixed acidity` | tartaric acid, g/dm³ |
| `volatile acidity` | acetic acid, g/dm³ (too high → vinegar taste) |
| `citric acid` | g/dm³ (adds freshness) |
| `residual sugar` | g/dm³ |
| `chlorides` | sodium chloride, g/dm³ |
| `free sulfur dioxide` | mg/dm³ (antioxidant) |
| `total sulfur dioxide` | mg/dm³ |
| `density` | g/cm³ |
| `pH` | |
| `sulphates` | g/dm³ (preservative) |
| `alcohol` | % vol. |
| `quality` | sensory score 0–10 (observed 3–8) |
