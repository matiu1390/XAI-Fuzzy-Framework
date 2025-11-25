# XAI-Fuzzy-Framework

Lightweight bundle combining:
- `skmoefs/` (fork of GionatanG/skmoefs) with expanded rule weights (`ponderaciones`), richer introspection, and a custom demo pipeline.
- `fuzzy_rb_system/` (fork of kgeorgiev42/Fuzzy-RB-System) with lowercase connectors, extra debugging, defuzzification/plot tweaks, and validation scripts.
- `Deffuzy.py`: utility to convert normalized triangular partitions and attribute ranges into trapezoidal parameters (center + widths) for downstream fuzzy rule export.

![Framework overview](Framework.png)

Fork changes (detailed)
- skmoefs core:
  - Rules now carry `ponderaciones`; `FuzzyRuleBasedClassifier` stores antecedent matrices/dicts, partitions, weights, and exposes getters (`fuzzy_set`, `antecedentes_matrix`, `granularidad`, etc.).
  - Prediction helpers return extra debug info: `predict` now keeps matching degrees and best-match vectors; `predict_bm` returns best-match value/index per sample.
  - Display helpers extended (`NEW_show_RB`, `NEW_show_DB`) to emit rule bases and membership plots with custom labels and attributes; support more granular labels (VL/L/ML/... up to 6).
  - Discretization: handles empty splits gracefully; triangular partitions guarded against empty feature partitions.
  - Example workflow replaced with an `iris/mpaes22` script using `FuzzyMDLFilter`, extended logging, and rule/weight inspection.
- fmdt/rcs/toolbox wiring:
  - FMDT nodes now store `ponderaciones`; `_csv_ruleMine` returns weights alongside rules.
  - RCS initializer passes `ponderaciones` into `ClassificationRule`; tree defaults tuned; decoder injects them when building classifiers.
  - Toolbox propagates `ponderaciones` through MOEA (MPAES_RCS), extends `show_model` to return multiple artifacts (classes, weights, antecedents, partitions), adds `show_predict` with best-match info, and logs more in Pareto plots/cross-val.
- fuzzy_rb_system:
  - Connectors normalized to lowercase (`and`/`or`), support up to three antecedents in inference/control, added debug prints, and enhanced defuzzification plotting (labeled centroid/bisector, shifted curves).
  - `fuzzy_validation.py` combines parsing, inference, defuzzification plots, and custom rule sets; plotting uses explicit x/y for seaborn.
- Deffuzy:
  - Helper to convert triangular partitions plus attribute ranges into trapezoidal parameters for rule export (centers + left/right widths).

What's included
- Core source for both components. The skmoefs sample datasets (`.dat`) are included under `skmoefs/dataset/`; CSV/XLSX results, images, and other heavy artifacts remain **omitted**.
- Fuzzy-RB sample scripts: `fuzzy_validation.py` plus `modules/` helpers. Bring your own `.fuzzy` rulebases and measurements.
- Skmoefs library code under `skmoefs/` (with bundled `dataset/` samples).

Datasets (see Biomedicines 2025, 13, 1483)
- Petry et al. (Cambridge Baby Growth Study, 438 pregnant women; age at menarche vs blood pressure) [42].
- Tatapudi et al. (prospective case-control, 100 women in India; hypertensive disorders) [43].
- Thitivichienlert et al. (34 preeclampsia patients; renal biomarkers and BP follow-up) [44].
- Pham et al. (cohort of 210 pregnant women, 198 labeled; fuzzy knowledge-graph diagnostic model) [22].
- Data availability: see the paper's Data Availability Statement for the original sources; skmoefs sample `.dat` files are included here, other datasets must be obtained from their original repositories.

Dataset locations (for reproducibility)
- Place `.dat` files for skmoefs under `skmoefs/dataset/` (sample `.dat` files are already included).
- Place `.fuzzy` or `.txt` rulebase files and measurement inputs for Fuzzy-RB scripts alongside `fuzzy_rb_system/fuzzy_validation.py` (or adjust paths in that script).

Quick start
1) Create a virtualenv and install dependencies appropriate to each part (e.g. `skmoefs` requirements and `fuzzy_rb_system/requirements.txt`).
2) For skmoefs, follow the usage in `skmoefs/example.py` (use included sample datasets or provide your own).
3) To transform partitions, run `python Deffuzy.py` with your own partition/range data (see script comments for expected structure).
4) For Fuzzy-RB, run `python fuzzy_rb_system/fuzzy_validation.py` after supplying a rulebase/measurements file.

Runtime versions (current env)
- numpy 1.21.6
- scipy 1.10.1
- numba 0.57.0
- matplotlib 3.7.4
- scikit-learn 1.3.2
- platypus 1.1.0 (Platypus-Opt)

## Citations
If you use this repository, please cite:
```
@article{Salinas2025PreeclampsiaFuzzy,
  title   = {An Explainable Fuzzy Framework for Assessing Preeclampsia Classification},
  author  = {Salinas, Matias and Velandia, Daira and Mayeta-Revilla, Leondry and Bertini, Ayleen and Querales, Marvin and Pardo, Fabian and Salas, Rodrigo},
  journal = {Biomedicines},
  year    = {2025},
  volume  = {13},
  number  = {6},
  pages   = {1483},
  doi     = {10.3390/biomedicines13061483}
}
```

Attribution
- Based on MIT-licensed upstream projects listed above; license files remain in their respective source directories.
- Please also cite the original upstream projects if you build upon this bundle:
  - GionatanG/skmoefs (fuzzy MOEFS library)
  - kgeorgiev42/Fuzzy-RB-System (fuzzy rule-base utilities)
