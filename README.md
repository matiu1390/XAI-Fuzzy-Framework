# XAI-Fuzzy-Framework

Lightweight bundle combining:
- `skmoefs/` (fork of GionatanG/skmoefs) with per-rule `ponderaciones`, richer introspection, and custom demo code.
- `fuzzy_rb_system/` (fork of kgeorgiev42/Fuzzy-RB-System) with lowercase connectors, extra debugging, and tweaked defuzzification/plotting.
- `Deffuzy.py`: utility to convert normalized triangular partitions and attribute ranges into trapezoidal parameters (center + widths) for downstream fuzzy rule export.

What's included
- Core source only; datasets, CSV/XLSX results, images, and heavy artifacts are **omitted**. Add your own data/rule files as needed.
- Fuzzy-RB sample scripts: `fuzzy_validation.py` plus `modules/` helpers. Bring your own `.fuzzy` rulebases and measurements.
- Skmoefs library code under `skmoefs/` (no bundled `dataset/` samples).

Datasets (not included; see Biomedicines 2025, 13, 1483)
- Petry et al. (Cambridge Baby Growth Study, 438 pregnant women; age at menarche vs blood pressure) [42].
- Tatapudi et al. (prospective case-control, 100 women in India; hypertensive disorders) [43].
- Thitivichienlert et al. (34 preeclampsia patients; renal biomarkers and BP follow-up) [44].
- Pham et al. (cohort of 210 pregnant women, 198 labeled; fuzzy knowledge-graph diagnostic model) [22].
- Data availability: see the paper's Data Availability Statement for the original sources; datasets must be obtained from those repositories and are not redistributed here.

Dataset locations (for reproducibility)
- Place `.dat` files for skmoefs under `skmoefs/dataset/` (mirroring the original repo structure).
- Place `.fuzzy` or `.txt` rulebase files and measurement inputs for Fuzzy-RB scripts alongside `fuzzy_rb_system/fuzzy_validation.py` (or adjust paths in that script).

Quick start
1) Create a virtualenv and install dependencies appropriate to each part (e.g. `skmoefs` requirements and `fuzzy_rb_system/requirements.txt`).
2) For skmoefs, follow the usage in `skmoefs/example.py` (provide your dataset separately).
3) For Fuzzy-RB, run `python fuzzy_rb_system/fuzzy_validation.py` after supplying a rulebase/measurements file.
4) To transform partitions, run `python Deffuzy.py` with your own partition/range data (see script comments for expected structure).

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
