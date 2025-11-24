# XAI-Fuzzy-Framework

Lightweight bundle combining:
- `skmoefs/` (fork of [GionatanG/skmoefs](https://github.com/GionatanG/skmoefs)) with per-rule `ponderaciones`, richer introspection, and custom demo code.
- `fuzzy_rb_system/` (fork of [kgeorgiev42/Fuzzy-RB-System](https://github.com/kgeorgiev42/Fuzzy-RB-System)) with lowercase connectors, extra debugging, and tweaked defuzzification/plotting.
- `Deffuzy.py`: utility to convert normalized triangular partitions and attribute ranges into trapezoidal parameters (center + widths) for downstream fuzzy rule export.

What’s included
- Core source only; datasets, CSV/XLSX results, images, and heavy artifacts are **omitted**. Add your own data/rule files as needed.
- Fuzzy-RB sample scripts: `fuzzy_validation.py` plus `modules/` helpers. Bring your own `.fuzzy` rulebases and measurements.
- Skmoefs library code under `skmoefs/` (no bundled `dataset/` samples).

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

Attribution
- Based on MIT-licensed upstream projects listed above; license files remain in their respective source directories.
