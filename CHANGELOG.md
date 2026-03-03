# Changelog

## 1.0.0 — 2026-03-03

### Added
- **Guide system**: `guide()` with 15+ topics — ML basics, data leakage, cheatsheet, and more.
- **Full pipeline**: `run_pipeline()` — one-line autopilot from CSV to trained model.
- **Individual modules**: `load_data`, `clean`, `split`, `encode`, `scale`, `train`, `evaluate`, `predict`, `compare_models`, `save_model`, `load_model`, `get_feature_importance`, `detect_task`, `sample_data`.
- **Data leakage prevention**: `PreprocessingOrderWarning` when encode/scale is called before split.
- **Student-friendly errors**: Typo suggestions, available column listing, clear next-step guidance.
- **Bundled sample datasets**: Regression, classification, and Titanic datasets for zero-setup practice.
- **Contextual hints**: Every function prints a "what to do next" hint when `verbose=True`.
