# English Docs Sync TODO (align with Chinese)

> Goal: make `docs/en` match the current `docs/zh` structure/content.

## ✅ Missing in English (needs add/translate)
- [x] `docs/en/3_full_detail_tutorials/02_data_analysis/01_asa_pipeline.ipynb`
- [x] `docs/en/3_full_detail_tutorials/02_data_analysis/02_cann1d_bump_fit.ipynb`
- [x] `docs/en/3_full_detail_tutorials/02_data_analysis/03_flipflop_tutorial.ipynb`
- [x] `docs/en/3_full_detail_tutorials/02_data_analysis/04_cell_classification.ipynb`
- [x] `docs/en/3_full_detail_tutorials/04_pipeline/01_asa_tui.rst`
- [x] `docs/en/3_full_detail_tutorials/04_pipeline/02_model_gallery_tui.rst`
- [x] `docs/en/3_full_detail_tutorials/04_pipeline/03_asa_gui.rst`
- [x] `docs/en/examples/index.rst`
- [x] `docs/en/examples/README.rst`
- [x] `docs/en/experiments/README.rst`

## 🧹 English-only files to remove or replace
- [x] `docs/en/3_full_detail_tutorials/04_pipeline/01_theta_sweep_pipeline.ipynb` (no zh counterpart; replace with ASA GUI/TUI docs)
- [x] `docs/en/3_full_detail_tutorials/01_cann_modeling/06_theta_sweep_hd_grid.ipynb.bak`
- [x] `docs/en/3_full_detail_tutorials/01_cann_modeling/07_theta_sweep_place_cell.ipynb.bak`

## 🔁 Rename / align filenames
- [x] Rename `docs/en/3_full_detail_tutorials/02_data_analysis/flipflop_tutorial.ipynb`
      → `docs/en/3_full_detail_tutorials/02_data_analysis/03_flipflop_tutorial.ipynb`

## 🧭 Update English toctree indexes
- [x] `docs/en/3_full_detail_tutorials/02_data_analysis/index.rst`
  - [x] Ensure order: `01_asa_pipeline`, `02_cann1d_bump_fit`, `03_flipflop_tutorial`, `04_cell_classification`
- [x] `docs/en/3_full_detail_tutorials/04_pipeline/index.rst`
  - [x] Ensure order: `03_asa_gui` (tutorial 1), `01_asa_tui` (tutorial 2, legacy), `02_model_gallery_tui` (tutorial 3)
  - [x] Add note: GUI recommended, TUI legacy

## 📌 Content alignment / wording
- [ ] README.md (EN) already updated to recommend GUI; verify consistency with docs
- [ ] Add/align citations in EN where zh recently added :cite:p: notes
- [ ] Remove references to TUI as primary entrypoint in EN docs

## ✅ Optional checks
- [ ] Build docs (EN) after alignment to ensure no missing toctree targets
- [ ] Validate links on https://routhleck.com/canns/en
