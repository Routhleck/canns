# CANNs Documentation Draft Workspace

This directory contains all draft materials and planning documents for the CANNs documentation refactoring project.

## 📁 Directory Structure

```
docs_draft/
├── planning/           # Question files and planning documents
│   ├── tier1_why_canns_questions.md
│   ├── tier2_basic_intro_questions.md
│   └── tier3_core_concepts_questions.md
├── drafts/             # Draft documentation (markdown and notebooks)
│   ├── *.md           # Original markdown sources
│   └── *.ipynb        # Generated Jupyter notebooks
├── scripts/            # Conversion and automation scripts
│   └── convert_md_to_nb.py
└── summaries/          # Completion summaries and status reports
    ├── tier2_completion_summary.md
    └── IMPLEMENTATION_COMPLETE.md
```

## 🎯 Purpose

This workspace is used for:
- **Planning**: Asking and answering questions to guide documentation development
- **Drafting**: Writing and iterating on documentation content
- **Converting**: Transforming markdown to Jupyter notebooks
- **Tracking**: Recording progress and completion status

## 📝 Workflow

1. **Planning Phase**: Create question files in `planning/`
2. **Answer Phase**: User fills in answers
3. **Drafting Phase**: Generate draft documentation in `drafts/`
4. **Conversion Phase**: Convert markdown to notebooks using `scripts/`
5. **Integration Phase**: Move finalized content to main `docs/` directory
6. **Summary Phase**: Document completion in `summaries/`

## 🔄 Current Status

### Completed
- ✅ Tier 1: Why CANNs? (1 notebook)
- ✅ Tier 2: Basic Intro (5 notebooks)
- ✅ All notebooks integrated into `docs/en/`

### In Progress
- 🟡 Tier 3: Core Concepts (question file ready, awaiting answers)

### Planned
- ⏸️ Tier 4: Full Details Tutorials

## 🛠️ Scripts

### `scripts/convert_md_to_nb.py`
Converts markdown documentation to Jupyter notebooks.

**Usage**:
```bash
cd docs_draft/scripts
uv run python convert_md_to_nb.py
```

---

**Last Updated**: 2025-11-15
**Status**: Active development
