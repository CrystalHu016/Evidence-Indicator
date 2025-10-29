# Scripts Directory

This directory contains utility scripts for dataset processing and project maintenance.

## Dataset Processing Scripts

### extract_jsquad_10k.py
Extract random 10,000 samples from the full JSQuAD training dataset.

**Usage:**
```bash
python3 scripts/extract_jsquad_10k.py
```

**Input:** `data/jsquad_train_full.json`
**Output:** `data/jsquad_train_10k.json`

### process_jsquad_add_answer_end.py
Add `answer_end` field to JSQuAD datasets based on the correct formula:
- Formula: `answer_end = answer_start + len(answer_text) - 1`
- Extraction: `answer_text = context[answer_start : answer_end + 1]`

**Usage:**
```bash
python3 scripts/process_jsquad_add_answer_end.py
```

**Processes:**
- `data/jsquad_train_10k.json`
- `data/jsquad_train_full.json`

## Archived Test Scripts

The `archived_tests/` subdirectory contains temporary test scripts that are no longer actively used but kept for reference:

- `test_airflow_query.py` - Test script for airflow-related queries
- `test_new_prompt.py` - Test script for prompt development
- `test_nyubai_query.py` - Test script for nyubai (rainy season entry) queries

These files are archived and not maintained for current development.
