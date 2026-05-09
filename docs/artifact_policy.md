# Artifact Policy

This project contains source code, documentation, tests, small evaluation files, processed data, trained models, FAISS indexes, figures, reports, and reproducible submission archives. Not all of these should be committed directly to a public GitHub repository.

## Recommended To Commit

- Source code under `src/`.
- Utility and run scripts under `scripts/`.
- Tests under `tests/`.
- Documentation under `docs/` and `README*.md`.
- Small configuration and evaluation files, such as `data/eval/module6_queries.jsonl`.
- Requirements files, `.gitignore`, and license files.
- Small report tables and figures when they are useful for the paper or README.

## Use Git LFS Or GitHub Releases

Use Git LFS or GitHub Release assets for large reproducibility artifacts:

- `models/module4/review_faiss.index`
- large files under `models/module2/` or `models/module4/`
- large `data/processed/*.csv` or `data/processed/*.jsonl`
- large generated figures, reports, or packaged exports

These files are useful for full reproducibility, but they can make the Git repository heavy and slow to clone.

## Do Not Commit

- `ecommerce_sentiment_rag_final_submission.zip`
- any other `.zip`, `.7z`, or `.rar` archive
- `__pycache__/`
- `.pytest_cache/`
- `.ipynb_checkpoints/`
- `*.pyc`
- local virtual environments such as `.venv/`, `venv/`, or `env/`
- local logs such as `*.log`

Archives and caches are reproducible or machine-local artifacts. Keeping them out of Git makes the project easier to review, clone, and maintain.

## Dataset And Model Notes

The MIT license in this repository applies to source code and documentation. Dataset files, review corpus exports, and model artifacts may have separate usage conditions depending on their original source. Before publishing a full reproducibility bundle, verify that the data and model artifacts are allowed to be redistributed.

## Suggested Public Release Pattern

1. Commit code, docs, tests, requirements, and small eval files directly.
2. Put large data/model artifacts in Git LFS or a GitHub Release.
3. Link the release artifacts from the README.
4. Keep final submission archives out of Git and regenerate them with `scripts/package_final_submission.py` when needed.
