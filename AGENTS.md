# Repository Guidelines

## Project Structure & Module Organization
- `TrainedModel/emotion/` holds the vision emotion-recognition pipeline (data split, baselines, MoR + EfficientNet experiments). Keep checkpoints in per-run subfolders and exclude `.pth` files from git.
- `TrainedModel/text/` contains the MoR-enhanced BERT classifier for MBTI text inference; keep the numeric script order aligned with the data workflow.
- `mbti_web_app/` is reserved for the Flask API and UI shell. Place `app.py`, blueprints, and static assets here so serving logic stays separate from training code.
- `README.md` owns high-level onboarding; update it when adding endpoints, models, or environment prerequisites.

## Build, Test, and Development Commands
- `python -m venv venv` followed by `venv\Scripts\activate` to work inside an isolated environment.
- `pip install -r requirements.txt` after editing dependencies; make sure any new packages are added before opening a PR.
- `python TrainedModel\emotion\3.3MoR_vision.py` trains the EfficientNet + CBAM model; pass in your own `DATA_ROOT` via env vars or argparse when you extend it.
- `python TrainedModel\text\4.3MoR_Text.py` fine-tunes the text model; cache Hugging Face weights locally to keep CI jobs responsive.
- From `mbti_web_app\`, run `python app.py` to launch the Flask server and manually hit endpoints before submitting changes.

## Coding Style & Naming Conventions
- Follow PEP 8, use 4-space indentation, and add type hints or docstrings when exposing new public functions.
- Preserve the numeric filename prefixes (`1Datasplit.py`, `3.3MoR_vision.py`) so collaborators recognize pipeline order.
- Keep configuration constants at the top of each script and provide override hooks instead of hard-coding contributor-specific paths.

## Testing Guidelines
- Add smoke tests under `tests/` (e.g., `pytest tests/test_api_smoke.py`) that exercise `/test`, `/api/process_text`, and `/api/process_frame` with mocked payloads.
- Log validation metrics for every training change and commit lightweight reports (confusion matrices, accuracy/F1) to `runs/<date>/` or the PR description.
- Replace local filesystem paths with config entries to keep tests reproducible across environments and CI.

## Commit & Pull Request Guidelines
- Write concise, present-tense subjects (`feat: add mor router`, `fix: adjust vision loss`) and keep each commit focused.
- Pull requests must describe the motivation, list verification steps (commands run, screenshots, or notebooks), and link any related issues.
- Request review before merging and confirm that linting and smoke tests finish cleanly.

## Model Assets & Configuration
- Store large weights under `TrainedModel/**/checkpoints/` and keep them out of version control; publish download links in the PR or wiki.
- Document new environment variables in `README.md` and supply a `.env.example` whenever you add secrets or external service keys.
