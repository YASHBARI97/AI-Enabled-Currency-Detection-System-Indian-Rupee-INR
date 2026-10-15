For Readable Preview ---> Code 

## AI Enabled Currency Detection System (INR)

## Project : 
Working prototype; saved model and offline evaluation artifacts are present. The frontend development server was reported as ready at `http://localhost:3000`. In the development session, Windows Application Control blocked direct execution of `uvicorn.exe`; starting the backend through the virtual environment's Python module runner is the documented workaround to try. This is an denomination classifier. It does not detect counterfeit currency, verify authenticity, or establish legal-tender status.

1. Executive summary

This project implements an end-to-end web application that accepts one JPG/PNG image of an Indian banknote and predicts one of six denominations: INR 10, 20, 50, 100, 200, or 500. The main saved classifier is a transfer-learned MobileNetV3-Small CNN initialized from torchvision ImageNet-1K weights. Its saved test evaluation reports 98.88% accuracy and 0.9859 macro-F1 on 893 images. A validation-selected confidence threshold is stored as 0.30; the saved validation and test results show 100% coverage at that threshold, so abstention did not reduce the evaluated sample count.

The repository also contains an HSV/edge-feature RBF-SVM baseline, an EfficientNet-B0 checkpoint without a recorded summary or evaluation, a FastAPI inference API, and a Next.js/React upload UI. Saved dataset statistics record 6,252 images divided into 4,464 train, 895 validation, and 893 test images across 2,394 groups.

  2. Goals, scope, and non-goals

 # Goals

- Classify a single banknote image into a supported INR denomination.
- Return a confidence score and top three candidate classes.
- Avoid forcing a prediction if the maximum score is below a configurable threshold.
- Provide a browser upload flow, API status, clear input errors, and a disclaimer.
- Use grouped data splitting to reduce near-duplicate leakage between train, validation, and test sets.

 # Out of scope

- Counterfeit detection or banknote authenticity verification.
- Legal-tender determination.
- Coins, foreign currency, multiple notes in one image, and unsupported denominations or designs.
- A measured robustness guarantee for lighting, wear, backgrounds, or capture devices.

  3. System architecture

```text
Browser (Next.js UI, port 3000)
       │ same-origin /api/ rewrite
     
FastAPI API (port 8000)
       │ validates and decodes image in memory
       
Predictor → preprocessing → selected model → confidence/abstention decision
```

The API exposes `GET /health` and `POST /predict`. By default, the frontend proxies API requests to `http://127.0.0.1:8000`; configure `API_URL` if the backend is elsewhere. Uploads are size bounded and validated for supported image format; the API does not save the image. Optional prediction history stores prediction metadata, not the uploaded image.

 # Main components

| Component | Location | Responsibility |
|---|---|---|
| API | `app/api.py` | Model startup, health endpoint, upload validation flow, predictions, HTTP status mapping |
| Dataset import/preparation | `scripts/import_coco.py`, `scripts/prepare_data.py`, `src/data/` | COCO import, image checks, duplicate grouping, grouped splits |
| CNN model/training | `src/models/factory.py`, `scripts/train.py` | Transfer learning, training, validation model selection |
| Classical baseline | `src/models/baseline.py`, `scripts/train_baseline.py` | HSV histogram and edge-grid features with RBF-SVM |
| Prediction and decisions | `src/inference/` | Preprocessing/inference, top-k scores, abstention, optional history |
| Evaluation | `scripts/evaluate.py`, `src/evaluation/metrics.py` | Classification metrics, threshold selection, latency and coverage reports |
| Web interface | `frontend/app/`, `frontend/components/`, `frontend/lib/` | Upload, preview, status, result and session-only history |

  4. Dataset and preparation

 Recorded dataset profile

The checked-in `reports/data_stats.json` records the following image counts. The dataset card does not record a source URL, licence/permission evidence, collection date, or capture-condition labels; these details cannot be established from the available project records and should be completed by the dataset owner.

| Denomination | Total | Train | Validation | Test |
|---|---:|---:|---:|---:|
| INR 10 | 433 | 309 | 62 | 62 |
| INR 20 | 1,192 | 852 | 170 | 170 |
| INR 50 | 1,282 | 915 | 183 | 184 |
| INR 100 | 1,155 | 825 | 165 | 165 |
| INR 200 | 999 | 711 | 144 | 144 |
| INR 500 | 1,191 | 852 | 171 | 168 |
| Total | 6,252 | 4,464 | 895 | 893 |

The reported split counts sum to the total. `reports/data_stats.json` records zero unreadable images removed and 2,394 groups. The class distribution is imbalanced: INR 10 has 433 images compared with 1,282 for INR 50.

 Import, cleaning, and split methodology

- `scripts/import_coco.py` maps the COCO annotations and denomination-like filename prefixes into class folders and records source-image groups in `reports/groups.csv`.
- `scripts/prepare_data.py` verifies image readability, assigns near-duplicate groups using perceptual hashes (default Hamming distance 6), then makes stratified, group-aware splits with seed 42. If the importer group manifest is provided, its source groups are used.
- The split builder checks that groups do not cross splits. This reduces leakage from duplicate/derived images, but does not prove independence by camera, session, physical note, or source dataset.
- The project documentation prescribes a 70/15/15 split. The saved counts are close to that target; group constraints and per-class rounding can make exact proportions differ.
- Augmentation is applied to training data only. Validation and test use deterministic evaluation transforms.

 Data provenance limitation

The source/URL and licence fields in `docs/DATASET_CARD.md` are blank. The report therefore cannot assert permission to redistribute or characterize the dataset's collection conditions. Verify provenance and use rights before sharing the image files or relying on the results beyond the project.

  5. Models and training

 Main model: MobileNetV3-Small

The selected saved model is a six-class MobileNetV3-Small at 224×224 pixels, initialized from torchvision's ImageNet-1K pretrained weights and fine-tuned for this task. The saved experiment configuration specifies:

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 0.0005 |
| Weight decay | 0.0001 |
| Batch size | 32 |
| Maximum epochs | 25 |
| Early-stopping patience | 5 epochs |
| Scheduler | Cosine annealing |
| Loss | Class-weighted cross-entropy |
| Seed | 42 |
| Training device recorded | CPU |
| PyTorch version recorded | 2.14.1+cpu |
| Training duration recorded | 2,614.2 seconds (about 43.6 minutes) |
| Image augmentation | Up to 15° rotation, random crop scale 0.8–1.0, brightness/contrast/saturation variation, blur |
| Model version | 1.0.0 |

Training selected and saved the checkpoint with the best validation macro-F1. The recorded run used all 25 configured epochs. The history file's final epoch shows train accuracy 99.91%, validation accuracy 98.44%, and validation macro-F1 0.9827. The summary's best validation macro-F1 is also 0.9827328. Training accuracy substantially exceeds validation accuracy, so the usual generalization gap is present; the held-out test score is the more relevant performance estimate for this split.

 # Comparison and other artifacts

| Model/artifact | Method | Available evidence | Status |
|---|---|---|---|
| HSV + RBF-SVM baseline | HSV color histogram and edge-grid features; SVM C=10 | Validation accuracy 65.70%, macro-F1 0.6567 | Trained and summary saved; no test metrics found |
| MobileNetV3-Small | ImageNet-1K transfer learning | Validation and test metrics, training history and summary | Main evaluated model |
| EfficientNet-B0 | ImageNet-pretrained architecture configured in `configs/efficientnet_b0.yaml` | `best_model.pt` checkpoint exists | No matching saved summary, metadata, history or evaluation metrics found; comparative score unknown |
| MobileNetV3-Small no-augmentation ablation | Config exists in `configs/mobilenetv3_noaug.yaml` | No saved run artifacts found | Not evidenced as completed |

The only completed model comparison supported by saved numeric results is the validation comparison between MobileNetV3-Small and the SVM baseline. Do not present EfficientNet or the no-augmentation run as having measured results without producing their run records.

  6. Validation, threshold selection, and test protocol

Validation is used to select the neural-network checkpoint and the abstention threshold. The evaluator's default threshold-selection objective is to choose the lowest threshold on its evaluated grid that reaches at least 95% accuracy among answered validation examples while retaining at least 70% coverage; if no grid value meets the target, it chooses the best accuracy subject to minimum coverage. The saved model metadata and both report JSON files record a threshold of 0.30.

The stored metrics at threshold 0.30 report coverage of 1.0 on both validation and test. Thus every evaluated image received a class prediction at this threshold. The system's abstention logic is implemented, but these reported results do not demonstrate a reduction in errors from abstaining. The saved validation/test result files are consistent with a threshold of 0.30; the exact original command history is not recorded.

The intended protocol is to freeze the selected model and validation threshold, then evaluate the test split once. The repository contains test metrics; the exact evaluation date and shell command history are not recorded, so adherence to the one-time rule cannot be independently audited from artifacts alone.

  7. Model evaluation results

 # Overall results

| Metric | Validation (n=895) | Test (n=893) |
|---|---:|---:|
| Accuracy | 0.9844 (98.44%) | 0.9888 (98.88%) |
| Macro precision | 0.9809 | 0.9868 |
| Macro recall | 0.9847 | 0.9851 |
| Macro-F1 | 0.9827 | 0.9859 |
| Weighted-F1 | 0.9844 | 0.9888 |
| Threshold | 0.30 | 0.30 |
| Coverage | 100% | 100% |
| Accuracy among answered samples | 98.44% | 98.88% |
| Mean inference latency, including preprocessing | 12.28 ms | 13.80 ms |
| 95th percentile latency | 15.85 ms | 18.68 ms |

Latency was measured over the corresponding evaluation split on the recorded environment. It should not be treated as a deployment benchmark across other hardware, concurrency levels, image sizes, or network conditions.

 # Test classification report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| INR 10 | 0.9672 | 0.9516 | 0.9593 | 62 |
| INR 100 | 0.9819 | 0.9879 | 0.9849 | 165 |
| INR 20 | 0.9825 | 0.9882 | 0.9853 | 170 |
| INR 200 | 1.0000 | 1.0000 | 1.0000 | 144 |
| INR 50 | 0.9892 | 0.9946 | 0.9919 | 184 |
| INR 500 | 1.0000 | 0.9881 | 0.9940 | 168 |
| Macro average | 0.9868 | 0.9851 | 0.9859 | 893 |
| Weighted average | 0.9888 | 0.9888 | 0.9888 | 893 |

 # Test confusion matrix

Rows are true classes and columns are predicted classes. Class order is `INR_10`, `INR_100`, `INR_20`, `INR_200`, `INR_50`, `INR_500`.

| True \ Predicted | INR 10 | INR 100 | INR 20 | INR 200 | INR 50 | INR 500 |
|---|---:|---:|---:|---:|---:|---:|
| INR 10 | 59 | 0 | 3 | 0 | 0 | 0 |
| INR 100 | 0 | 163 | 0 | 0 | 2 | 0 |
| INR 20 | 2 | 0 | 168 | 0 | 0 | 0 |
| INR 200 | 0 | 0 | 0 | 144 | 0 | 0 |
| INR 50 | 0 | 1 | 0 | 0 | 183 | 0 |
| INR 500 | 0 | 2 | 0 | 0 | 0 | 166 |

There are 10 incorrect predictions in the 893-image test set. The largest observed confusion is three INR 10 examples classified as INR 20. This is a description of the recorded split, not a general guarantee about future images.

 # Validation classification report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| INR 10 | 0.9531 | 0.9839 | 0.9683 | 62 |
| INR 100 | 0.9880 | 0.9939 | 0.9909 | 165 |
| INR 20 | 0.9881 | 0.9765 | 0.9822 | 170 |
| INR 200 | 0.9728 | 0.9931 | 0.9828 | 144 |
| INR 50 | 0.9834 | 0.9727 | 0.9780 | 183 |
| INR 500 | 1.0000 | 0.9883 | 0.9941 | 171 |
| Macro average | 0.9809 | 0.9847 | 0.9827 | 895 |

 # Metric definitions

- Accuracy: correct predictions divided by all predictions.
- Precision (per class): of samples predicted as a class, the fraction that truly belong to it.
- Recall (per class): of samples truly in a class, the fraction correctly predicted.
- F1-score: harmonic mean of precision and recall.
- Macro average: unweighted average over the six classes; useful with class imbalance.
- Weighted average: class metrics averaged using each class's sample count.
- Coverage: fraction of samples with maximum model score at or above the abstention threshold.
- Latency: measured time for preprocessing and model inference in the evaluation script.

Softmax confidence is a model score, not a calibrated probability. No calibration analysis (such as reliability curves or expected calibration error) is recorded.

  8. Testing and verification status

The repository includes backend tests for decisions, inference/API behavior, metrics, preprocessing, splits, and image validation under `tests/`. The frontend includes a validation test file and scripts for type checking and tests. The README documents `pytest -q`, `npm run typecheck`, and `npm test` as checks.

This report does not claim these automated tests were run. No test-run output or CI report was present in the examined project records. The saved model-evaluation metrics above are offline evaluation artifacts, not software test-suite results. Record test commands, dates, environment and pass/fail counts when those checks are run.

The frontend startup output shared during project work showed Next.js 16.4.1 ready at `http://localhost:3000`. It also showed a React 18 support deprecation warning and Next.js making suggested `tsconfig.json` changes. This confirms the dev server started, but does not by itself verify the end-to-end prediction flow. A prior backend launch attempt was blocked by Windows Application Control because it could not confirm the publisher of `uvicorn.exe`; use the Python module invocation in Section 11 or the organization's approved application-control process.

  9. Application behavior and API

 # User interface

The UI supports drag-and-drop/file selection, image preview, type/size prechecks, model status, denomination/confidence display, top-k scores, low-confidence state, errors, and session-only prediction history. The browser sends requests through the Next.js API rewrite. Client checks improve usability; the API remains responsible for server-side validation.

 # API routes

| Route | Purpose | Typical response |
|---|---|---|
| `GET /health` | Reports model availability, version, classes, threshold and max upload size | `ok` or `model_unavailable` |
| `POST /predict` | Accepts multipart field `file` and returns prediction information | denomination, confidence, latency, top-k, disclaimer, or low-confidence result |

Error status mapping includes empty input 400, unsupported format 415, unreadable image 422, too-large upload 413, inference error 500, and model unavailable 503. The configured upload limit defaults to 5 MB. See `app/api.py` and `src/inference/` for the implementation.

  10. Results, interpretation, and limitations

The saved test metrics indicate strong performance on this project's held-out, grouped split for the six represented denominations. The SVM validation score is substantially lower than the MobileNetV3-Small validation score, supporting the choice of the CNN as the primary model for this dataset. The results apply only to the supplied data and split; they do not demonstrate performance on new note series, different cameras, real-world capture conditions, or counterfeit notes.

Key limitations:

1. Dataset source, licence, and collection/capture metadata are undocumented in the dataset card.
2. Grouping is based on available source identifiers/perceptual hashes and does not establish independent acquisition sessions.
3. INR 10 has the fewest examples and the lowest test recall (95.16%).
4. Threshold 0.30 yielded 100% coverage in the saved split, so abstention behavior was not exercised by these reported evaluations.
5. Confidence calibration was not measured.
6. EfficientNet-B0's checkpoint has no recorded evaluation and cannot be compared numerically.
7. The prototype classifies denomination only and must not be used to authenticate currency or support consequential decisions without human verification.
8. Test results are from one split and do not quantify run-to-run or dataset-shift uncertainty.

  11. How to run

Run commands from the project root in PowerShell. The virtual environment shown in the project is `.venv`.

 # Backend

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.api:app --reload --port 8000
```

This uses Python to launch Uvicorn and avoids directly invoking `uvicorn.exe`. If dependencies are missing:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Check `http://127.0.0.1:8000/health` and API docs at `http://127.0.0.1:8000/docs`.

 # Frontend

In a separate terminal:

```powershell
Push-Location frontend
npm run dev
```

Open `http://localhost:3000`. Keep both servers running for predictions. `Ctrl+C` stops each server.

  12. Reproduction commands

The repository's documented workflow is:

```powershell
python -m scripts.import_coco --src data/original --apply
python -m scripts.prepare_data --data-dir data/raw --groups-csv reports/groups.csv
python -m scripts.train_baseline --config configs/baseline_svm.yaml
python -m scripts.train --config configs/mobilenetv3_small.yaml
python -m scripts.evaluate --model-dir artifacts/mobilenetv3_small_224 --split val --select-threshold
python -m scripts.evaluate --model-dir artifacts/mobilenetv3_small_224 --split test
```

For a new experiment, preserve the existing evaluation artifacts and record the exact config, software versions, hardware, commands, and dates. Do not tune the threshold or model against the test split.

  13. Reproducibility and artifact inventory

Key saved evidence:

| Artifact | Contents |
|---|---|
| `data/splits/splits.csv` | Per-image label, split, group and path |
| `reports/data_stats.json` | Dataset totals and per-class split counts |
| `reports/groups.csv` | Imported source grouping manifest (if retained) |
| `artifacts/mobilenetv3_small_224/best_model.pt` | Main CNN checkpoint |
| `artifacts/mobilenetv3_small_224/model_meta.json` | Architecture, classes, image size, threshold and version |
| `artifacts/mobilenetv3_small_224/experiment_summary.json` | Config, best validation macro-F1, epochs, hardware, duration and Torch version |
| `artifacts/mobilenetv3_small_224/history.csv` | Per-epoch training and validation history |
| `reports/metrics_mobilenetv3_small_224_val.json` | Validation metrics and confusion matrix |
| `reports/metrics_mobilenetv3_small_224_test.json` | Test metrics and confusion matrix |
| `reports/confusion_mobilenetv3_small_224_{val,test}.png` | Rendered confusion matrices |
| `reports/errors_mobilenetv3_small_224_{val,test}.csv` | Misclassified validation/test examples |
| `reports/coverage_curve_mobilenetv3_small_224_{val,test}.csv` | Accuracy/coverage at score thresholds |
| `artifacts/baseline/` | SVM model, metadata, and validation summary |
| `artifacts/efficientnet_b0_224/best_model.pt` | EfficientNet checkpoint; no matching evaluation record found |

Paths in split and error CSV files can disclose local file structure. Review before sharing. Dataset images and model weights may also carry licensing or distribution constraints.

  14. Suggested next steps

- Complete the dataset provenance/licence and collection fields in `docs/DATASET_CARD.md`.
- Run and save software test/type-check outputs separately from model metrics.
- Add documented run metadata and evaluate EfficientNet-B0 if it is intended as a comparison.
- Evaluate the no-augmentation ablation if making claims about augmentation benefit.
- Assess probability calibration and set an abstention threshold based on an explicitly documented validation objective.
- Add capture-condition labels and test performance across lighting, backgrounds, sides, wear, and note designs.
- Confirm the selected artifact copied to `artifacts/best_model.pt` matches the evaluated MobileNet checkpoint and metadata before serving.
- Keep the authenticity disclaimer visible and require manual verification for any important use.

  15. Source files consulted

This report was assembled from the repository's README and execution guide, dataset/experiment documentation, training/evaluation code and configs, saved dataset statistics, model metadata and summaries, training history, and validation/test metric JSON files. Where these sources did not contain evidence (notably dataset provenance, test-suite output, and EfficientNet evaluation), the report states that explicitly rather than filling in assumed values.

## Workflow

Computer-vision prototype that predicts the denomination of a single Indian banknote from a JPG/PNG image,
returns a confidence score, and abstains ("Unable to classify confidently") when unsure.
Stack: Python (PyTorch / scikit-learn) + FastAPI backend, Next.js (React + TypeScript) frontend.

> Scope: educational prototype. It identifies denomination only. It does not verify authenticity,
> legal-tender status, or detect counterfeits.

## Architecture
```
Browser --> Next.js (React UI, :3000) --/api/* proxy--> FastAPI (:8000) --> Predictor --> model + abstention
```
The browser only talks to the Next.js origin. `frontend/next.config.mjs` rewrites `/api/*` to the FastAPI service
(`API_URL`), so no CORS setup is needed and the backend address stays server-side.

## Approach
| Stage | Choice | Why |
|---|---|---|
| Baseline | HSV colour histogram + edge-grid features -> RBF-SVM (`configs/baseline_svm.yaml`) | Colour is a strong cue; cheap, torch-free reference point |
| Main model | Transfer learning: MobileNetV3-Small / EfficientNet-B0 (`configs/*.yaml`) | Small data + single-note classification -> fine-tuning pretrained CNNs is the sensible first choice |
| Decision | Max-softmax with abstention threshold chosen on validation data | Avoids forcing a label on unclear input |
| Extension (optional) | Two-stage detect->crop->classify, or YOLO | Only if your data contains clutter / multiple notes |

Decide classification vs. detection after inspecting your data and document the reasoning in the report.

## Setup
Requirements: Python 3.10+, Node.js 18.17+ (20 LTS recommended).
```bash
# Backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # export these variables or load them with your shell/process manager

# Frontend
cd frontend && npm install && cp .env.local.example .env.local && cd ..

# Verify
pytest -q                                       # 29 backend tests (no torch / real data needed)
cd frontend && npm run typecheck && npm test    # frontend checks (Node 22+ for `npm test`)
```
Tested with: Python ____ , torch ____ , Node ____ , OS ____ (fill in).

Windows activation: `.venv\Scripts\activate` (PowerShell: `.venv\Scripts\Activate.ps1`; if blocked, run
`Set-ExecutionPolicy -Scope Process Bypass` first).

VS Code shows `Import "fastapi" could not be resolved`? The editor is using a Python interpreter that lacks the
packages. Run `pip install -r requirements.txt` inside the venv, then press `Ctrl+Shift+P` -> *Python: Select Interpreter*
-> choose `.venv`. If the warning lingers: *Developer: Reload Window*.

## ML workflow
```bash
# 1. Put permitted images in data/raw/<INR_10|INR_20|INR_50|INR_100|INR_200|INR_500>/*.jpg  (see docs/DATASET_CARD.md)
python -m scripts.prepare_data --data-dir data/raw             # clean, group near-duplicates, leakage-free split

# 2. Baseline + experiments (log each in docs/EXPERIMENT_LOG.md)
python -m scripts.train_baseline --config configs/baseline_svm.yaml
python -m scripts.train --config configs/mobilenetv3_small.yaml
python -m scripts.train --config configs/efficientnet_b0.yaml
python -m scripts.train --config configs/mobilenetv3_noaug.yaml  # augmentation ablation

# 3. Choose model + threshold using VALIDATION only
python -m scripts.evaluate --model-dir artifacts/mobilenetv3_small_224 --split val --select-threshold

# 4. Freeze everything, then evaluate on TEST exactly once
python -m scripts.evaluate --model-dir artifacts/mobilenetv3_small_224 --split test

# 5. Promote the chosen model for serving
cp artifacts/mobilenetv3_small_224/best_model.pt     artifacts/best_model.pt
cp artifacts/mobilenetv3_small_224/model_meta.json   artifacts/model_meta.json
```
(For the sklearn baseline, point `CURRENCY_MODEL_PATH` at `artifacts/baseline/model.joblib` instead.)

## Run the app
Two terminals (or `make api` / `make web`):
```bash
uvicorn app.api:app --reload --port 8000     # backend: POST /predict, GET /health, docs at /docs
cd frontend && npm run dev                   # UI at http://localhost:3000
```
Production-style: `cd frontend && npm run build && npm start`.
If the backend runs elsewhere, set `API_URL` in `frontend/.env.local` (server-side only).

### UI features
Drag-and-drop or file-picker upload; client-side pre-checks (type/size) with the server as source of truth;
image preview; denomination + confidence bar with the abstention threshold marked; explicit
"Unable to classify confidently" state; top-k scores and raw JSON details; clear errors for invalid, oversized and
unreadable files; offline-service banner; model version/threshold badge from `/health`; session-only prediction
history (React state, nothing persisted, images never stored); dark mode; keyboard accessible.

## API
`POST /predict` (multipart field `file`) -> illustrative response (values come from real inference):
```json
{"status": "prediction_available", "predicted_denomination": "INR_500", "confidence": 0.93,
 "inference_ms": 42.7, "model_version": "1.0.0", "top_k": [], "disclaimer": "..."}
```
Statuses / HTTP codes: `prediction_available` 200, `low_confidence` 200, `empty_input` 400, `file_too_large` 413,
`unsupported_format` 415, `unreadable_image` 422, `inference_error` 500, `model_unavailable` 503.
`GET /health` -> `{status, model_version, classes, threshold, max_upload_mb}`.
Confidence is a softmax score, not a calibrated probability unless you evaluate calibration.

## Leakage controls
Near-duplicate images are grouped by perceptual hash (`src/data/dedupe.py`); whole groups go to one split.
Add `--groups-csv` with real recording/session IDs if you have them (stronger than hashing).
Splits are stratified (`StratifiedGroupKFold`); `assert_no_leakage` runs inside `prepare_data`.
Augmentation is training-only; the test split is used once after model and threshold are frozen.
No horizontal flips and no hue shifts (they destroy denomination evidence).

## Privacy and security
Uploads are validated (extension, magic bytes, size; bounded read) and never written to disk. Optional server history
(`CURRENCY_ENABLE_HISTORY=true`) stores timestamp/prediction/confidence/latency only. Errors shown to users never
include paths or stack traces. If you expose the API publicly, add authentication, rate limiting and HTTPS.

## Repository map
`app/api.py` FastAPI; `frontend/` Next.js UI (`app/`, `components/`, `lib/`); `src/data` loading/validation/splits;
`src/preprocessing` shared transforms; `src/models` baseline + factory; `src/inference` predictor, abstention, history;
`src/evaluation` metrics; `scripts/` train/evaluate; `configs/` experiments; `tests/`; `docs/` dataset card,
experiment log, report/viva guides.

## Known limitations (update with your measured findings)
Unsupported denominations/new designs, foreign currency, coins, multiple notes, heavy occlusion, and counterfeits are
out of scope. Complete the robustness table in the report with real numbers.
#   A I - E n a b l e d - C u r r e n c y - D e t e c t i o n - S y s t e m - I n d i a n - R u p e e - I N R - 
## Yash Bari - Background: - B.Tech Artificial Intelligence and Machine Learning  
 
