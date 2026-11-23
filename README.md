# CIFAKE-Guard v2

Detects AI-generated images on the CIFAKE benchmark. CIFAKE pairs 32×32 CIFAR-10 photos with Stable Diffusion v1.4 images.
The system is a **calibrated ensemble of four detectors** with explanations, and a web dashboard.

| Model | Type | Params |
|---|---|---|
| LightCNN | CNN trained from scratch (from v1) | 0.58 M |
| EfficientNet-B0 | ImageNet transfer learning (from v1) | 4.01 M |
| TinyViT | Vision Transformer trained from scratch (from v1) | 1.80 M |
| **SFHybrid** (new) | Spatial-frequency CNN-Transformer hybrid: SE-attention CNN + FFT-spectrum token + self-attention fusion | 0.72 M |

The v2 additions:
- **Evaluation:** one shared protocol for every model, including a de-duplicated test split, a robustness benchmark and an unseen-generator benchmark (DDPM).
- **Calibration and ensembling:** temperature scaling plus five ensemble strategies, all fitted on a separate calibration split.
- **Explanations:** Grad-CAM, Grad-CAM++, Integrated Gradients, attention rollout and frequency analysis, checked against each other with a quantitative faithfulness test.
- **App:** a FastAPI backend and dashboard with downloadable reports.
- **Engineering:** tests, configs and documentation.

➡️ **Results, methodology and limitations:** [docs/RESEARCH.md](docs/RESEARCH.md).
Auto-generated metric tables are in `runs/cpu_experiment/results/RESULTS.md`.

### Headline results

These were measured on the full official test split of 20,000 images. The training set was compute-limited to 20k images on a laptop CPU.

| | Accuracy | ROC-AUC | Mean accuracy under 24 perturbations | Unseen generator (DDPM) AUC |
|---|---|---|---|---|
| Best single model (EfficientNet-B0) | 0.957 | 0.991 | 0.813 | 0.43 |
| Weighted ensemble | **0.967** | **0.995** | 0.822 | 0.44 |
| Weighted ensemble of robust-trained members (**deployed**) | **0.967** | **0.995** | **0.874** | 0.48 |

**The ensemble's gain is statistically significant** (McNemar p < 10⁻¹⁷).

**Honest negative results:**

| Finding | Evidence |
|---|---|
| The new SFHybrid does not beat the simple LightCNN on clean data | Its frequency branch gives only a small gain over the same model without it |
| Standard training is fragile to resizing and blurring | Accuracy falls to near chance (≈0.5–0.6) |
| No model generalises to an unseen generator | Fewer than 7 % of DDPM fakes are detected |

Details are in [docs/RESEARCH.md](docs/RESEARCH.md).

---

## Quick start (local, Windows / Linux / macOS)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

**Use the trained models.** The app and CLI load `runs/cpu_experiment_robust_ensemble/` by default. That run uses the robust-trained LightCNN and SFHybrid plus EfficientNet-B0 and TinyViT; see RESEARCH §5.4 for why. All seven trained checkpoints are in `runs/cpu_experiment/`. To serve the standard ensemble instead, set `CIFAKE_RUN_DIR=runs/cpu_experiment`.

```bash
uvicorn app.main:app --port 8000          # open http://localhost:8000
python scripts/predict.py some_image.jpg --report-dir reports/   # command-line alternative
```

**Reproduce everything.** CIFAKE downloads automatically through `kagglehub` (about 105 MB, no Kaggle account needed).

```bash
python scripts/audit_data.py --config configs/cpu_experiment.yaml          # leakage / shortcut audit
python scripts/train.py      --config configs/cpu_experiment.yaml --threads 4
pip install diffusers
python scripts/generate_unseen_fakes.py --n 1000 --steps 25                # unseen-generator benchmark
python scripts/evaluate.py   --config configs/cpu_experiment.yaml --unseen data_unseen/ddpm_cifar10
# deployed robust ensemble (copy LightCNN_robust, SFHybrid_robust, EfficientNetB0, TinyViT checkpoints + results/splits.json first):
python scripts/evaluate.py   --config configs/cpu_experiment.yaml --unseen data_unseen/ddpm_cifar10 --skip faithfulness \
       --set paths.output_dir=runs/cpu_experiment_robust_ensemble "evaluation.ensemble_members=[LightCNN_robust,SFHybrid_robust,EfficientNetB0,TinyViT]"
pytest                                                                     # 59 tests, about 1 min
```

Fast end-to-end sanity check (about 5 minutes; its numbers are meaningless): `--config configs/smoke.yaml`.

**Data location.** By default the data comes through kagglehub. To use an existing copy instead, set `paths.data_root` in the config or set the environment variable `CIFAKE_DATA_ROOT`. Either must point to a folder that contains `train/{REAL,FAKE}` and `test/{REAL,FAKE}`.

## Google Colab (GPU, full dataset)

1. Zip this folder as `CIFAKE_Guard_v2.zip` and upload it to Colab, or put the folder on Google Drive.
2. Open [notebooks/CIFAKE_Guard_v2_Colab.ipynb](notebooks/CIFAKE_Guard_v2_Colab.ipynb) and select *Runtime → T4 GPU*.
3. Run all cells. The notebook runs audit → train → unseen benchmark → evaluate on `configs/full_gpu.yaml`, optionally launches the dashboard through Colab's port proxy, and downloads the results.

> The full-data GPU protocol is provided but was **not** executed by the v2 author (no GPU was available). All reported numbers come from `configs/cpu_experiment.yaml`.

## Project layout

```
cifake_guard/            core library
  config.py              YAML config + defaults + dotted CLI overrides (--set a.b=c)
  data.py                CIFAKE discovery, cached uint8 arrays, deterministic disjoint splits, fast loader
  transforms.py          augmentations + robustness perturbations (shared code)
  models/                LightCNN, EfficientNet-B0, TinyViT, SFHybrid (+ registry)
  training.py            AdamW + warmup/cosine, AMP, early stopping, self-describing checkpoints
  metrics.py             all metrics, ECE, bootstrap CIs, McNemar
  ensemble.py            temperature scaling, 5 ensemble strategies, agreement
  explain.py             Grad-CAM(++), Integrated Gradients, attention rollout, spectra, deletion test
  inference.py           preprocessing that matches CIFAKE storage, Predictor service, plain-language summary
  report.py              self-contained HTML report
app/                     FastAPI backend (main.py) + dashboard (static/)
scripts/                 audit_data, train, evaluate, generate_unseen_fakes, predict
configs/                 smoke.yaml, cpu_experiment.yaml (reported), full_gpu.yaml (Colab)
tests/                   pytest suite (models, data, metrics, calibration, explainers, API)
notebooks/               Colab runner + original v1 notebook (unchanged, for reference)
runs/cpu_experiment/     checkpoints/, results/ (CSV/JSON/RESULTS.md), plots/
docs/RESEARCH.md         methodology, experiments, results, limitations, future work
```

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | dashboard |
| GET | `/api/health` | status and loaded models |
| GET | `/api/models` | model metadata, temperatures, ensemble weights, test metrics |
| POST | `/api/predict?explain=true` | multipart `file` → verdict, per-model and ensemble probabilities, agreement, explanations (base64 PNG), summary, `report_id` |
| GET | `/api/report/{id}?format=html\|json` | downloadable report |
| GET | `/api/results`, `/api/plots/{name}` | evaluation artifacts for the dashboard |

Interactive docs are at `/docs`.

**Input validation:**

| Condition | Response |
|---|---|
| Content type is not an image | 415 |
| File larger than 10 MB (`CIFAKE_MAX_UPLOAD_MB`) | 413 |
| Empty or undecodable file | 400 |
| Image over 40 MP (decompression-bomb guard) | rejected |
| Models missing | `/api/health` reports degraded; predictions return 503 |

## What changed from v1 (summary of fixes)

| v1 issue | v2 fix |
|---|---|
| The FAKE class has **1,336 exact duplicate pairs**; 378 test FAKE images are pixel-identical to training images (leakage) | `scripts/audit_data.py` finds them; metrics are also reported on a de-duplicated test split |
| Validation split doubled as the only held-out tuning set | Separate **calibration** split for temperature scaling and ensemble fitting |
| EfficientNet "frozen" epoch still updated BatchNorm running statistics | Frozen backbone is kept in eval mode (regression test included) |
| Raw, uncalibrated probabilities | Temperature scaling, ECE and reliability diagrams |
| No ensemble, no explanations, no robustness or generalisation tests, no app | All added (see RESEARCH.md) |
| `best_state=None` crash if validation never improved; silent NaN losses | Explicit errors |
| TinyViT used `nn.TransformerEncoder` (attention not accessible) | Same architecture, explicit attention modules, so attention rollout works |
| Notebook-only, global state, no tests | Package + scripts + configs + 59 tests |

The original notebook is preserved in `notebooks/original/`.
