# CIFAKE-Guard v2: methodology, experiments, results and limitations

All numbers below were produced by the scripts in this repository on 2026-11-23. The machine was a laptop with an
Intel Core Ultra 5 125U (CPU only, no GPU). Nothing was hand-entered or tuned on the test set.

- **Raw outputs:** `runs/cpu_experiment/results/`. The auto-generated tables are in `RESULTS.md` and the figures in `runs/cpu_experiment/plots/`.
- **Status labels used in this document:**
  - **verified**: measured in the experiment described here.
  - **not run**: implemented, but not executed by us.
  - **proposed**: an idea only; not implemented.

---

## 1. Audit of the v1 project

v1 was a single notebook. It trained LightCNN, EfficientNet-B0 and TinyViT and reported test metrics. Contrary
to the task description, it had **no ensemble, no Grad-CAM, no backend and no frontend**. All of these were built in v2.

| # | Finding | Severity | Action |
|---|---|---|---|
| 1 | **Train/test leakage in CIFAKE itself.** The FAKE class has 1,336 exact pixel-duplicate pairs. 378 test FAKE images are identical to training FAKE images (3.8 % of test FAKE); 2 REAL test images are near-duplicates of training images. *(verified, `scripts/audit_data.py`, `results/data_audit.json`)* | High for validity | All metrics are also reported on a de-duplicated test split (§5.2) |
| 2 | No format shortcut. Both classes are 32×32 RGB baseline JPEG, *identical* quantisation tables (libjpeg quality 75, verified by re-encoding), 4:2:0 chroma, same file-size distribution (≈923 vs ≈925 bytes). *(verified)* | Good news | Used to design the unseen-generator benchmark and API preprocessing |
| 3 | EfficientNet "frozen backbone" epoch still updated BatchNorm running statistics (`requires_grad=False` does not stop BN buffers) | Bug | Frozen backbone kept in eval mode; regression test added |
| 4 | Validation split used for early stopping was the only held-out tuning data; no calibration and no room to fit an ensemble without biasing the test | Methodology | A disjoint **calibration** split was added (train / val / calib / test) |
| 5 | Probabilities never calibrated; a single fixed 0.5 threshold | Methodology | Temperature scaling, ECE, reliability diagrams |
| 6 | `model.load_state_dict(best_state)` crashed if validation loss never improved; NaN losses passed silently | Robustness | Explicit errors |
| 7 | TinyViT used `nn.TransformerEncoder`, which hides attention weights | Limits explainability | Same architecture re-implemented with explicit attention |
| 8 | Notebook global state, no tests, no configs, no CLI | Engineering | Package, scripts, YAML configs, 59 tests |

## 2. Methodology

### 2.1 Data protocol (identical for every model)

- **Dataset:** CIFAKE v3: 50k + 50k training images, 10k + 10k test images, 32×32. REAL images are CIFAR-10; FAKE images are Stable Diffusion v1.4.
- **Splits:** deterministic (seed 42) and stratified, drawn from the official training split:
  - **train:** 10,000 per class. This is a compute-limited subset of 20 % of the training split; see §6.
  - **val:** 1,000 per class, used for early stopping and checkpoint selection.
  - **calib:** 1,000 per class, used for temperature scaling and ensemble fitting.
- **Test:** the full official test split (20,000 images), plus a de-duplicated version (19,620 images).
- **Labels:** positive class = FAKE. Decision threshold 0.5 on calibrated probabilities.
- **Input contract:** every model takes 32×32 RGB in [0, 1]. Resizing and normalisation happen *inside* the model, which makes ensembling and explanation uniform.

### 2.2 Models

| Model | Origin | Design |
|---|---|---|
| LightCNN | v1 (unchanged) | 4 conv stages, BN, global average pooling; 0.58 M params |
| EfficientNet-B0 | v1 (fixed BN freeze) | ImageNet weights, input upsampled to 64 px on CPU (128 px in the GPU config), 1 frozen epoch, discriminative learning rates |
| TinyViT | v1 (re-implemented) | 4×4 patches, 6 pre-norm blocks, d=192, 3 heads; 1.80 M params |
| **SFHybrid** | **new** | **Spatial branch:** CNN with Squeeze-Excite channel attention and *no stride in the first stage*, which preserves pixel-level traces.<br>**Frequency branch:** log-magnitude FFT of the high-pass residual (x − blur(x)), followed by a position-preserving CNN, giving one "frequency token".<br>**Fusion:** [CLS, FREQ, 64 spatial tokens] pass through 2 transformer blocks.<br>0.72 M params. |

**Why SFHybrid?** Generator artefacts often show up in the Fourier spectrum (Durall et al., CVPR 2020; Frank et al., ICML 2020).
§5.6 confirms that CIFAKE's two classes have different average residual spectra. A frequency branch therefore
gives the network an explicit, low-cost view of that evidence. Self-attention fusion lets the model weigh spectral
evidence against spatial evidence. The attention paid to the frequency token is also an interpretable quantity.

**Training (shared code):**
- **Optimisation:** AdamW with linear warmup and cosine decay, gradient clipping at 1.0, and early stopping on validation loss (patience 4).
- **Augmentation:** horizontal flip plus a random shift of ±4 px with reflect padding.
- **Robust variants** (`*_robust`) additionally apply, with p = 0.5 per image, one of these: JPEG at quality 40–95, Gaussian blur with σ 0.3–1.2, Gaussian noise with σ 0.003–0.03, down-up resizing at scale 0.6–0.95, or colour jitter. These ranges deliberately differ from the evaluation severities.

### 2.3 Calibration and ensembles

- **Calibration:** per-model temperature scaling (Guo et al., 2017), fitted by NLL minimisation on the calibration split.
- **Ensembles:** five strategies, all fitted on the calibration split:
  - soft voting on raw probabilities;
  - hard majority vote;
  - mean of calibrated probabilities;
  - **weighted:** a convex combination of calibrated logits, with weights fitted by log-loss minimisation;
  - **stacking:** logistic regression on calibrated logits.
- **Default:** the dashboard uses the weighted ensemble.

### 2.4 Evaluation

The evaluation covers:
- **Classification metrics:** accuracy, balanced accuracy, precision, recall (TPR), specificity (TNR), F1, MCC.
- **Ranking metrics:** ROC-AUC, PR-AUC, EER, TPR at 1 % FPR.
- **Calibration metrics:** log-loss, Brier score, ECE (15 bins).
- **Statistics:** 95 % percentile-bootstrap confidence intervals (1,000 resamples), and McNemar's exact test.
- **Robustness:** 9 perturbation families with 2–4 severities each, on a fixed stratified subset of 2,000 test images.
- **Unseen generator:** see §2.5.
- **Complexity:** parameters, MACs, and CPU latency and throughput.
- **Explanation faithfulness:** a deletion test on 200 test images per model (§2.6).

### 2.5 Unseen-generator benchmark

CIFAKE contains only one generator, so a second one was created: `google/ddpm-cifar10-32` (Ho et al., 2020) with the DDIM sampler at 25 steps.
- **Why this generator:** it is a *pixel-space* diffusion model with a different architecture from Stable Diffusion, and it was trained on CIFAR-10. Its outputs therefore mimic the REAL class's distribution.
- **Size:** 500 fakes were generated (CPU budget). The real counterpart is 500 CIFAKE test REAL images.
- **Storage:** fakes are stored exactly like CIFAKE (32×32 JPEG, quality 75), so file format cannot act as a cue.

### 2.6 Explainability

| Method | Applies to |
|---|---|
| Grad-CAM, Grad-CAM++ | Conv models; the layer is chosen so the map is 8×8 |
| Integrated Gradients | All models. Uses a *blurred* baseline, which removes exactly the high-frequency detail where traces live. 32 steps; the completeness error is checked in the tests. |
| Attention rollout | TinyViT, SFHybrid. Also reports the share of attention on the frequency token. |
| Residual spectrum | The image's spectrum compared with the class-average spectra (a model-independent cue) |

**Faithfulness test (deletion, after Petsiuk et al. 2018):**
1. Rank pixels by saliency.
2. Replace them, most salient first, with their blurred values.
3. Measure the area under P(predicted class).

Lower is better. A random saliency map is the control.

---

## 3. Results on the official test split (20,000 images)

Calibrated probabilities, threshold 0.5 (from `RESULTS.md`).

| Model | Accuracy [95 % CI] | F1 | ROC-AUC | PR-AUC | EER | TPR@1 %FPR | ECE |
|---|---|---|---|---|---|---|---|
| EfficientNet-B0 | 0.9570 [0.9541–0.9596] | 0.9572 | 0.9908 | 0.9902 | 0.0430 | 0.802 | 0.0073 |
| LightCNN | 0.9554 [0.9525–0.9580] | 0.9555 | 0.9915 | 0.9917 | 0.0444 | 0.848 | 0.0034 |
| SFHybrid (new) | 0.9502 [0.9470–0.9531] | 0.9501 | 0.9900 | 0.9898 | 0.0500 | 0.838 | 0.0083 |
| TinyViT | 0.9159 [0.9119–0.9197] | 0.9155 | 0.9745 | 0.9748 | 0.0839 | 0.653 | 0.0177 |
| Ensemble, soft vote (raw) | 0.9635 | 0.9635 | 0.9938 | 0.9936 | 0.0369 | 0.876 | 0.0362 |
| Ensemble, majority | 0.9595 | 0.9600 | 0.9848 | 0.9770 | 0.0421 | 0.683 | 0.0174 |
| Ensemble, mean calibrated | 0.9628 | 0.9629 | 0.9937 | 0.9936 | 0.0376 | 0.875 | 0.0336 |
| **Ensemble, weighted** | **0.9673 [0.9648–0.9698]** | **0.9674** | **0.9950** | **0.9948** | **0.0325** | **0.899** | 0.0105 |
| Ensemble, stacking | 0.9674 [0.9647–0.9699] | 0.9674 | 0.9949 | 0.9948 | 0.0332 | 0.895 | 0.0044 |

**Verified findings**

1. **The learned ensembles beat every single model.** The weighted ensemble gains +1.0 point of accuracy over the best single model, EfficientNet-B0, and the confidence intervals do not overlap. McNemar's test gives 384 vs 179 discordant predictions, p = 3.5×10⁻¹⁸. Learned weighting also beats naive soft voting (0.9673 vs 0.9635), and it improves calibration (ECE 0.0105 vs 0.0362). Averaging probabilities makes them under-confident, and fitted weights correct that.
2. **The fitted weights are informative:**

   | Model | Weight |
   |---|---|
   | LightCNN | 0.43 |
   | EfficientNet-B0 | 0.43 |
   | SFHybrid | 0.10 |
   | TinyViT | 0.04 |

   The optimiser down-weights the weaker and more correlated models.
3. **The new SFHybrid did *not* beat the simpler LightCNN on clean data** (0.9502 vs 0.9554; the confidence intervals do not overlap). We report this as a negative result. Its frequency branch does help relative to the same architecture without it (§5.1), but the hybrid as a whole is not the best single model in this compute-limited setting.
4. **TinyViT is clearly the weakest** (0.916). This is consistent with ViTs trained from scratch on small data, which lack convolutional inductive biases; the model was still improving at epoch 12.
5. **Seed variance is small.** LightCNN trained with seeds 42, 43 and 44 on the identical split reached 0.9556 ± 0.0005 accuracy and 0.9917 ± 0.0002 AUC (`runs/seed_study/LightCNN_seed_study.csv`). Differences of 0.4 points or more between models are therefore well above training noise; this is measured for LightCNN and assumed for the others.

## 4. Calibration

| Model | T | ECE raw → calibrated | Log-loss raw → calibrated |
|---|---|---|---|
| EfficientNet-B0 | 1.13 | 0.0062 → 0.0073 | 0.1195 → 0.1195 |
| LightCNN | 1.21 | 0.0094 → 0.0034 | 0.1188 → 0.1169 |
| SFHybrid | 1.26 | 0.0113 → 0.0083 | 0.1298 → 0.1282 |
| TinyViT | 0.75 | 0.0349 → 0.0177 | 0.2237 → 0.2128 |

All models were already reasonably calibrated, which is typical for low-capacity models trained with early stopping.
- **Clear gain:** temperature scaling roughly halved ECE for TinyViT. TinyViT was *under*-confident (T < 1) because of label smoothing.
- **Small gains:** LightCNN and SFHybrid improved slightly.
- **Slight loss:** EfficientNet-B0 got slightly worse (0.0062 → 0.0073), within noise. Temperature scaling fitted on 2,000 images is not guaranteed to help when a model is already calibrated.

Calibration matters more for the **ensemble** than for the individual models, as the numbers in §3 show.

## 5. Ablations, robustness, generalisation and explanations

### 5.1 Does the frequency branch help? (verified)

| Model | Accuracy | ROC-AUC | TPR@1 %FPR |
|---|---|---|---|
| SFHybrid (with FFT token) | 0.9502 [0.9470–0.9531] | 0.9900 | 0.838 |
| SFHybrid_noFreq (identical, no FFT token) | 0.9465 [0.9433–0.9495] | 0.9880 | 0.794 |

The frequency token improves every metric:
- **accuracy:** +0.37 points;
- **AUC:** +0.002;
- **TPR at 1 % FPR:** +4.4 points, the largest gain, which matters at strict operating points.

The accuracy CIs overlap slightly, so the accuracy gain is suggestive. The low-FPR gain is the more substantive one. Seed variance for SFHybrid itself was not measured.

**The frequency evidence does not survive resizing or blurring (§5.3).** Spectral traces are exactly what those operations destroy.

### 5.2 Effect of the duplicate leakage (verified)

On the de-duplicated test split (378 duplicated FAKE and 2 REAL images removed), every model's accuracy changes by **≤ 0.1 points**:
- LightCNN: 0.9554 → 0.9552;
- weighted ensemble: 0.9673 → 0.9672.

The leakage is real but does **not** materially inflate these models' scores. It should still be removed in future CIFAKE work, because higher-capacity models that memorise more could benefit from it.

### 5.3 Robustness (verified; 2,000 test images; accuracy)

| Model | Clean | Mean over 24 perturbations | Worst case |
|---|---|---|---|
| EfficientNet-B0 | 0.962 | 0.813 | 0.523 |
| LightCNN | 0.956 | 0.812 | 0.513 |
| SFHybrid | 0.948 | 0.794 | 0.503 |
| TinyViT | 0.909 | 0.819 | 0.621 |
| Weighted ensemble | 0.968 | 0.822 | 0.514 |
| **LightCNN_robust** | 0.952 | **0.896** | **0.741** |
| **SFHybrid_robust** | 0.937 | **0.889** | **0.775** |

The full grid is in `results/robustness.csv` and `plots/robustness_accuracy.png`.

- **Harmless:** colour, brightness and hue changes, and JPEG at quality 75 or higher (all models stay at 0.88 or above).
- **Catastrophic for standard training:** **resizing, blurring and cropping.** Down-up resizing at 0.875×, which is barely visible, drops LightCNN from 0.956 to 0.594 and SFHybrid to 0.550. Blur with σ = 1.0 drops most models to about 0.52–0.58, near chance.
  - These operations remove high-frequency content, so the detectors depend heavily on fine-scale traces.
  - SFHybrid, which explicitly uses spectral evidence, is hit hardest.
- **A model-specific weakness:** EfficientNet-B0 collapses under Gaussian noise with σ = 0.05 (0.608), while the from-scratch models stay at about 0.88–0.92.
- **Robust training largely fixes the problem** at a small clean cost:
  - LightCNN_robust: mean 0.812 → 0.896, worst case 0.513 → 0.741, for −0.4 points clean;
  - SFHybrid_robust: worst case 0.503 → 0.775;
  - under resize at 0.875×: 0.594 → 0.896 for LightCNN and 0.550 → 0.886 for SFHybrid.
- **The ensemble inherits the weakness of its members:** its worst case is 0.514. Ensembling improves clean accuracy, not robustness. See §5.4 for an ensemble built from robust members.

### 5.4 Robust ensemble (verified)

This ensemble swaps LightCNN and SFHybrid for their robust-trained variants and keeps EfficientNet-B0 and TinyViT. It was fitted and evaluated with the identical protocol. Outputs are in `runs/cpu_experiment_robust_ensemble/`.

| Weighted ensemble | Clean test accuracy [95 % CI] | ROC-AUC | Mean accuracy over perturbations | Worst case | DDPM ROC-AUC |
|---|---|---|---|---|---|
| Standard members | 0.9673 [0.9648–0.9698] | 0.9950 | 0.822 | 0.514 | 0.438 |
| **Robust members** | **0.9671 [0.9644–0.9696]** | 0.9948 | **0.874** | **0.639** | 0.484 |

- **Same clean accuracy, much better robustness.** The two ensembles are indistinguishable on clean data, but the robust one gains +5.2 points of mean robustness and +12.5 points in the worst case.
- **Fitted weights:**

  | Member | Weight |
  |---|---|
  | EfficientNet-B0 | 0.45 |
  | LightCNN_robust | 0.32 |
  | SFHybrid_robust | 0.22 |
  | TinyViT | 0.00 |

  TinyViT gets no weight once stronger members are available.
- **Remaining weakness:** the worst case is still limited by the non-robust EfficientNet-B0. A robust-trained EfficientNet is the obvious next step (*proposed*).
- **Deployment choice:** uploaded images are always resized, which is the failure mode in §5.3. **The dashboard therefore deploys this robust ensemble by default** (`CIFAKE_RUN_DIR=runs/cpu_experiment_robust_ensemble`).

### 5.5 Generalisation to an unseen generator (verified; 500 DDPM fakes and 500 real images)

| Model | Accuracy | ROC-AUC | Recall (fakes caught) | Specificity |
|---|---|---|---|---|
| EfficientNet-B0 | 0.490 | 0.428 | 0.018 | 0.962 |
| LightCNN | 0.476 | 0.481 | 0.010 | 0.942 |
| SFHybrid | 0.474 | 0.407 | 0.014 | 0.934 |
| TinyViT | 0.483 | 0.455 | 0.048 | 0.918 |
| LightCNN_robust | 0.488 | 0.564 | 0.038 | 0.938 |
| SFHybrid_robust | 0.500 | 0.544 | 0.066 | 0.934 |
| Weighted ensemble | 0.487 | 0.438 | 0.008 | 0.966 |

**This is the most important limitation we found. Every detector fails completely on an unseen generator.**
- **Fakes are not caught:** fewer than 7 % of DDPM fakes are flagged.
- **Ranking is below chance:** ROC-AUC falls *below* 0.5 for the standard models, so DDPM fakes look "more real than the real images" to them.
- **Robust training helps only slightly:** AUC 0.54–0.56, but recall is still under 7 %.

**Interpretation (a hypothesis, not tested here):** the models learned *Stable-Diffusion-specific* cues, most likely the VAE decoder's high-frequency signature and the down-scaling from 512 px. They did not learn a generic notion of "generated". A pixel-space DDPM trained natively at 32×32 on CIFAR-10 has neither cue. This matches the cross-generator literature (Wang et al., CVPR 2020; Ojha et al., CVPR 2023). Any claim of "AI-image detection" beyond Stable Diffusion v1.4 is unsupported by these models.

### 5.6 Spectral fingerprint (verified)

`plots/residual_spectra.png` shows the average residual spectrum of 10k FAKE and 10k REAL training images.
- **The pattern:** FAKE images carry more energy along the vertical-frequency axis and less along the horizontal one, an anisotropic generator/resampling trace.
- **Strength as a cue:** projecting a test image's spectrum onto this FAKE − REAL direction classifies the first 4,000 test images with **69.0 %** accuracy, without any neural network. That is weak evidence on its own, but real.
- **In the dashboard:** shown as a transparent, model-independent cue.

### 5.7 Explanation faithfulness (verified; deletion AUC, lower = more faithful; 200 images per model)

| Model | Random (control) | Grad-CAM | Grad-CAM++ | Integrated Gradients | Attention rollout |
|---|---|---|---|---|---|
| LightCNN | 0.737 | 0.698 | 0.707 | **0.595** | – |
| EfficientNet-B0 | 0.793 | 0.702 | 0.764 | **0.633** | – |
| SFHybrid | 0.721 | 0.731 | 0.751 | **0.650** | 0.766 |
| TinyViT | 0.805 | – | – | **0.393** | 0.772 |

- **Integrated Gradients with a blurred baseline is the most faithful method for every model.** It is therefore the default pixel-level explanation shown to users.
- **Grad-CAM is only modestly better than random** for the CNNs. For SFHybrid it is *not* better than random, probably because part of that model's decision comes from the frequency token, which has no spatial location.
- **Grad-CAM++ was never better than Grad-CAM here.**
- **Attention rollout is no better than random.** For SFHybrid it is worse than random. The dashboard still shows it, but labels it as "where attention flowed", not as evidence.

These results warn against reading CAM or attention heatmaps as proof. The UI text and the downloadable report say so.

### 5.8 Complexity (CPU; batch 1; this laptop, measured while other jobs were running, so latencies are indicative)

| Model | Params | MMACs per image | Latency (ms) | Throughput (img/s, batch 64) |
|---|---|---|---|---|
| LightCNN | 0.58 M | 43 | 13 | 605 |
| SFHybrid | 0.72 M | 56 | 21 | 349 |
| TinyViT | 1.80 M | 125 | 21 | 277 |
| EfficientNet-B0 (64 px) | 4.01 M | 32* | 107 | 155 |

\* MACs for EfficientNet are counted by `torch.utils.flop_counter`. Its depthwise convolutions are cheap in MACs but slow on CPU, which explains its high latency despite the low MAC count.

LightCNN gives the best accuracy per unit of compute. The full four-model ensemble costs about 160 ms per image on this laptop.

---

## 6. Limitations

1. **Compute-limited training.** Models saw 20 % of the training split (20k images) for 5–15 epochs. EfficientNet used 64 px inputs instead of 128 px. Absolute numbers will be higher with `configs/full_gpu.yaml` (**not run**). The *relative* conclusions (ensemble gain, robustness collapse, unseen-generator failure) are what this study supports.
2. **A single seed for most models.** Seed variance was measured only for LightCNN (3 seeds).
3. **A single unseen generator** (DDPM, 500 images, DDIM with 25 steps). Fewer sampling steps can change image statistics. We did not test other generators (GANs, SDXL, Midjourney and so on).
4. **The test REAL images all come from CIFAR-10.** No other camera or photo sources were tested.
5. **32×32 only.** The app downsizes uploads to 32×32 and re-encodes them as JPEG quality 75 to match the training distribution. Real-world photos are far outside the training distribution, and §5.3 shows that resizing alone breaks the standard models. Predictions on everyday images should not be trusted. The UI says this explicitly.
6. **Saliency methods are only partially faithful** (§5.7).

## 7. Future work (proposed; not implemented)

- **Train on several generators** (GAN, diffusion and autoregressive) and evaluate leave-one-generator-out. This is the only reliable route to generalisation.
- **Use features from large pre-trained vision-language models** (for example, CLIP-based UnivFD by Ojha et al. 2023). These are known to transfer across generators better than artefact-based detectors.
- **Train on full CIFAKE at native resolution** with the GPU config, run 3+ seeds per model, and report mean ± std.
- **Detect on patches at native resolution** instead of global downscaling, for real-world images.
- **Use conformal prediction** for calibrated abstention ("I don't know") instead of a fixed uncertainty band.
- **Add adversarial-robustness evaluation** (PGD, compression-aware attacks).

## 8. Reproducing

```bash
python scripts/audit_data.py --config configs/cpu_experiment.yaml
python scripts/train.py      --config configs/cpu_experiment.yaml --threads 4          # several hours on a laptop CPU
python scripts/generate_unseen_fakes.py --n 500 --steps 25
python scripts/evaluate.py   --config configs/cpu_experiment.yaml --unseen data_unseen/ddpm_cifar10
python scripts/seed_study.py --config configs/cpu_experiment.yaml --model LightCNN \
       --runs runs/cpu_experiment runs/seed_study/seed43 runs/seed_study/seed44       # after training seeds 43/44
```

Seeds, splits (`results/splits.json`), configs and checkpoints are saved with every run.
- **Determinism:** CPU runs are deterministic for a given thread count. GPU runs can differ slightly unless `deterministic: true` is set.
- **The 500-image cap:** in our run, the DDPM generator was stopped at 500 images after being started with `--n 1000`. Because the sampler RNG is sequential, `--n 500` reproduces the same 500 images.

## References

Durall et al., *Watch your Up-Convolution*, CVPR 2020 · Frank et al., *Leveraging Frequency Analysis for Deep Fake Image
Recognition*, ICML 2020 · Wang et al., *CNN-generated images are surprisingly easy to spot… for now*, CVPR 2020 ·
Ojha et al., *Towards Universal Fake Image Detectors*, CVPR 2023 · Bird & Lotfi, *CIFAKE*, IEEE Access 2024 ·
Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017 · Selvaraju et al., *Grad-CAM*, ICCV 2017 ·
Chattopadhay et al., *Grad-CAM++*, WACV 2018 · Sundararajan et al., *Axiomatic Attribution (IG)*, ICML 2017 ·
Abnar & Zuidema, *Quantifying Attention Flow*, ACL 2020 · Petsiuk et al., *RISE*, BMVC 2018 · Hu et al.,
*Squeeze-and-Excitation Networks*, CVPR 2018 · Ho et al., *DDPM*, NeurIPS 2020 · Song et al., *DDIM*, ICLR 2021 ·
Tan & Le, *EfficientNet*, ICML 2019 · Dosovitskiy et al., *ViT*, ICLR 2021.
