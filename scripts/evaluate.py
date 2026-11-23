"""Unified evaluation of every trained model and of the ensembles (one protocol for all).

Steps (each writes into <run>/results and <run>/plots):
  1. logits on calib + test for every checkpoint; temperature scaling fitted on calib
  2. test metrics (raw + calibrated), bootstrap 95% CIs, metrics on the de-duplicated test split
  3. ensembles fitted on calib (5 strategies) and evaluated on test; McNemar vs. best single model
  4. complexity: parameters, size, FLOPs, CPU latency (batch 1) and throughput
  5. robustness: 9 perturbation families x severities on a fixed test subset
  6. unseen generator benchmark (folder with REAL/ and FAKE/), if --unseen is given
  7. explanation faithfulness (deletion test) for Grad-CAM / Grad-CAM++ / IG / rollout / random
  8. figures + results/summary.json + results/RESULTS.md

  python scripts/evaluate.py --config configs/cpu_experiment.yaml --unseen data_unseen/ddpm_cifar10
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve  # noqa: E402

from cifake_guard import explain  # noqa: E402
from cifake_guard.config import load_config  # noqa: E402
from cifake_guard.data import load_image_folder  # noqa: E402
from cifake_guard.ensemble import ENSEMBLE_METHODS, Ensemble, fit_temperature, sigmoid  # noqa: E402
from cifake_guard.metrics import bootstrap_ci, compute_metrics, mcnemar_test, reliability_curve  # noqa: E402
from cifake_guard.pipeline import load_checkpoint, predict_logits, prepare_data  # noqa: E402
from cifake_guard.transforms import PERTURBATIONS, apply_perturbation  # noqa: E402
from cifake_guard.utils import (add_file_log, count_params, get_logger, load_json, model_size_mb,  # noqa: E402
                                save_json, seed_everything)

log = get_logger("evaluate")
KEY_METRICS = ["accuracy", "precision", "recall_TPR", "specificity_TNR", "f1", "roc_auc", "pr_auc", "eer",
               "tpr@1%fpr", "log_loss", "brier", "ece"]


# ------------------------------------------------------------------------------------ complexity
def flops_per_image(model) -> float:
    from torch.utils.flop_counter import FlopCounterMode
    x = torch.rand(1, 3, 32, 32)
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model(x)
    return fc.get_total_flops() / 2      # report multiply-accumulates (MACs)


@torch.no_grad()
def latency(model, n_runs: int = 30, batch: int = 64) -> tuple[float, float]:
    def timed(bs, runs):
        x = torch.rand(bs, 3, 32, 32)
        for _ in range(3):
            model(x)
        t0 = time.perf_counter()
        for _ in range(runs):
            model(x)
        return (time.perf_counter() - t0) / runs
    return timed(1, n_runs) * 1000, batch / timed(batch, max(3, n_runs // 5))


# ------------------------------------------------------------------------------------------ plots
def plot_roc_pr(curves: dict, metrics: pd.DataFrame, path: Path, title_suffix=""):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for name, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        ls = "--" if name.startswith("Ensemble") else "-"
        a1.plot(fpr, tpr, ls, label=f"{name} (AUC={metrics.loc[name, 'roc_auc']:.4f})")
        a2.plot(rec, prec, ls, label=f"{name} (AP={metrics.loc[name, 'pr_auc']:.4f})")
    a1.plot([0, 1], [0, 1], "k:", lw=0.8)
    a1.set(xlabel="False positive rate", ylabel="True positive rate", title=f"ROC{title_suffix}")
    a2.set(xlabel="Recall", ylabel="Precision", title=f"Precision-recall{title_suffix}")
    for a in (a1, a2):
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusions(preds: dict, path: Path):
    n = len(preds)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.6))
    for ax, (name, (y, p)) in zip(np.atleast_1d(axes), preds.items()):
        cm = confusion_matrix(y, (p >= 0.5).astype(int), labels=[0, 1])
        cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n({cmn[i, j]:.1%})", ha="center", va="center",
                        color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
        ax.set_xticks([0, 1], ["REAL", "FAKE"])
        ax.set_yticks([0, 1], ["REAL", "FAKE"])
        ax.set(xlabel="Predicted", ylabel="Actual", title=name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_reliability(raw: dict, cal: dict, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, title in [(axes[0], raw, "Before calibration"), (axes[1], cal, "After temperature scaling")]:
        ax.plot([0, 1], [0, 1], "k:", lw=0.8, label="perfect calibration")
        for name, (y, p) in data.items():
            c, f, _ = reliability_curve(y, p, 10)
            ax.plot(c, f, marker="o", ms=3, label=name)
        ax.set(xlabel="Predicted P(FAKE)", ylabel="Observed fraction FAKE", title=f"Reliability: {title} (test)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_robustness(df: pd.DataFrame, path: Path, metric: str = "accuracy"):
    kinds = list(PERTURBATIONS)
    cols = 3
    rows = int(np.ceil(len(kinds) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.6 * rows))
    clean = df[df.perturbation == "clean"].set_index("model")[metric]
    for ax, kind in zip(axes.ravel(), kinds):
        sub = df[df.perturbation == kind]
        for model, g in sub.groupby("model", sort=False):
            xs = ["clean"] + [str(s) for s in g.severity]
            ys = [clean[model]] + list(g[metric])
            ax.plot(xs, ys, marker="o", ms=3, label=model, ls="--" if model.startswith("Ensemble") else "-")
        ax.set_title(kind)
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(kinds):]:
        ax.axis("off")
    axes.ravel()[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_spectra(ref: dict, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (title, img, cmap) in zip(axes, [("Mean REAL residual spectrum", ref["real"], "magma"),
                                             ("Mean FAKE residual spectrum", ref["fake"], "magma"),
                                             ("FAKE - REAL", ref["fake"] - ref["real"], "RdBu_r")]):
        if cmap == "RdBu_r":
            v = np.abs(img).max()
            im = ax.imshow(img, cmap=cmap, vmin=-v, vmax=v)
        else:
            im = ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_tradeoff(df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, r in df.iterrows():
        ax.scatter(r["latency_ms_bs1"], r["roc_auc"], s=30 + 40 * r["params_M"])
        ax.annotate(f"{name}\n{r['params_M']:.2f}M", (r["latency_ms_bs1"], r["roc_auc"]), fontsize=8,
                    xytext=(5, 5), textcoords="offset points")
    ax.set(xlabel="CPU latency, batch 1 (ms)", ylabel="Test ROC-AUC", title="Accuracy vs. cost (marker ~ params)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------------------- helpers
def table_md(df: pd.DataFrame, cols: list[str], floatfmt: str = "{:.4f}") -> str:
    head = "| model | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n"
    rows = []
    for name, r in df.iterrows():
        vals = [floatfmt.format(r[c]) if isinstance(r[c], (float, np.floating)) else str(r[c]) for c in cols]
        rows.append(f"| {name} | " + " | ".join(vals) + " |")
    return head + "\n".join(rows) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--models", nargs="*", help="checkpoints to evaluate (default: all in the run)")
    ap.add_argument("--unseen", nargs="*", default=[], help="REAL/FAKE folders of unseen generators")
    ap.add_argument("--skip", nargs="*", default=[], choices=["robustness", "faithfulness", "latency"])
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--faithfulness-n", type=int, default=200)
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    if args.threads:
        torch.set_num_threads(args.threads)
    run = Path(cfg["paths"]["output_dir"])
    res_dir, plot_dir = run / "results", run / "plots"
    res_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    add_file_log(run / "logs" / "evaluate.log")
    seed_everything(cfg["seed"])
    device = torch.device("cpu")
    data = prepare_data(cfg, run)
    y_cal, y_test = data["calib"].labels, data["test"].labels
    bs = cfg["data"]["eval_batch_size"]

    names = args.models or sorted(p.stem for p in (run / "checkpoints").glob("*.pt"))
    if not names:
        raise SystemExit(f"No checkpoints in {run / 'checkpoints'}; train first")
    models, ckpts = {}, {}
    for n in names:
        models[n], ckpts[n] = load_checkpoint(run / "checkpoints" / f"{n}.pt", device)

    # ---------------------------------------------------------------- 1. logits + calibration
    log.info("Computing logits for %s", names)
    L_cal = {n: predict_logits(m, data["calib"].images, device, bs) for n, m in models.items()}
    L_test = {n: predict_logits(m, data["test"].images, device, bs) for n, m in models.items()}
    temps = {n: fit_temperature(L_cal[n], y_cal) for n in names}
    log.info("Temperatures: %s", {k: round(v, 3) for k, v in temps.items()})

    # near-duplicate mask for the de-duplicated test evaluation
    dup_file = Path(cfg["paths"]["cache_dir"]) / "test_near_duplicates.json"
    split_idx = np.asarray(load_json(res_dir / "splits.json")["indices"]["test"])
    dedup_mask = np.ones(len(y_test), dtype=bool)
    if dup_file.exists():
        dup = set(load_json(dup_file)["test_indices"])
        dedup_mask = np.array([i not in dup for i in split_idx])
        log.info("De-duplicated test: removing %d near-duplicate images", int((~dedup_mask).sum()))

    # ---------------------------------------------------------------- 2. single-model metrics
    rows, rows_raw, rows_dedup, cis = {}, {}, {}, {}
    probs_test = {}
    for n in names:
        p_raw = sigmoid(L_test[n])
        p_cal = sigmoid(L_test[n] / temps[n])
        probs_test[n] = p_cal
        rows[n] = compute_metrics(y_test, p_cal)
        rows_raw[n] = compute_metrics(y_test, p_raw)
        rows_dedup[n] = compute_metrics(y_test[dedup_mask], p_cal[dedup_mask])
        cis[n] = bootstrap_ci(y_test, p_cal, cfg["evaluation"]["bootstrap"], cfg["seed"])
        log.info("%s: acc %.4f auc %.4f (raw ECE %.4f -> calibrated %.4f)", n, rows[n]["accuracy"],
                 rows[n]["roc_auc"], rows_raw[n]["ece"], rows[n]["ece"])

    # ---------------------------------------------------------------- 3. ensembles
    members = [m for m in (cfg["evaluation"].get("ensemble_members") or names) if m in names]
    ens = None
    if len(members) >= 2:
        Zc = np.stack([L_cal[m] for m in members], 1)
        Zt = np.stack([L_test[m] for m in members], 1)
        ens = Ensemble.fit(Zc, y_cal, members, method="weighted")
        for method in ENSEMBLE_METHODS:
            p = ens.predict_proba(Zt, method)
            key = f"Ensemble[{method}]"
            probs_test[key] = p
            rows[key] = compute_metrics(y_test, p)
            rows_dedup[key] = compute_metrics(y_test[dedup_mask], p[dedup_mask])
            cis[key] = bootstrap_ci(y_test, p, cfg["evaluation"]["bootstrap"], cfg["seed"])
        save_json(ens.to_dict(), res_dir / "ensemble.json")
        log.info("Ensemble weights: %s", dict(zip(members, np.round(ens.weights, 3))))
    else:
        log.warning("Fewer than 2 ensemble members available (%s); skipping ensembles", members)

    metrics_df = pd.DataFrame(rows).T
    raw_df = pd.DataFrame(rows_raw).T
    dedup_df = pd.DataFrame(rows_dedup).T

    # McNemar: best single model vs weighted ensemble
    stats = {}
    best_single = max(names, key=lambda n: rows[n]["accuracy"])
    if ens is not None:
        stats["mcnemar_weighted_ensemble_vs_" + best_single] = mcnemar_test(
            y_test, (probs_test["Ensemble[weighted]"] >= 0.5).astype(int), (probs_test[best_single] >= 0.5).astype(int))

    # ---------------------------------------------------------------- 4. complexity
    if "latency" not in args.skip:
        for n, m in models.items():
            lat, thr = latency(m)
            metrics_df.loc[n, "latency_ms_bs1"], metrics_df.loc[n, "throughput_img_s"] = lat, thr
            metrics_df.loc[n, "params_M"] = count_params(m) / 1e6
            metrics_df.loc[n, "size_MB"] = model_size_mb(m)
            metrics_df.loc[n, "MMACs"] = flops_per_image(m) / 1e6
    metrics_df.to_csv(res_dir / "test_metrics.csv")
    raw_df.to_csv(res_dir / "test_metrics_uncalibrated.csv")
    dedup_df.to_csv(res_dir / "test_metrics_dedup.csv")
    save_json(cis, res_dir / "bootstrap_ci.json")

    # plots
    curves = {k: (y_test, v) for k, v in probs_test.items()
              if k in names or k in ("Ensemble[weighted]", "Ensemble[stacking]")}
    plot_roc_pr(curves, metrics_df, plot_dir / "roc_pr_curves.png", " (official test split)")
    plot_confusions({k: v for k, v in curves.items()}, plot_dir / "confusion_matrices.png")
    plot_reliability({n: (y_test, sigmoid(L_test[n])) for n in names},
                     {n: (y_test, probs_test[n]) for n in names}, plot_dir / "calibration_curves.png")
    if "latency" not in args.skip:
        plot_tradeoff(metrics_df.loc[names], plot_dir / "accuracy_vs_cost.png")

    # spectral reference (train split -> used by the dashboard's frequency explanation)
    ref = explain.mean_residual_spectra(data["train"].images, data["train"].labels)
    np.savez(res_dir / "spectral_reference.npz", **ref)
    plot_spectra(ref, plot_dir / "residual_spectra.png")
    spec_scores = np.array([explain.spectral_evidence(explain.residual_spectrum(
        torch.from_numpy(im).permute(2, 0, 1)[None].float() / 255), ref)["score"] for im in data["test"].images[:4000]])
    stats["spectral_cue_test_accuracy_first4000"] = float(((spec_scores > 0).astype(int) ==
                                                           data["test"].labels[:4000]).mean())

    # ---------------------------------------------------------------- 5. robustness
    rob_rows = []
    if "robustness" not in args.skip:
        k = cfg["evaluation"]["robustness_per_class"]
        rng = np.random.default_rng(cfg["seed"])
        idx = np.concatenate([rng.choice(np.where(y_test == c)[0], k, replace=False) for c in (0, 1)])
        sub_imgs, sub_y = data["test"].images[idx], y_test[idx]
        settings = [("clean", None)] + [(kind, s) for kind, (_, sev) in PERTURBATIONS.items() for s in sev]
        for kind, sev in settings:
            tf = None if kind == "clean" else (lambda x, kind=kind, sev=sev: apply_perturbation(x, kind, sev, cfg["seed"]))
            Ls = {n: predict_logits(m, sub_imgs, device, bs, transform=tf) for n, m in models.items()}
            for n in names:
                mt = compute_metrics(sub_y, sigmoid(Ls[n] / temps[n]))
                rob_rows.append({"model": n, "perturbation": kind, "severity": sev, "accuracy": mt["accuracy"],
                                 "roc_auc": mt["roc_auc"], "recall_TPR": mt["recall_TPR"],
                                 "specificity_TNR": mt["specificity_TNR"]})
            if ens is not None:
                mt = compute_metrics(sub_y, ens.predict_proba(np.stack([Ls[m] for m in members], 1), "weighted"))
                rob_rows.append({"model": "Ensemble[weighted]", "perturbation": kind, "severity": sev,
                                 "accuracy": mt["accuracy"], "roc_auc": mt["roc_auc"],
                                 "recall_TPR": mt["recall_TPR"], "specificity_TNR": mt["specificity_TNR"]})
            log.info("robustness %s %s done", kind, sev)
        rob = pd.DataFrame(rob_rows)
        rob.to_csv(res_dir / "robustness.csv", index=False)
        plot_robustness(rob, plot_dir / "robustness_accuracy.png", "accuracy")
        plot_robustness(rob, plot_dir / "robustness_auc.png", "roc_auc")

    # ---------------------------------------------------------------- 6. unseen generators
    unseen_tables = {}
    for folder in args.unseen:
        folder = Path(folder)
        if not folder.exists():
            log.warning("Unseen folder %s not found; skipping", folder)
            continue
        ud = load_image_folder(folder)
        Lu = {n: predict_logits(m, ud.images, device, bs) for n, m in models.items()}
        urows = {n: compute_metrics(ud.labels, sigmoid(Lu[n] / temps[n])) for n in names}
        if ens is not None:
            urows["Ensemble[weighted]"] = compute_metrics(
                ud.labels, ens.predict_proba(np.stack([Lu[m] for m in members], 1), "weighted"))
        udf = pd.DataFrame(urows).T
        udf.to_csv(res_dir / f"unseen_{folder.name}.csv")
        unseen_tables[folder.name] = udf
        log.info("Unseen %s: %s", folder.name, udf[["accuracy", "roc_auc", "recall_TPR", "specificity_TNR"]].round(4).to_dict())

    # ---------------------------------------------------------------- 7. explanation faithfulness
    faith = None
    if "faithfulness" not in args.skip:
        rng = np.random.default_rng(cfg["seed"] + 1)
        idx = rng.choice(len(y_test), args.faithfulness_n, replace=False)
        frows = []
        for n, m in models.items():
            for i in idx:
                x = torch.from_numpy(data["test"].images[i]).permute(2, 0, 1)[None].float() / 255
                maps = {"random": rng.random((32, 32))}
                if getattr(m, "cam_layer", None):
                    maps["gradcam"] = explain.grad_cam(m, x)
                    maps["gradcam_pp"] = explain.grad_cam(m, x, plus_plus=True)
                ig, _ = explain.integrated_gradients(m, x, target="pred", steps=16)
                maps["integrated_gradients"] = np.clip(ig, 0, None)      # evidence FOR the predicted class
                if any(isinstance(mod, explain.Attention) for mod in m.modules()):
                    maps["attention_rollout"] = explain.attention_rollout(m, x)[0]
                for method, sal in maps.items():
                    frows.append({"model": n, "method": method, "deletion_auc": explain.deletion_score(m, x, sal)})
        faith = pd.DataFrame(frows).groupby(["model", "method"]).deletion_auc.agg(["mean", "std", "count"])
        faith.to_csv(res_dir / "explanation_faithfulness.csv")
        log.info("Faithfulness (deletion AUC, lower = better):\n%s", faith.round(4))

    # ---------------------------------------------------------------- 8. summary
    summary = {"run": str(run), "test_size": int(len(y_test)), "dedup_test_size": int(dedup_mask.sum()),
               "temperatures": temps, "stats": stats, "best_single_model": best_single,
               "ensemble": ens.to_dict() if ens else None,
               "metrics": metrics_df.to_dict(orient="index"), "bootstrap_ci": cis}
    save_json(summary, res_dir / "summary.json")

    md = [f"# Results for `{run}`\n", "Auto-generated by `scripts/evaluate.py`. Positive class = FAKE. "
          f"Test images: {len(y_test)} (official CIFAKE test split"
          f"{'' if cfg['data']['test_per_class'] is None else ' subset'}). Probabilities are temperature-scaled "
          "on the calibration split; threshold 0.5.\n",
          "## Test metrics (calibrated)\n", table_md(metrics_df, [c for c in KEY_METRICS if c in metrics_df]),
          "\n## 95% bootstrap confidence intervals\n",
          "| model | accuracy | F1 | ROC-AUC |\n|---|---|---|---|\n" + "\n".join(
              f"| {k} | {v['accuracy'][0]:.4f}-{v['accuracy'][1]:.4f} | {v['f1'][0]:.4f}-{v['f1'][1]:.4f} | "
              f"{v['roc_auc'][0]:.4f}-{v['roc_auc'][1]:.4f} |" for k, v in cis.items()) + "\n",
          "\n## Calibration: ECE / log-loss before vs after temperature scaling\n",
          "| model | T | ECE raw | ECE calibrated | log-loss raw | log-loss calibrated |\n|---|---|---|---|---|---|\n"
          + "\n".join(f"| {n} | {temps[n]:.3f} | {raw_df.loc[n, 'ece']:.4f} | {metrics_df.loc[n, 'ece']:.4f} | "
                      f"{raw_df.loc[n, 'log_loss']:.4f} | {metrics_df.loc[n, 'log_loss']:.4f} |" for n in names) + "\n",
          f"\n## De-duplicated test split ({int(dedup_mask.sum())} images; near-duplicates of training images removed)\n",
          table_md(dedup_df, ["accuracy", "f1", "roc_auc", "recall_TPR", "specificity_TNR"])]
    if "latency" not in args.skip:
        md += ["\n## Complexity (CPU, this machine)\n",
               table_md(metrics_df.loc[names], ["params_M", "size_MB", "MMACs", "latency_ms_bs1", "throughput_img_s"],
                        "{:.2f}")]
    if rob_rows:
        rob = pd.DataFrame(rob_rows)
        pivot = rob.pivot_table(index="model", columns=["perturbation", "severity"], values="accuracy", sort=False)
        worst = rob[rob.perturbation != "clean"].groupby("model", sort=False).accuracy.agg(["mean", "min"])
        clean = rob[rob.perturbation == "clean"].set_index("model").accuracy
        worst.insert(0, "clean", clean)
        worst.columns = ["clean acc", "mean acc over perturbations", "worst-case acc"]
        md += [f"\n## Robustness (accuracy, {2 * cfg['evaluation']['robustness_per_class']} test images)\n",
               table_md(worst, list(worst.columns)), "\nFull grid: `results/robustness.csv`.\n"]
        pivot.to_csv(res_dir / "robustness_accuracy_grid.csv")
    for nm, udf in unseen_tables.items():
        md += [f"\n## Unseen generator: {nm}\n", table_md(udf, ["accuracy", "roc_auc", "recall_TPR", "specificity_TNR",
                                                                 "ece"])]
    if faith is not None:
        fm = faith["mean"].unstack()
        md += ["\n## Explanation faithfulness (deletion AUC, lower = more faithful)\n", table_md(fm, list(fm.columns))]
    md += ["\n## Other statistics\n", "```json\n" + pd.Series(stats).to_json(indent=2) + "\n```\n"]
    (res_dir / "RESULTS.md").write_text("".join(md), encoding="utf-8")
    log.info("Wrote %s", res_dir / "RESULTS.md")


if __name__ == "__main__":
    main()
