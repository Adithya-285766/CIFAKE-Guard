"""Self-contained HTML forensic report (all images embedded as base64; no external resources)."""
from __future__ import annotations

import datetime as dt
from html import escape

from . import __version__

EXPLANATION_LABELS = {
    "gradcam": ("Grad-CAM", "Regions whose features most increased the model's decision."),
    "gradcam_pp": ("Grad-CAM++", "Like Grad-CAM, but better at highlighting several separate regions."),
    "integrated_gradients": ("Integrated Gradients", "Pixel-level evidence: red pushes toward AI-generated, "
                             "blue toward real (relative to a blurred copy of the image)."),
    "attention_rollout": ("Attention rollout", "Where the transformer's attention flowed when deciding."),
}


def _img(b64: str, alt: str) -> str:
    return f'<img alt="{escape(alt)}" src="data:image/png;base64,{b64}">'


def render_html_report(result: dict, model_info: list[dict]) -> str:
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    verdict = result["verdict"]
    colour = {"FAKE": "#c2410c", "REAL": "#047857"}.get(verdict, "#a16207")
    rows = "".join(
        f"<tr><td>{escape(n)}</td><td>{m['prob_calibrated']:.1%}</td><td>{m['prob_raw']:.1%}</td>"
        f"<td>{m['verdict']}</td><td>{m['weight']:.2f}</td></tr>" for n, m in result["models"].items())
    strategies = "".join(f"<tr><td>{escape(k)}</td><td>{v:.1%}</td></tr>"
                         for k, v in result["ensemble_strategies"].items())
    expl = ""
    for model, maps in result.get("explanations", {}).items():
        cells = ""
        for key, (label, desc) in EXPLANATION_LABELS.items():
            if key in maps:
                cells += f"<figure>{_img(maps[key], label)}<figcaption><b>{label}</b><br>{desc}</figcaption></figure>"
        extra = ""
        if "attention_frequency_token_share" in maps:
            extra = (f"<p class='note'>Share of attention on the frequency token: "
                     f"{maps['attention_frequency_token_share']:.1%}</p>")
        expl += f"<h3>{escape(model)}</h3><div class='grid'>{cells}</div>{extra}"
    vis = result["visuals"]
    spectral = ""
    if "spectrum" in vis:
        spectral = f"<figure>{_img(vis['spectrum'], 'spectrum')}<figcaption><b>Residual spectrum of this image</b></figcaption></figure>"
    if "spectrum_reference_diff" in vis:
        spectral += (f"<figure>{_img(vis['spectrum_reference_diff'], 'reference')}<figcaption><b>Training data: "
                     "average FAKE minus REAL spectrum</b> (red = stronger in AI images)</figcaption></figure>")
    pre = result["preprocessing"]
    metrics_rows = "".join(
        f"<tr><td>{escape(m['name'])}</td><td>{m['params_M']:.2f} M</td>"
        f"<td>{m.get('test_metrics', {}).get('accuracy', float('nan')):.4f}</td>"
        f"<td>{m.get('test_metrics', {}).get('roc_auc', float('nan')):.4f}</td></tr>" for m in model_info)
    summary = "".join(f"<li>{escape(s)}</li>" for s in result["summary"])
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>CIFAKE-Guard report</title>
<style>
body{{font-family:system-ui,Segoe UI,Roboto,sans-serif;max-width:960px;margin:24px auto;padding:0 16px;color:#1f2937;background:#fff}}
h1{{margin-bottom:0}} .muted{{color:#6b7280}} table{{border-collapse:collapse;width:100%;margin:8px 0 16px}}
td,th{{border:1px solid #e5e7eb;padding:6px 8px;text-align:left;font-size:14px}} th{{background:#f9fafb}}
.verdict{{border-left:6px solid {colour};padding:12px 16px;background:#f9fafb;margin:16px 0}}
.verdict b{{color:{colour};font-size:22px}} .grid{{display:flex;flex-wrap:wrap;gap:12px}}
figure{{margin:0;width:200px}} figure img{{width:200px;height:200px;image-rendering:pixelated;border-radius:6px}}
figcaption{{font-size:12px;color:#4b5563}} .note{{font-size:13px;color:#4b5563}}
.top{{display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap}}
</style></head><body>
<h1>CIFAKE-Guard forensic report</h1>
<p class="muted">Generated {now} · CIFAKE-Guard v{__version__} · file: {escape(str(result.get('filename')))}</p>
<div class="top"><figure>{_img(vis['input'], 'analysed image')}<figcaption>Image as analysed (32x32)</figcaption></figure>
<div class="verdict" style="flex:1"><b>{verdict}</b> — {escape(result['confidence_band'])}<br>
P(AI-generated) = {result['prob_fake']:.1%} · ensemble: {escape(result['ensemble_method'])} ·
agreement {result['agreement']['agreement']:.0%} ({result['agreement']['n_fake_votes']} FAKE / {result['agreement']['n_real_votes']} REAL)</div></div>
<h2>Plain-language summary</h2><ul>{summary}</ul>
<h2>Individual models</h2><table><tr><th>Model</th><th>P(fake), calibrated</th><th>P(fake), raw</th><th>Vote</th><th>Ensemble weight</th></tr>{rows}</table>
<h2>Ensemble strategies</h2><table><tr><th>Strategy</th><th>P(fake)</th></tr>{strategies}</table>
<h2>Explanations</h2>{expl or '<p>Not computed.</p>'}
<h2>Frequency analysis</h2><div class="grid">{spectral}</div>
<h2>Pre-processing</h2><p class="note">Original size {pre['original_size']}, mode {escape(str(pre['original_mode']))};
resized: {pre['resized']}; cropped to square: {pre['cropped_to_square']}; JPEG re-encoded (q75, like CIFAKE): {pre['jpeg_reencoded']}.</p>
<h2>Models and held-out performance</h2><table><tr><th>Model</th><th>Params</th><th>Test accuracy</th><th>Test ROC-AUC</th></tr>{metrics_rows}</table>
<h2>Limitations</h2><p class="note">Models were trained on CIFAKE only (32x32 CIFAR-10 photographs vs. Stable Diffusion v1.4
images). Accuracy on other generators, resolutions and post-processing is substantially lower (see the project's
robustness and unseen-generator evaluation). This report is decision support, not proof.</p>
</body></html>"""
