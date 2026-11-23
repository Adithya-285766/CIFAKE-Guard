// CIFAKE-Guard dashboard (vanilla JS, no build step). All server strings are inserted with textContent.
"use strict";
const $ = (s) => document.querySelector(s);
const el = (tag, props = {}, ...kids) => {
  const n = Object.assign(document.createElement(tag), props);
  kids.flat().forEach((k) => n.append(k instanceof Node ? k : document.createTextNode(String(k))));
  return n;
};
const pct = (p) => `${(100 * p).toFixed(1)}%`;
const COLORS = { FAKE: "var(--fake)", REAL: "var(--real)", UNCERTAIN: "var(--unsure)" };
// Order reflects measured faithfulness (docs/RESEARCH.md §5.7): IG was the most faithful method for every model.
const EXPL = {
  integrated_gradients: ["Integrated Gradients ★", "Pixel evidence: red pushes toward AI-generated, blue toward real. The most faithful method in our tests."],
  gradcam: ["Grad-CAM", "Bright = regions whose features most influenced the decision (coarse; only modestly faithful)."],
  gradcam_pp: ["Grad-CAM++", "Refined Grad-CAM; was not more faithful than Grad-CAM in our tests."],
  attention_rollout: ["Attention rollout", "Where the transformer's attention flowed. Not evidence: it was no more faithful than random in our tests."],
};
let selectedFile = null, lastResult = null, modelInfo = {};

// ------------------------------------------------------------------ tabs
function openTab(name) {
  const b = document.querySelector(`.tab[data-tab="${name}"]`) || document.querySelector(".tab");
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === b));
  document.querySelectorAll(".tabpane").forEach((p) => p.classList.toggle("active", p.id === `tab-${b.dataset.tab}`));
  if (b.dataset.tab === "performance") loadPerformance();
}
document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => { history.replaceState(null, "", `#${b.dataset.tab}`); openTab(b.dataset.tab); }));
window.addEventListener("hashchange", () => openTab(location.hash.slice(1)));

// ------------------------------------------------------------------ health
async function health() {
  const s = $("#status");
  try {
    const r = await (await fetch("/api/health")).json();
    if (r.status === "ok") {
      s.textContent = `${r.models_loaded.length} models ready`; s.className = "pill ok";
      const m = await (await fetch("/api/models")).json();
      m.models.forEach((x) => (modelInfo[x.name] = x));
    } else { s.textContent = "models not loaded"; s.className = "pill bad"; s.title = r.error || ""; }
  } catch { s.textContent = "server offline"; s.className = "pill bad"; }
}
health();

// ------------------------------------------------------------------ upload
const drop = $("#drop"), input = $("#file");
const MAX_MB = 10;
function showError(msg) { const e = $("#error"); e.textContent = msg; e.hidden = !msg; }
function pick(file) {
  showError("");
  if (!file) return;
  if (!file.type.startsWith("image/")) return showError("Please choose an image file (JPEG, PNG, WebP, BMP or GIF).");
  if (file.size > MAX_MB * 1024 * 1024) return showError(`File is larger than ${MAX_MB} MB.`);
  selectedFile = file;
  const img = $("#preview");
  img.src = URL.createObjectURL(file); img.hidden = false; $("#drop-empty").hidden = true;
  img.onload = () => {
    img.classList.toggle("tiny", img.naturalWidth < 128);       // crisp pixels for CIFAKE-sized images
    $("#file-meta").textContent = `${file.name} · ${img.naturalWidth}×${img.naturalHeight}px · ${(file.size / 1024).toFixed(0)} KB`;
  };
  $("#analyse").disabled = false;
}
input.addEventListener("change", () => pick(input.files[0]));
drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") input.click(); });
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => pick(e.dataTransfer.files[0]));
document.addEventListener("paste", (e) => { const f = [...(e.clipboardData?.files || [])][0]; if (f) pick(f); });

$("#analyse").addEventListener("click", async () => {
  if (!selectedFile) return;
  const btn = $("#analyse");
  btn.disabled = true; btn.classList.add("loading"); btn.textContent = "Analysing";
  showError("");
  try {
    const fd = new FormData(); fd.append("file", selectedFile);
    const res = await fetch(`/api/predict?explain=${$("#opt-explain").checked}`, { method: "POST", body: fd });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.detail || `Request failed (${res.status})`);
    lastResult = body; render(body);
  } catch (err) { showError(err.message); }
  finally { btn.disabled = false; btn.classList.remove("loading"); btn.textContent = "Analyse image"; }
});

// ------------------------------------------------------------------ rendering
function render(r) {
  $("#placeholder").hidden = true; $("#results").hidden = false; $("#input-card").hidden = false;
  // gauge
  const arc = $("#gauge-arc"), C = 2 * Math.PI * 52;
  arc.style.strokeDasharray = C; arc.style.strokeDashoffset = C * (1 - r.prob_fake);
  arc.style.stroke = COLORS[r.verdict];
  $("#gauge-val").textContent = pct(r.prob_fake);
  const v = $("#verdict"); v.textContent = r.verdict === "UNCERTAIN" ? "UNCERTAIN" : r.verdict === "FAKE" ? "AI-GENERATED" : "REAL";
  v.className = `verdict ${r.verdict}`;
  $("#band").textContent = `${r.confidence_band} · ensemble: ${r.ensemble_method} · ${r.latency_ms.toFixed(0)} ms`;
  const a = r.agreement, agree = $("#agree"); agree.replaceChildren(
    el("span", {}, a.unanimous ? "Unanimous: " : `Agreement ${pct(a.agreement)}: `),
    ...Object.entries(r.models).map(([n, m]) => el("span", { className: `vote ${m.verdict}`, title: n }, `${n}: ${m.verdict}`)));
  $("#dl-html").href = `/api/report/${r.report_id}?format=html`;
  $("#dl-json").href = `/api/report/${r.report_id}?format=json`;
  $("#summary").replaceChildren(...r.summary.map((s) => el("li", {}, s)));
  // model bars
  $("#model-bars").replaceChildren(...Object.entries(r.models).map(([n, m]) => {
    const fill = el("div", { className: "fill" });
    fill.style.width = pct(m.prob_calibrated); fill.style.background = COLORS[m.verdict];
    const desc = modelInfo[n]?.description || "";
    return el("div", { className: "bar-row", title: desc },
      el("div", { className: "name" }, n, el("small", {}, `weight ${m.weight.toFixed(2)} · raw ${pct(m.prob_raw)}`)),
      el("div", { className: "bar" }, fill, el("div", { className: "mid" })),
      el("div", { className: "val" }, pct(m.prob_calibrated)));
  }));
  $("#strategies").replaceChildren(el("tr", {}, el("th", {}, "strategy"), el("th", {}, "P(AI-generated)")),
    ...Object.entries(r.ensemble_strategies).map(([k, p]) => el("tr", {}, el("td", {}, k), el("td", {}, pct(p)))));
  // preprocessing
  $("#seen-img").src = `data:image/png;base64,${r.visuals.input}`;
  const pre = r.preprocessing, notes = [];
  notes.push(`Original: ${pre.original_size[0]}×${pre.original_size[1]} (${pre.original_format || "?"}, ${pre.original_mode})`);
  if (pre.resized) notes.push("Centre-cropped and shrunk to 32×32 px, then JPEG re-encoded at quality 75 to match the training data.");
  else notes.push("Already 32×32: analysed as-is.");
  $("#pre-notes").replaceChildren(...notes.map((t) => el("p", {}, t)));
  renderExplanations(r);
  // frequency
  const freq = [];
  freq.push(fig(r.visuals.spectrum, "This image's residual spectrum", "Fourier magnitude of the fine detail (centre = low frequencies, edges = high)."));
  if (r.visuals.spectrum_reference_diff)
    freq.push(fig(r.visuals.spectrum_reference_diff, "AI minus real (training average)", "Red frequencies are stronger in AI-generated training images, blue in real ones."));
  if (r.spectral_evidence) freq.push(el("div", { className: "small" }, el("p", {}, el("b", {}, "Spectral cue: "),
    `this image's spectrum leans ${r.spectral_evidence.leans} (score ${r.spectral_evidence.score.toFixed(2)}).`),
    el("p", { className: "muted" }, "This simple cue is independent of the neural networks and much weaker than they are; it is shown for transparency.")));
  $("#freq").replaceChildren(...freq);
}

function fig(b64, title, caption) {
  return el("figure", {}, el("img", { src: `data:image/png;base64,${b64}`, alt: title }), el("figcaption", {}, el("b", {}, title), caption));
}

function renderExplanations(r) {
  const names = Object.keys(r.explanations || {});
  $("#explain-card").hidden = names.length === 0;
  if (!names.length) return;
  const chips = $("#model-chips");
  const show = (name) => {
    chips.querySelectorAll(".chip").forEach((c) => c.classList.toggle("active", c.dataset.m === name));
    const e = r.explanations[name], figs = [];
    if (e.error) figs.push(el("p", { className: "error" }, `Explanation failed: ${e.error}`));
    for (const [k, [title, cap]] of Object.entries(EXPL)) if (e[k]) figs.push(fig(e[k], title, cap));
    if (e.attention_frequency_token_share !== undefined)
      figs.push(el("div", { className: "small" }, el("p", {}, el("b", {}, "Frequency reliance: "),
        `${pct(e.attention_frequency_token_share)} of the final attention went to the frequency (FFT) token rather than to image regions.`)));
    $("#xgrid").replaceChildren(...figs);
  };
  chips.replaceChildren(...names.map((n) => { const c = el("button", { className: "chip" }, n); c.dataset.m = n; c.onclick = () => show(n); return c; }));
  // default: the most confident model
  const best = names.reduce((b, n) => Math.abs(r.models[n].prob_calibrated - 0.5) > Math.abs(r.models[b].prob_calibrated - 0.5) ? n : b, names[0]);
  show(best);
}

// ------------------------------------------------------------------ performance tab
let perfLoaded = false;
async function loadPerformance() {
  if (perfLoaded) return;
  try {
    const r = await (await fetch("/api/results")).json();
    perfLoaded = true;
    const cols = ["accuracy", "precision", "recall_TPR", "specificity_TNR", "f1", "roc_auc", "pr_auc", "eer", "ece", "params_M", "latency_ms_bs1"];
    table($("#perf-table"), r.test_metrics || {}, cols, ["accuracy", "f1", "roc_auc", "pr_auc"]);
    const u = $("#unseen"); u.replaceChildren();
    const unseen = Object.entries(r.unseen || {});
    if (!unseen.length) u.append(el("p", { className: "muted" }, "No unseen-generator evaluation found for this run."));
    unseen.forEach(([name, rows]) => { const t = el("table", { className: "tbl" }); table(t, rows, ["accuracy", "roc_auc", "recall_TPR", "specificity_TNR", "ece"], ["accuracy", "roc_auc"]); u.append(el("h3", { className: "small" }, name), t); });
    if (r.robustness) {
      const by = {};
      r.robustness.forEach((x) => { const k = x.perturbation === "clean" ? "clean" : `${x.perturbation} ${x.severity}`; (by[x.model] ||= {})[k] = x.accuracy; });
      const keys = Object.keys(Object.values(by)[0] || {});
      table($("#rob-table"), by, keys, []);
    }
    $("#plots").replaceChildren(...(r.plots || []).filter((p) => !/_(loss|accuracy)\.png$/.test(p) || p.startsWith("robustness"))
      .map((p) => { const src = `/api/plots/${encodeURIComponent(p)}`;
        return el("figure", {}, el("a", { href: src, target: "_blank", rel: "noopener", title: "open full size" },
          el("img", { src, alt: p, loading: "lazy" })), el("figcaption", {}, p)); }));
  } catch (e) { $("#perf-table").replaceChildren(el("tr", {}, el("td", {}, `Could not load results: ${e.message}`))); }
}

function table(t, rows, cols, higherBetter) {
  const names = Object.keys(rows);
  const best = {};
  higherBetter.forEach((c) => (best[c] = Math.max(...names.map((n) => rows[n][c] ?? -Infinity))));
  t.replaceChildren(el("tr", {}, el("th", {}, "model"), ...cols.map((c) => el("th", {}, c))),
    ...names.map((n) => el("tr", { className: n.startsWith("Ensemble") ? "ens" : "" }, el("td", {}, n),
      ...cols.map((c) => { const v = rows[n][c]; const td = el("td", {}, v == null ? "–" : typeof v === "number" ? v.toFixed(4) : v);
        if (higherBetter.includes(c) && v === best[c]) td.className = "best"; return td; }))));
}

// initial tab from the URL hash (at the end so every declaration above is initialised)
if (location.hash) openTab(location.hash.slice(1));
