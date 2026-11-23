"""Command-line prediction for one or more images.

  python scripts/predict.py photo.jpg other.png --run runs/cpu_experiment [--report-dir reports/] [--no-explain]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cifake_guard.inference import InvalidImageError, Predictor, load_image_bytes  # noqa: E402
from cifake_guard.report import render_html_report  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--run", default="runs/cpu_experiment_robust_ensemble")
    ap.add_argument("--report-dir", default=None, help="write one HTML report per image here")
    ap.add_argument("--no-explain", action="store_true")
    ap.add_argument("--json", action="store_true", help="print full JSON (without images)")
    args = ap.parse_args(argv)

    pred = Predictor(args.run)
    status = 0
    for path in args.images:
        try:
            result = pred.predict(load_image_bytes(Path(path).read_bytes()),
                                  None if (args.no_explain or not args.report_dir) else "all")
        except (OSError, InvalidImageError) as e:
            print(f"{path}: ERROR {e}", file=sys.stderr)
            status = 1
            continue
        result["filename"] = Path(path).name
        if args.json:
            print(json.dumps({k: v for k, v in result.items() if k not in ("explanations", "visuals")}, indent=2))
        else:
            per_model = ", ".join(f"{n} {m['prob_calibrated']:.0%}" for n, m in result["models"].items())
            print(f"{path}: {result['verdict']} (P(fake)={result['prob_fake']:.1%}, {result['confidence_band']}; "
                  f"agreement {result['agreement']['agreement']:.0%}) [{per_model}]")
        if args.report_dir:
            out = Path(args.report_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{Path(path).stem}_report.html").write_text(render_html_report(result, pred.model_info()),
                                                               encoding="utf-8")
    return status


if __name__ == "__main__":
    sys.exit(main())
