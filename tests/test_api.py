import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


def _png(size=(64, 48), mode="RGB") -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)).convert(mode).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def client(tiny_run):
    import os
    os.environ["CIFAKE_RUN_DIR"] = str(tiny_run)
    os.environ["CIFAKE_MAX_UPLOAD_MB"] = "1"
    from app.main import app
    with TestClient(app) as c:
        yield c


def test_health_and_models(client):
    r = client.get("/api/health").json()
    assert r["status"] == "ok" and len(r["models_loaded"]) == 4
    m = client.get("/api/models").json()
    assert {x["name"] for x in m["models"]} == {"LightCNN", "EfficientNetB0", "TinyViT", "SFHybrid"}
    assert abs(sum(m["ensemble"]["weights"]) - 1) < 1e-6


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "CIFAKE-Guard" in r.text
    assert client.get("/static/app.js").status_code == 200


def test_predict_full(client):
    r = client.post("/api/predict", files={"file": ("x.png", _png(), "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["verdict"] in {"FAKE", "REAL", "UNCERTAIN"} and 0 <= body["prob_fake"] <= 1
    assert set(body["models"]) == {"LightCNN", "EfficientNetB0", "TinyViT", "SFHybrid"}
    assert body["agreement"]["n_models"] == 4 and body["summary"]
    assert "gradcam" in body["explanations"]["LightCNN"]
    assert "attention_rollout" in body["explanations"]["TinyViT"]
    assert "attention_frequency_token_share" in body["explanations"]["SFHybrid"]
    assert body["preprocessing"]["resized"]
    # downloadable reports
    html = client.get(f"/api/report/{body['report_id']}?format=html")
    assert html.status_code == 200 and "forensic report" in html.text and "attachment" in html.headers["content-disposition"]
    js = client.get(f"/api/report/{body['report_id']}?format=json").json()
    assert "explanations" not in js and js["verdict"] == body["verdict"]


def test_predict_without_explanations(client):
    r = client.post("/api/predict?explain=false", files={"file": ("x.png", _png((32, 32)), "image/png")})
    assert r.status_code == 200 and r.json()["explanations"] == {}


@pytest.mark.parametrize("payload,ctype,code", [
    (b"", "image/png", 400),
    (b"not an image at all", "image/png", 400),
    (b"%PDF-1.4", "application/pdf", 415),
])
def test_predict_rejects_bad_input(client, payload, ctype, code):
    r = client.post("/api/predict", files={"file": ("bad", payload, ctype)})
    assert r.status_code == code


def test_predict_rejects_oversized(client):
    r = client.post("/api/predict", files={"file": ("big.png", b"\x89PNG" + b"0" * (1024 * 1024 + 10), "image/png")})
    assert r.status_code == 413


def test_unknown_report_and_plot(client):
    assert client.get("/api/report/doesnotexist").status_code == 404
    assert client.get("/api/plots/..%2F..%2Fsecret.png").status_code == 404


def test_results_endpoint(client):
    r = client.get("/api/results")
    assert r.status_code == 200 and "plots" in r.json()
