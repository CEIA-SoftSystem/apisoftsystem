"""Testes do endpoint de health check e informações de modelos."""


def test_health_returns_200(client):
    r = client.get("/health")
    assert r.status_code == 200


def test_health_schema(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert isinstance(body["seg_model_loaded"], bool)
    assert isinstance(body["det_model_loaded"], bool)
    assert isinstance(body["device"], str)
    assert isinstance(body["version"], str)


def test_health_version_matches_settings(client):
    from app.core.config import settings
    assert client.get("/health").json()["version"] == settings.app_version


def test_root_redirects_to_docs(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (301, 302, 307, 308)
    assert r.headers["location"].endswith("/docs")


def test_models_info_endpoint(client):
    r = client.get("/api/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert "seg_model_loaded" in body
    assert "det_model_loaded" in body
    assert "device" in body


def test_models_info_when_loaded(client, loaded_models):
    body = client.get("/api/v1/models").json()
    assert body["seg_model_loaded"] is True
    assert body["det_model_loaded"] is True
    assert body["seg_image_size"] == 512
