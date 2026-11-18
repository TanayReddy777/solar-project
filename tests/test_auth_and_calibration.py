"""Calibration and HTTP Basic authorization regression tests."""
import json
from unittest.mock import patch

from bson import ObjectId
from fastapi.testclient import TestClient

import app
from auth import hash_password

client = TestClient(app.app, base_url="https://testserver")
http_client = TestClient(app.app, base_url="http://testserver")


class StubModel:
    def __init__(self, prediction=1000.0):
        self.prediction, self.called = prediction, False

    def predict(self, frame):
        self.called = True
        return [self.prediction]


class Cursor:
    def __init__(self, docs): self.docs = list(docs)
    def sort(self, *_): return self
    def skip(self, count): self.docs = self.docs[count:]; return self
    def limit(self, count): self.docs = self.docs[:count]; return self
    def __iter__(self): return iter(self.docs)


class Collection:
    def __init__(self): self.docs = {}
    def find(self, query): return Cursor(d for d in self.docs.values() if all(d.get(k) == v for k, v in query.items()))
    def count_documents(self, query): return len(self.find(query).docs)
    def find_one(self, query): return self.docs.get(query["_id"])
    def insert_one(self, doc):
        doc = dict(doc); doc["_id"] = ObjectId(); self.docs[doc["_id"]] = doc
        return type("InsertResult", (), {"inserted_id": doc["_id"]})()
    def update_one(self, query, update): self.docs[query["_id"]].update(update["$set"])
    def delete_one(self, query):
        found = self.docs.pop(query["_id"], None) is not None
        return type("DeleteResult", (), {"deleted_count": int(found)})()


def conditions(**updates):
    return {"plant_id": 4136001, "date_time": "13-06-2020 13:30", "ambient_temperature": 31.0,
            "module_temperature": 45.0, "irradiation": 0.7, **updates}


def reading_body(**updates): return {**conditions(), "ac_power": 1000.0, **updates}


def setup_users(monkeypatch):
    users = {name: {"password_hash": hash_password(name + "-password-long"), "role": name}
             for name in ("viewer", "operator", "admin")}
    monkeypatch.setenv("API_USERS_JSON", json.dumps(users))


def auth(username, password=None):
    return (username, password if password is not None else username + "-password-long")


def collections(monkeypatch):
    readings, forecasts = Collection(), Collection()
    monkeypatch.setattr(app.db, "readings", readings)
    monkeypatch.setattr(app.db, "forecasts", forecasts)
    return readings, forecasts


def test_missing_and_invalid_basic_credentials_are_rejected(monkeypatch):
    setup_users(monkeypatch); collections(monkeypatch)
    missing = client.get("/readings")
    assert missing.status_code == 401
    assert missing.headers["www-authenticate"].startswith("Basic")
    assert client.get("/readings", auth=("viewer", "incorrect")).status_code == 401
    assert client.get("/readings", auth=("unknown", "some-password")).status_code == 401


def test_missing_or_invalid_user_configuration_fails_closed(monkeypatch):
    monkeypatch.delenv("API_USERS_JSON", raising=False)
    assert client.get("/readings", auth=("admin", "password")).status_code == 503
    monkeypatch.setenv("API_USERS_JSON", "{not json")
    assert client.get("/forecasts", auth=("admin", "password")).status_code == 503


def test_basic_credentials_are_refused_over_http(monkeypatch):
    setup_users(monkeypatch); collections(monkeypatch)
    response = http_client.get("/readings", auth=auth("viewer"))
    assert response.status_code == 400
    assert response.json()["detail"] == "HTTPS is required for Basic authentication"


def test_all_reading_and_forecast_crud_requires_authentication(monkeypatch):
    setup_users(monkeypatch); collections(monkeypatch)
    calls = [("get", "/readings"), ("get", "/readings/bad"), ("post", "/readings", reading_body()),
      ("put", "/readings/bad", {}), ("delete", "/readings/bad"), ("get", "/forecasts"),
      ("get", "/forecasts/bad"), ("post", "/forecasts", conditions()),
      ("put", "/forecasts/bad", {}), ("delete", "/forecasts/bad")]
    for method, path, *body in calls:
        kwargs = {"json": body[0]} if body else {}
        assert getattr(client, method)(path, **kwargs).status_code == 401
        kwargs["auth"] = ("admin", "wrong-password")
        assert getattr(client, method)(path, **kwargs).status_code == 401


def test_health_endpoint_is_public_and_does_not_require_database():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_jwt_token_endpoint_is_removed():
    assert client.post("/auth/token", json={"username": "viewer", "password": "password"}).status_code == 404


def test_viewer_can_read_but_cannot_create(monkeypatch):
    setup_users(monkeypatch); readings, forecasts = collections(monkeypatch)
    assert client.get("/readings", auth=auth("viewer")).status_code == 200
    assert client.get("/forecasts", auth=auth("viewer")).status_code == 200
    assert client.post("/readings", json=reading_body(), auth=auth("viewer")).status_code == 403
    assert client.post("/forecasts", json=conditions(), auth=auth("viewer")).status_code == 403
    assert not readings.docs and not forecasts.docs


def test_operator_can_read_and_create_but_cannot_update_or_delete(monkeypatch):
    setup_users(monkeypatch); readings, forecasts = collections(monkeypatch)
    with patch.object(app, "model", StubModel()), patch.object(app, "CENTERS", {}), patch.object(app, "REL", None):
        reading = client.post("/readings", json=reading_body(), auth=auth("operator"))
        forecast = client.post("/forecasts", json=conditions(), auth=auth("operator"))
    assert reading.status_code == forecast.status_code == 201
    rid, fid = reading.json()["id"], forecast.json()["id"]
    assert client.get(f"/readings/{rid}", auth=auth("operator")).status_code == 200
    for method in ("put", "delete"):
        kwargs = {"json": {"note": "x"}} if method == "put" else {}
        assert getattr(client, method)(f"/readings/{rid}", auth=auth("operator"), **kwargs).status_code == 403
        assert getattr(client, method)(f"/forecasts/{fid}", auth=auth("operator"), **kwargs).status_code == 403
    assert len(readings.docs) == len(forecasts.docs) == 1


def test_admin_can_read_create_update_and_delete_both_resources(monkeypatch):
    setup_users(monkeypatch); collections(monkeypatch)
    with patch.object(app, "model", StubModel()), patch.object(app, "CENTERS", {}), patch.object(app, "REL", None):
        reading = client.post("/readings", json=reading_body(), auth=auth("admin"))
        forecast = client.post("/forecasts", json=conditions(), auth=auth("admin"))
        assert reading.status_code == forecast.status_code == 201
        rid, fid = reading.json()["id"], forecast.json()["id"]
        assert client.get(f"/readings/{rid}", auth=auth("admin")).status_code == 200
        assert client.get(f"/forecasts/{fid}", auth=auth("admin")).status_code == 200
        assert client.put(f"/readings/{rid}", json={"note": "x"}, auth=auth("admin")).status_code == 200
        assert client.put(f"/forecasts/{fid}", json={"note": "x"}, auth=auth("admin")).status_code == 200
        assert client.delete(f"/readings/{rid}", auth=auth("admin")).status_code == 200
        assert client.delete(f"/forecasts/{fid}", auth=auth("admin")).status_code == 200


def test_predict_remains_public_and_zero_irradiation_skips_model():
    estimator = StubModel()
    with patch.object(app, "model", estimator):
        response = client.post("/predict", json=conditions(irradiation=0.0))
    assert response.status_code == 200 and response.json()["predicted_ac_power_kw"] == 0.0
    assert not estimator.called


def assess(actual=1000.0, center=0.0, threshold=500.0, relative=None, irradiation=0.7, evidence=None):
    reading = app.Reading(**conditions(irradiation=irradiation), ac_power=actual, **(evidence or {}))
    centers = {} if center is None else {"4136001": center}
    with patch.object(app, "model", StubModel()), patch.object(app, "CENTERS", centers), \
         patch.object(app, "THRESHOLDS", {"4136001": threshold}), patch.object(app, "REL", relative):
        return app.assess(reading, actual, app.evidence_of(reading))


def test_both_available_rules_false_is_confirmed_normal():
    r = assess(relative={"min_predicted_kw": 100, "ratio": 0.5})
    assert r["size_rule"] is False and r["relative_rule"] is False and r["deviation"] is False
    assert r["calibration_status"] == "complete" and r["reason"].startswith("Normal")


def test_size_rule_true():
    r = assess(actual=100, threshold=800, relative={"min_predicted_kw": 100, "ratio": 0.05})
    assert r["size_rule"] is True and r["deviation"] is True


def test_relative_rule_true():
    r = assess(actual=700, center=-300, relative={"min_predicted_kw": 100, "ratio": 0.8})
    assert r["size_rule"] is False and r["relative_rule"] is True and r["deviation"] is True


def test_size_unavailable_relative_false_is_undetermined():
    r = assess(center=None, relative={"min_predicted_kw": 100, "ratio": 0.5})
    assert r["size_rule"] is None and r["relative_rule"] is False and r["deviation"] is None
    assert r["calibration_status"] == "partial" and "undetermined" in r["reason"].lower()


def test_relative_unavailable_size_false_is_undetermined():
    r = assess(relative=None)
    assert r["size_rule"] is False and r["relative_rule"] is None and r["deviation"] is None


def test_both_rules_unavailable_are_undetermined():
    r = assess(center=None, relative=None)
    assert r["size_rule"] is None and r["relative_rule"] is None and r["deviation"] is None
    assert r["calibration_status"] == "incomplete"


def test_available_true_rule_overrides_other_missing_rule():
    r = assess(actual=700, center=None, relative={"min_predicted_kw": 100, "ratio": 0.8})
    assert r["size_rule"] is None and r["relative_rule"] is True and r["deviation"] is True


def test_missing_centre_and_relative_config_are_not_normal():
    assert assess(center=None, relative={"min_predicted_kw": 100, "ratio": 0.5})["deviation"] is None
    assert assess(relative=None)["deviation"] is None


def test_nighttime_does_not_flag_deviation():
    r = assess(actual=0, irradiation=0.0, relative={"min_predicted_kw": 100, "ratio": 0.5})
    assert not r["daytime"] and r["size_rule"] is False and r["relative_rule"] is False
    assert r["deviation"] is False and "night-time" in r["reason"].lower()


def test_direct_telemetry_survives_incomplete_calibration():
    r = assess(actual=700, center=None, relative=None,
      evidence={"inverters_reporting": 20, "inverters_zero_ac": 1})
    assert r["deviation"] is None and r["calibration_status"] == "incomplete"
    assert "Missing inverter telemetry" in r["likely_causes"]
