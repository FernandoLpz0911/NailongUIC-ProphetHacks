def test_health(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_stub(client, sample_event) -> None:
    response = client.post("/predict", json=sample_event)
    assert response.status_code == 200
    body = response.json()
    assert body["event_id"] == sample_event["event_id"]
    assert abs(body["prediction"]["YES"] + body["prediction"]["NO"] - 1.0) < 1e-6
    assert "rationale" in body
