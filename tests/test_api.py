def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_single(client, sample_student):
    response = client.post("/api/predict/single", json=sample_student)
    assert response.status_code == 200

    data = response.json()
    assert "will_worsen" in data
    assert "probability" in data
    assert isinstance(data["will_worsen"], bool)
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_batch(client, sample_student):
    response = client.post("/api/predict", json={"students": [sample_student, sample_student]})
    assert response.status_code == 200

    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2


def test_predict_invalid_payload(client):
    response = client.post("/api/predict/single", json={"ieg": 7.5})
    assert response.status_code == 422


def test_model_info(client):
    response = client.get("/api/model/info")
    assert response.status_code == 200

    data = response.json()
    assert "feature_columns" in data
    assert "feature_importance" in data
    assert "metrics" in data
    assert "f1" in data["metrics"]
