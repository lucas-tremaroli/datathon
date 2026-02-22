import pytest
from pydantic import ValidationError

from datathon.api.models.predict import PredictRequest, StudentFeatures, StudentPrediction


def test_student_features_valid(sample_student):
    features = StudentFeatures(**sample_student)
    assert features.ieg == 7.5
    assert features.age == 14


def test_student_features_missing_field():
    with pytest.raises(ValidationError):
        StudentFeatures(ieg=7.5, iaa=8.0)


def test_predict_request_empty_list():
    with pytest.raises(ValidationError):
        PredictRequest(students=[])


def test_student_prediction_bounds():
    pred = StudentPrediction(will_worsen=True, probability=0.85)
    assert pred.probability == 0.85

    with pytest.raises(ValidationError):
        StudentPrediction(will_worsen=True, probability=1.5)
