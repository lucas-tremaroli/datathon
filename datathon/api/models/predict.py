from pydantic import BaseModel, Field


class StudentFeatures(BaseModel):
    """
    Features for a single student prediction.
    """
    ieg: float = Field(..., description="Engagement indicator (IEG)")
    iaa: float = Field(..., description="Self-assessment indicator (IAA)")
    ips: float = Field(..., description="Psychosocial indicator (IPS)")
    ida: float = Field(..., description="Academic performance indicator (IDA)")
    ian: float = Field(..., description="Level adequacy indicator (IAN)")
    ipv: float = Field(..., description="Turning point indicator (IPV)")
    inde: float = Field(..., description="Overall development index (INDE)")
    stone: int = Field(..., description="Current stone category (encoded)")
    age: int = Field(..., description="Student age")


class PredictRequest(BaseModel):
    """
    Request model for the predict endpoint.
    """
    students: list[StudentFeatures] = Field(
        ...,
        description="List of students with their features for prediction",
        min_length=1,
    )


class StudentPrediction(BaseModel):
    """
    Prediction result for a single student.
    """
    will_worsen: bool = Field(..., description="Whether lag is predicted to worsen")
    probability: float = Field(
        ..., description="Probability of lag worsening (0-1)", ge=0.0, le=1.0
    )


class PredictResponse(BaseModel):
    """
    Response model for the predict endpoint.
    """
    predictions: list[StudentPrediction] = Field(
        ..., description="Predictions for each student in the request"
    )
