from pydantic import BaseModel
from typing import Optional, Any, List


# TODO: add descriptions and details
class Prediction(BaseModel):
    model: str
    y_predict: List[int]
    corresponding_pages: Optional[List[List[Any]]]


class PredictionWrapper(BaseModel):
    file_name: str
    predictions: List[Prediction]
