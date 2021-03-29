from pydantic import BaseModel, Field
from typing import Optional, Any, List


class Prediction(BaseModel):
    model: str = Field(
        ..., 
        description="name of the used model", 
        example="model.hdf5")
    y_predict: List[int] =  Field(
        ..., description="rounded predictions from the model", 
        example=[1,0,0,1,0,0])
    y_exact: Optional[List[float]] =  Field(
        None, 
        description="exact predictions from the model", 
        example=[0.9715075492858887, 3.751408712560078e-07, 3.751408712560078e-07, 0.9715075492858887,  3.751408712560078e-07,  3.751408712560078e-07])
    corresponding_pages: Optional[List[List[str]]] =  Field(
        None, 
        description="corresponding pages from the processed document", 
        example=[["page 0", "page 1", "page 2"], ["page 3", "page 4", "page 5"]])


class PredictionWrapper(BaseModel):
    file_name: str = Field(..., description="name of the processed documents", example="document.pdf")
    predictions: List[Prediction] =  Field(..., description="list of the predictions created for the document")
