from pydantic import BaseModel

class AssessmentRequest(BaseModel):
    name: str
    role: str
    ticket_volume: int 
    deadline_proximity: int 
    sleep_quality: int 
    complexity: int 
    interruptions: int 

class PredictionResponse(BaseModel):
    state: str
    eeg_data: dict
    advice: str
