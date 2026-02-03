from pydantic import BaseModel, Field

class AssessmentRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Member name")
    role: str = Field(..., min_length=1, description="Department role")
    ticket_volume: int = Field(..., ge=0, le=3, description="Ticket intensity (0-3)")
    deadline_proximity: int = Field(..., ge=0, le=3, description="Urgency (0-3)")
    sleep_quality: int = Field(..., ge=0, le=3, description="Sleep level (0-3)")
    complexity: int = Field(..., ge=0, le=3, description="Technical complexity (0-3)")
    interruptions: int = Field(..., ge=0, le=3, description="Context switching frequency (0-3)")

class PredictionResponse(BaseModel):
    state: str
    eeg_data: dict
    advice: str
