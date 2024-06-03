from pydantic import BaseModel
from typing import Dict

class ScheduleResponse(BaseModel):
    status: str
    message: str
    data: Dict
