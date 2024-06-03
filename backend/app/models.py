from pydantic import BaseModel
from datetime import datetime
from typing import List, Dict

class Task(BaseModel):
    id: int
    name: str

class ScheduleRequest(BaseModel):
    project_id: int
    time_limit: int
    tasks: List[Task]

class FeedbackRequest(BaseModel):
    schedule: List[Dict]

class ScheduledTask(BaseModel):
    taskId: int
    startDate: datetime
    endDate: datetime
