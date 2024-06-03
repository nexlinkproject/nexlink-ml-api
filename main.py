from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import List, Dict
from google.cloud import storage, bigquery
import joblib
import os

app = FastAPI()

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

class ScheduleResponse(BaseModel):
    status: str
    message: str
    data: Dict

def download_model(bucket_name: str, model_path: str, local_path: str):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(model_path)
    blob.download_to_filename(local_path)
    print(f"Model downloaded to {local_path}")

def load_model(local_path: str):
    model = joblib.load(local_path)
    return model

def generate_schedule_ml(model, project_id: int, time_limit: int, tasks: List[Task]) -> List[ScheduledTask]:
    task_count = len(tasks)
    start_date = datetime.now()
    task_duration = time_limit // task_count
    schedule = []
    for task in tasks:
        end_date = start_date + timedelta(days=task_duration)
        schedule.append(ScheduledTask(taskId=task.id, startDate=start_date, endDate=end_date))
        start_date = end_date + timedelta(days=1)
    return schedule

@app.post("/predict", response_model=ScheduleResponse)
def predict_schedule(request: ScheduleRequest):
    if not request.tasks:
        raise HTTPException(status_code=400, detail="Tasks cannot be empty")

    try:
        bucket_name = "your-bucket-name"
        model_path = "path/to/model.pkl"
        local_path = "/tmp/model.pkl"
        download_model(bucket_name, model_path, local_path)
        model = load_model(local_path)

        schedule = generate_schedule_ml(model, request.project_id, request.time_limit, request.tasks)
        response_data = {
            "schedule": [task.dict() for task in schedule]
        }
        return ScheduleResponse(status="success", message="Schedule generated successfully", data=response_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest, request: Request):
    feedback_data = feedback.dict()
    client_host = request.client.host
    feedback_data["submitted_at"] = datetime.now().isoformat()
    feedback_data["client_host"] = client_host

    client = bigquery.Client()
    table_id = "your-project.your_dataset.feedback"
    errors = client.insert_rows_json(table_id, [feedback_data])
    if errors:
        raise HTTPException(status_code=500, detail=str(errors))
    
    return {"status": "success", "message": "Feedback submitted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
