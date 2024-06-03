from fastapi import FastAPI, HTTPException, Request
from models import Task, ScheduleRequest, FeedbackRequest, ScheduleResponse, ScheduledTask
from utils import download_model, load_model, generate_schedule_ml
from google.cloud import bigquery
from datetime import datetime

app = FastAPI()

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
    feedback_data["submitted_at"] = datetime.now().isoformat()
    feedback_data["client_host"] = request.client.host

    client = bigquery.Client()
    table_id = "your-project.your_dataset.feedback"
    errors = client.insert_rows_json(table_id, [feedback_data])
    if errors:
        raise HTTPException(status_code=500, detail=str(errors))
    
    return {"status": "success", "message": "Feedback submitted successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
