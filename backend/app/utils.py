from google.cloud import storage
import joblib
from datetime import datetime, timedelta
from models import Task, ScheduledTask

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
