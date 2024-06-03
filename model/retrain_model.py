import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
from google.cloud import bigquery, storage

def retrain_model():
    client = bigquery.Client()
    query = """
    SELECT * FROM `your-project.your_dataset.feedback`
    """
    feedback_data = client.query(query).to_dataframe()

    X = feedback_data[['taskId', 'project_id', 'time_limit']]
    y = feedback_data['task_duration']

    model = LinearRegression()
    model.fit(X, y)

    local_model_path = "/tmp/new_model.pkl"
    joblib.dump(model, local_model_path)

    bucket_name = "your-bucket-name"
    model_path = "path/to/model.pkl"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(model_path)
    blob.upload_from_filename(local_model_path)
    print(f"New model uploaded to {model_path}")

if __name__ == "__main__":
    retrain_model()
