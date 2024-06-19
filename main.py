import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import urllib.request
import re
import pickle
from flask import Flask, request, jsonify
from pydantic import BaseModel
from typing import List, Union
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Function to download model from URL
def download_model_from_url(model_url, save_path):
    urllib.request.urlretrieve(model_url, save_path)

# URL of the model
model_url = "https://storage.googleapis.com/nexlink-ml-api/text_classify.h5"
model_save_path = "text_classify.h5"

# Download the model file
download_model_from_url(model_url, model_save_path)

# Load the saved model
loaded_model = load_model(model_save_path)

# Load tokenizer and label encoder
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pkl', 'rb') as handle:
    label_encoder = pickle.load(handle)

# Define your max_length
max_length = 100  # Adjust this based on your preprocessing

def predict_task_labels(model, tokenizer, label_encoder, tasks):
    sequences = tokenizer.texts_to_sequences(tasks)
    logging.debug(f"Sequences: {sequences}")
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    logging.debug(f"Padded Sequences: {padded_sequences}")
    predictions = model.predict(padded_sequences)
    predicted_classes = np.argmax(predictions, axis=1)
    predicted_labels = label_encoder.inverse_transform(predicted_classes)
    return predicted_labels

def get_task_durations(task):
    data = pd.read_csv('dataset5.csv')
    data_choice = data[data['Label_Task'] == task]
    if data_choice.empty:
        raise ValueError(f"Task '{task}' not found in the dataset.")
    duration = data_choice['Estimated_hours'].iloc[0]
    return duration

class Task(BaseModel):
    taskId: str
    name: str
    startDate: str
    deadline: str
    userID: Union[str, List[str]]  # Updated to handle both single and multiple user IDs
    projectId: str

class RequestData(BaseModel):
    tasks: List[Task]

@app.route('/schedule', methods=['POST'])
def schedule_tasks():
    try:
        request_data = request.get_json()
        logging.debug(f"Request Data: {request_data}")

        tasks_data = request_data.get('data', {}).get('tasks', [])
        logging.debug(f"Tasks Data: {tasks_data}")

        tasks = []
        for task_data in tasks_data:
            task = Task(
                taskId=task_data.get('taskId'),
                name=task_data.get('name'),
                startDate=task_data.get('startDate'),
                deadline=task_data.get('deadline'),
                userID=task_data.get('userID'),
                projectId=task_data.get('projectId')
            )
            tasks.append(task)
        logging.debug(f"Parsed Tasks: {tasks}")

        # Assuming that working hours per day is 8
        working_hours_per_day = 8

        response_data = []
        for task in tasks:
            start_date = datetime.strptime(task.startDate, '%Y-%m-%d')
            deadline = datetime.strptime(task.deadline, '%Y-%m-%d')
            predicted_task = predict_task_labels(loaded_model, tokenizer, label_encoder, [task.name])

            # Calculate the due date based on task duration
            task_duration = int(get_task_durations(predicted_task[0]))

            # Adjust task duration based on the number of users assigned
            if isinstance(task.userID, list):
                task_duration = task_duration / len(task.userID)

            duration_days = int((task_duration + working_hours_per_day - 1) // working_hours_per_day)
            due_date = start_date + timedelta(days=duration_days - 1)

            response_data.append({
                "taskId": task.taskId,
                "userID": task.userID,
                "name": predicted_task[0],
                "startDate": task.startDate,
                "dueDate": due_date.strftime('%Y-%m-%d'),
                "projectId": task.projectId
            })

        logging.debug(f"Response Data: {response_data}")
        return jsonify(response_data)

    except Exception as e:
        logging.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=8080)
