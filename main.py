from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
import pandas as pd
import pickle
from keras.models import load_model
from keras.utils import pad_sequences
from dateutil import parser as date_parser

app = FastAPI()

# Define Pydantic models for request and response
class Task(BaseModel):
    id: int
    title: str
    description: str
    status: str
    priority: str
    dueDate: str
    projectId: int
    assigneeId: int
    createdAt: str
    updatedAt: str

class TasksData(BaseModel):
    tasks: list[Task]

class TasksResponse(BaseModel):
    data: TasksData

# Load the saved model, tokenizer, and label encoder
loaded_model = load_model("text_classify.h5")

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pkl', 'rb') as handle:
    label_encoder = pickle.load(handle)

# Define your max_length
max_length = 100  # Adjust this based on your preprocessing

# Prediction and scheduling functions
def predict_task_labels(model, tokenizer, label_encoder, tasks):
    sequences = tokenizer.texts_to_sequences(tasks)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
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

def assign_workers_to_tasks(task_labels, task_workers):
    worker_assignments = {}
    for idx, (task, worker) in enumerate(zip(task_labels, task_workers)):
        unique_task = f"{task}_{idx}"  # Ensure each task is unique
        duration = get_task_durations(task)
        worker_assignments[unique_task] = (worker, duration)
    return worker_assignments

def calculate_earliest_times(tasks, dependencies):
    earliest_start = {task: 0 for task in tasks}
    earliest_finish = {task: duration for task, (worker, duration) in tasks.items()}

    adj_list = defaultdict(list)
    in_degree = {task: 0 for task in tasks}

    for task, deps in dependencies.items():
        for dep in deps:
            adj_list[dep].append(task)
            in_degree[task] += 1

    topo_order = []
    zero_in_degree_queue = deque([task for task in tasks if in_degree[task] == 0])

    while zero_in_degree_queue:
        task = zero_in_degree_queue.popleft()
        topo_order.append(task)

        for neighbor in adj_list[task]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                zero_in_degree_queue.append(neighbor)

    for task in topo_order:
        for neighbor in adj_list[task]:
            earliest_start[neighbor] = max(earliest_start[neighbor], earliest_finish[task])
            earliest_finish[neighbor] = earliest_start[neighbor] + tasks[neighbor][1]

    return earliest_start, earliest_finish

def calculate_latest_times(tasks, dependencies, project_duration):
    latest_finish = {task: project_duration for task in tasks}
    latest_start = {task: project_duration - duration for task, (worker, duration) in tasks.items()}

    adj_list = defaultdict(list)
    for task, deps in dependencies.items():
        for dep in deps:
            adj_list[task].append(dep)

    for task in reversed(list(tasks.keys())):
        for dep in adj_list[task]:
            latest_finish[dep] = min(latest_finish[dep], latest_start[task])
            latest_start[dep] = latest_finish[dep] - tasks[dep][1]

    return latest_start, latest_finish

def find_critical_path(earliest_start, latest_start):
    critical_path = []
    for task in earliest_start:
        if earliest_start[task] == latest_start[task]:
            critical_path.append(task)
    return critical_path

def generate_daily_schedule(tasks, earliest_start, earliest_finish, start_date):
    worker_schedule = defaultdict(list)
    max_daily_hours = 8

    for task, (worker, duration) in tasks.items():
        start = earliest_start[task]
        remaining_hours = duration

        current_hour = start
        while remaining_hours > 0:
            hours_worked = min(remaining_hours, max_daily_hours - (current_hour % max_daily_hours))
            day = int((current_hour // max_daily_hours) + 1)
            task_date = start_date + timedelta(days=day-1)
            worker_schedule[worker].append((task_date, task, hours_worked))
            remaining_hours -= hours_worked
            current_hour += hours_worked

    for worker in worker_schedule:
        worker_schedule[worker].sort(key=lambda x: x[0], reverse=False)

    return worker_schedule

def critical_path_method(tasks, dependencies, start_date):
    earliest_start, earliest_finish = calculate_earliest_times(tasks, dependencies)
    project_duration = max(earliest_finish.values())
    latest_start, latest_finish = calculate_latest_times(tasks, dependencies, project_duration)
    critical_path = find_critical_path(earliest_start, latest_start)
    worker_schedule = generate_daily_schedule(tasks, earliest_start, earliest_finish, start_date)

    return {
        "project_duration": project_duration,
        "worker_schedule": worker_schedule,
        "critical_path": critical_path
    }

common_dependencies = {
    "Deployment": ["Frontend Development", "Backend Development", "Desain UI/UX"],
    "Frontend Development": ["Desain UI/UX"],
    "Backend Development": ["Frontend Development"]
}

def apply_common_dependencies(predicted_labels):
    predicted_labels = [str(label) for label in predicted_labels]
    unique_labels = [f"{label}_{idx}" for idx, label in enumerate(predicted_labels)]
    label_to_unique = dict(zip(predicted_labels, unique_labels))
    dependencies = {}
    for task, deps in common_dependencies.items():
        if task in label_to_unique:
            unique_task = label_to_unique[task]
            unique_deps = [label_to_unique[dep] for dep in deps if dep in label_to_unique]
            dependencies[unique_task] = unique_deps
    return dependencies

@app.post("/transform_and_schedule")
def transform_and_schedule(response: TasksResponse):
    tasks_data = response.data.tasks

    new_tasks = [task.title for task in tasks_data]
    task_workers = [str(task.assigneeId) for task in tasks_data]

    predicted_labels = predict_task_labels(loaded_model, tokenizer, label_encoder, new_tasks)
    tasks = assign_workers_to_tasks(predicted_labels, task_workers)
    dependencies = apply_common_dependencies(predicted_labels)

    start_date = datetime(2024, 6, 10)

    cpm_results = critical_path_method(tasks, dependencies, start_date)

    project_duration = cpm_results['project_duration']
    project_duration_hours = float(project_duration)

    worker_schedule = cpm_results['worker_schedule']
    worker_schedule_output = {
        worker: [
            {
                "date": date.strftime('%d/%m/%y'),
                "task": task,
                "hours": hours
            } for date, task, hours in schedule
        ] for worker, schedule in worker_schedule.items()
    }

    return {
        "worker_schedule": worker_schedule_output,
        "critical_path": cpm_results['critical_path']
    }
