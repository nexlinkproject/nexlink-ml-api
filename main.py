from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
import pandas as pd
import pickle
from keras.models import load_model
from keras.utils import pad_sequences

app = FastAPI()

# Define Pydantic models for request and response
class Task(BaseModel):
    taskId: str
    name: str
    description: str
    status: str
    startDate: str
    userID: str
    priority: str
    projectId: str

class TasksData(BaseModel):
    tasks: list[Task]

class TasksRequest(BaseModel):
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
        try:
            duration = get_task_durations(task)
            worker_assignments[unique_task] = (worker, duration)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    return worker_assignments

def calculate_earliest_times(tasks, dependencies, start_dates):
    earliest_start = {task: start_dates[task] for task in tasks}
    earliest_finish = {task: start_dates[task] + timedelta(hours=tasks[task][1]) for task in tasks}

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
            earliest_finish[neighbor] = earliest_start[neighbor] + timedelta(hours=tasks[neighbor][1])

    return earliest_start, earliest_finish

def calculate_latest_times(tasks, dependencies, project_duration):
    latest_finish = {task: project_duration for task in tasks}
    latest_start = {task: project_duration - timedelta(hours=tasks[task][1]) for task in tasks}

    adj_list = defaultdict(list)
    for task, deps in dependencies.items():
        for dep in deps:
            adj_list[task].append(dep)

    for task in reversed(list(tasks.keys())):
        for dep in adj_list[task]:
            latest_finish[dep] = min(latest_finish[dep], latest_start[task])
            latest_start[dep] = latest_finish[dep] - timedelta(hours=tasks[dep][1])

    return latest_start, latest_finish

def find_critical_path(earliest_start, latest_start):
    critical_path = []
    for task in earliest_start:
        if earliest_start[task] == latest_start[task]:
            critical_path.append(task)
    return critical_path

def generate_daily_schedule(tasks, earliest_start, earliest_finish):
    worker_schedule = defaultdict(list)
    max_daily_hours = 8

    for task, (worker, duration) in tasks.items():
        start = earliest_start[task]
        remaining_hours = duration
        current_time = start

        while remaining_hours > 0:
            hours_till_end_of_day = max_daily_hours - (current_time.hour % max_daily_hours)
            hours_worked = min(remaining_hours, hours_till_end_of_day)
            task_date = current_time + timedelta(hours=hours_worked)
            worker_schedule[worker].append((current_time.date(), task, hours_worked))
            remaining_hours -= hours_worked
            current_time += timedelta(hours=hours_worked)

            # Move to the next day if there are remaining hours and the current day is full
            if current_time.hour % max_daily_hours == 0:
                current_time = current_time.replace(hour=0) + timedelta(days=1)

    for worker in worker_schedule:
        worker_schedule[worker].sort(key=lambda x: x[0])

    return worker_schedule

def critical_path_method(tasks, dependencies, start_dates):
    earliest_start, earliest_finish = calculate_earliest_times(tasks, dependencies, start_dates)
    
    if not earliest_finish:
        raise ValueError("No tasks found or empty task list.")
    
    project_duration = max(earliest_finish.values()) - min(start_dates.values())
    latest_start, latest_finish = calculate_latest_times(tasks, dependencies, project_duration)
    critical_path = find_critical_path(earliest_start, latest_start)
    worker_schedule = generate_daily_schedule(tasks, earliest_start, earliest_finish)

    return {
        "project_duration": project_duration,
        "worker_schedule": worker_schedule,
        "critical_path": critical_path,
        "earliest_finish": earliest_finish
    }

common_dependencies = {
    "Analisis Kebutuhan": [],
    "Desain UI/UX": ["Analisis Kebutuhan"],
    "Perancangan Basis Data": ["Analisis Kebutuhan"],
    "Pembuatan Basis Data": ["Perancangan Basis Data"],
    "Frontend Development": ["Desain UI/UX"],
    "Backend Development": ["Perancangan Basis Data"],
    "Pengembangan API": ["Backend Development"],
    "Integrasi API": ["Pengembangan API"],
    "Pengujian Unit": ["Frontend Development", "Backend Development"],
    "Pengujian Integrasi": ["Integrasi API", "Frontend Development", "Backend Development"],
    "Integrasi Model": ["Integrasi API"],
    "Pengujian Sistem": ["Pengujian Integrasi"],
    "Pengujian Fungsionalitas": ["Pengujian Sistem"],
    "Pengujian User Acceptance (UAT)": ["Pengujian Fungsionalitas"],
    "Pengujian dan Perbaikan": ["Pengujian User Acceptance (UAT)"],
    "Evaluasi Model": ["Integrasi Model"],
    "Pembersihan dan Preprocessing Data": ["Pengumpulan Data"],
    "Pengumpulan Data": [],
    "Visualisasi Data": ["Pembersihan dan Preprocessing Data"],
    "Implementasi Fitur": ["Frontend Development", "Backend Development"],
    "Dokumentasi": ["Implementasi Fitur", "Frontend Development", "Backend Development"],
    "Deployment": ["Pengujian dan Perbaikan", "Frontend Development", "Backend Development", "Desain UI/UX"],
    "Presentasi dan Demo": ["Deployment"]
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
def transform_and_schedule(request: TasksRequest):
    tasks_data = request.data.tasks

    new_tasks = [task.name for task in tasks_data]
    task_workers = [str(task.userID) for task in tasks_data]
    task_ids = [task.taskId for task in tasks_data]
    task_start_dates = {f"{task.name}_{idx}": datetime.strptime(task.startDate, '%Y-%m-%d') for idx, task in enumerate(tasks_data)}

    predicted_labels = predict_task_labels(loaded_model, tokenizer, label_encoder, new_tasks)
    print(f"Predicted Labels: {predicted_labels}")

    try:
        tasks = assign_workers_to_tasks(predicted_labels, task_workers)
        print(f"Assigned Tasks: {tasks}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    dependencies = apply_common_dependencies(predicted_labels)
    print(f"Dependencies: {dependencies}")

    try:
        cpm_results = critical_path_method(tasks, dependencies, task_start_dates)
    except Exception as e:
        print(f"Error in CPM calculation: {e}")
        raise HTTPException(status_code=500, detail=f"Error in CPM calculation: {str(e)}")

    project_duration = cpm_results['project_duration']
    print(f"Project Duration: {project_duration}")

    earliest_finish = cpm_results['earliest_finish']
    
    response = []
    for task_data in tasks_data:
        task_label = f"{task_data.name}_{task_ids.index(task_data.taskId)}"
        due_date = earliest_finish.get(task_label)
        response.append({
            "taskId": task_data.taskId,
            "name": task_data.name,
            "description": task_data.description,
            "status": task_data.status,
            "startDate": task_data.startDate,
            "dueDate": due_date.strftime('%Y-%m-%d') if due_date else None,
            "priority": task_data.priority,
            "projectId": task_data.projectId
        })

    return response
