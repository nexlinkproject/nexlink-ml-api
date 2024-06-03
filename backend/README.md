# Backend for Project Planning App

This directory contains the FastAPI backend application.

## Setup

1. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

2. Run the application:
    ```
    uvicorn main:app --reload
    ```

3. Build and run Docker container:
    ```
    docker build -t project-planning-app-backend .
    docker run -p 8000:8000 project-planning-app-backend
    ```
