# nexlink-ml-api

## Directory Structure

- **backend/**: Contains the FastAPI backend application.
- **model/**: Contains the model retraining script.

## Setup

### Backend

1. Navigate to the `backend` directory:
    ```
    cd backend
    ```

2. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

3. Run the application:
    ```
    uvicorn app.main:app --reload
    ```

4. Build and run Docker container:
    ```
    docker build -t image-name .
    docker run -p 8000:8000 image-name
    ```

### Model Retraining

1. Navigate to the `model` directory:
    ```
    cd model
    ```

2. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

3. Run the retraining script:
    ```
    python retrain_model.py
    ```