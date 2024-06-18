# nexlink-ml-api

## Setup

1. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

2. Run the application:
    ```
    uvicorn app.main:app --reload
    ```

### Predict Task

**Endpoint**

`POST /transform_and_schedule

**Headers**

- Authorization: Bearer `<JWT_TOKEN>`
- Content-Type: `<application/json>`

**Body**
```json
{
    "data": {
        "tasks": [
            {
                "taskId": "2",
                "name": "Pengujian User Acceptance (UAT)",
                "description": "Pengujian User Acceptance (UAT)",
                "status": "in-progress",
                "startDate": "2024-01-02",
                "userID": "1",
                "priority": "high",
                "projectId": "1"
            }
        ]
    }
}
