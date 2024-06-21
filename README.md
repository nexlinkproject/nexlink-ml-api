# Nexlink Machine Learning Main Model API

## Description

This repository contains file exclusively for the predicting task and generating timeline for the user input sentences.

The purpose of this API is to get the request about the user input data and give the response about predicted task and also the completion information about the task

### Main directory files explanation :

- Dockerfile : to run the docker
- dataset5.csv : the main dataset to retrieve the data from. especially for scheduling algorithm
- label_encoder.pkl : the output of trained text classification model
- main.py : main python file
- requirements.txt : requirements that needs to be installed
- test.json : test json file for testing request
- text_classify.h5 : output of the trained text classification NLP Model
- tokenizer.pkl : output of the trained text classification NLP Model

## Step to build the dockerfile

### 1. Build the DockerFile

```
docker build -t feedback-learning-api .
```

### 2. Run the docker

```
docker run -d --name feedback-learning-container -p 8080:8080 feedback-learning-api
```

### 3. Check the logs

```
docker logs feedback-learning-container
```

