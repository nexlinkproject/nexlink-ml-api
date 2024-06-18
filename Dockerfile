FROM ubuntu:22.04

ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y python3.11 python3.11-venv python3-pip && \
    apt-get install -y libatlas-base-dev && \
    apt-get clean

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt /app/
RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app

# Expose the port the app runs on
EXPOSE 80

# Specify the command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--reload"]
