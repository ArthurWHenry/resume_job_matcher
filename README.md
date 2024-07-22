# resume_job_matcher

Contains files that allow the user to use an NLP model that checks to see if their resume matches the job description they are applying for.

Currently it's running a sample model to help me get more familiar with nlp/ml.

# Prerequisites

Python3 or pip3 or anaconda installed.

To ensure that you are installing packages into your virtual environment (venv), follow these steps:

# Running Locally

```bash
python app.py
```

## Testing to make sure the Flask API is running

Via curl:

```bash
curl -X POST http://127.0.0.1:5000/match -H "Content-Type: application/json" -d '{"resume": "Your resume text here", "job_description": "Job description text here"}'
```

## Running the Docker Container:

Make sure that the Docker daemon is running and the port is available before running the command below

```bash
docker build -t nlp-ml-service .
docker run -p 5000:5000 nlp-ml-service
```

# Creating the sample model

```bash
python create_model.py
```

# Setting up Virtual Environment

## Step 1: Create a Virtual Environment

Create the virtual environment:

```bash
python -m venv venv
```

This command creates a new directory called venv in your project folder.

## Step 2: Activate the Virtual Environment

Activate the virtual environment:

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

Once activated, you should see the virtual environment’s name in your terminal prompt, indicating that the venv is active (e.g., (venv)).

## Step 3: Install Packages

Install the necessary packages:

```bash
pip install torch transformers flask
```

## Step 4: Verify Installation

Verify that the packages are installed in the virtual environment:

```bash
pip list
```

This command will list all the packages installed in the current environment, which should include torch, transformers, and flask.

## Step 5: Freeze Requirements

Freeze the installed packages into a requirements.txt file:

```bash
pip freeze > requirements.txt
```

This will create a requirements.txt file listing all the installed packages and their versions, which is useful for recreating the environment later or for deployment.

## Step 6: Deactivate the Virtual Environment

Deactivate the virtual environment when you're done working:

```bash
deactivate
# or
conda deactivate
```
