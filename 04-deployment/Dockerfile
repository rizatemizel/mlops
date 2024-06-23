# Use the provided base image
FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the Pipfile and Pipfile.lock into the container
COPY Pipfile* /app/

# Install pipenv
RUN pip install pipenv

# Install dependencies specified in the Pipfile
RUN pipenv install --deploy --ignore-pipfile

# Create the output directory
RUN mkdir -p /workspaces/mlops/04-deployment

# Copy your script into the container
COPY script.py /app/script.py

# Command to run your script with default arguments (example: May 2023)
CMD ["pipenv", "run", "python", "script.py", "2023", "05"]