FROM python:3.11-slim

# Install Poetry
RUN pip install poetry

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the pyproject.toml and poetry.lock files into the container
COPY pyproject.toml poetry.lock* /usr/src/app/

# Disable virtual environments created by poetry,
# as the docker container itself provides isolation
RUN poetry config virtualenvs.create false

# Install project dependencies
RUN poetry install --no-dev --no-interaction --no-ansi

# Copy the rest of your application's code
COPY . /usr/src/app

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run main.py when the container launches using Streamlit
CMD ["streamlit", "run", "main.py"]
