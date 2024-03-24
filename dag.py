from datetime import datetime
from airflow import DAG
from airflow.operators.empty import EmptyOperator
import pendulum
from airflow.providers.docker.operators.docker import DockerOperator

# Defining the DAG's start date
start_date: datetime = pendulum.today('Europe/Moscow').subtract(days=1)

with DAG(
    dag_id="uplift_model_retraining",
    default_args={
        "owner": "me",
    },
    schedule_interval="@daily",
    start_date=start_date,
    tags=["uplift_model", "streamlit", "docker"],
    catchup=False,
) as dag:
    """
    DAG for daily retraining of the uplift model using Docker containers.

    This DAG includes tasks for starting the process, training and evaluating the models using
    a Docker container, and then concluding the process. The `train_and_evaluate_models` task runs
    a Streamlit application inside a Docker container, which presumably trains and evaluates
    uplift models.
    """
    
    # Start of the DAG's tasks
    start: EmptyOperator = EmptyOperator(task_id="start")
    
    # Task for training and evaluating models inside a Docker container
    train_and_evaluate: DockerOperator = DockerOperator(
        task_id="train_and_evaluate_models",
        image="streamlit_homework:latest",  # Docker image to use
        docker_url='unix://var/run/docker.sock',  # URL to Docker daemon
        network_mode="bridge",  # Network mode for Docker container
        command="streamlit run main.py",  # Command to run in the container
        auto_remove=True,  # Automatically remove the container when the task is finished
    )

    # End of the DAG's tasks
    end: EmptyOperator = EmptyOperator(task_id="end")

    # Defining the task dependencies
    start >> train_and_evaluate >> end
