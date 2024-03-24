from airflow import DAG
from airflow.operators.empty import EmptyOperator
import pendulum
from airflow.providers.docker.operators.docker import DockerOperator  # Updated import path

with DAG(
    dag_id="test_dag",  # Ensure the dag_id is unique and descriptive
    default_args={
        "owner": "me",
    },
    schedule_interval="@once",
    start_date=pendulum.today('Europe/Moscow').subtract(days=1),  # Use subtract for past date
    tags=["custom_dag"],
) as dag:
    start = EmptyOperator(task_id="start")

    run = DockerOperator(
        task_id="run_app",
        image="ubuntu",
        docker_url='unix://var/run/docker.sock',  # Use Unix socket if running Docker locally
        network_mode="bridge",
        command="echo Hello",
        auto_remove=True,  # Remove container after execution
    )

    end = EmptyOperator(task_id="end")

    start >> run >> end
