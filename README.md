# Uplift Model Training and EDA on the X5 Dataset

In this repository, I attempted to build a Streamlit dashboard to train an uplift model and perform EDA on the X5 dataset. To simulate data updates, a random 80% of the training data is used. The dashboard allows selecting between the Solo Model and Two Model approaches, and between `CatBoostClassifier` and `RandomForestClassifier`. Daily model retraining is set up in Airflow using the `DockerOperator`.

**Comments:**
1. My RAM wasn't enough to load the dataset, and my computer's "health" wasn't enough to load even a part of it. This was very frustrating and threw off the whole process of solving the task.
2. Because of that, I've described how the problem *would have been* solved if I had been able to load the dataset. In any case, I still launched Airflow and managed to get most of what was required done. There's at least a screenshot showing the airflow-cluster appearing in Docker, and the launch in kind is also there, though it's more of a terminal screenshot.
