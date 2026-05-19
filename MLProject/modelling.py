import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

mlflow.set_tracking_uri("sqlite:///mlflow.db") 
mlflow.set_experiment("CI_Titanic_Experiment")
mlflow.sklearn.autolog()

df = pd.read_csv('titanic_preprocessing.csv')
X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="CI_Random_Forest") as run:
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    rf.score(X_test, y_test)
    
    with open("run_id.txt", "w") as f:
        f.write(run.info.run_id)

print(f"Training CI selesai! Run ID: {run.info.run_id}")