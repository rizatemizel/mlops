import os
import pickle
import click
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Set the MLflow tracking URI
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

# Set the experiment name
mlflow.set_experiment("newyork_taxi_experiment")

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):

    with mlflow.start_run():
        # Enable MLflow auto-logging
        mlflow.sklearn.autolog()

        # Load training and validation data
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        # Create and train RandomForestRegressor model
        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)

        # Predict on validation data
        y_pred = rf.predict(X_val)

        # Calculate RMSE
        rmse = mean_squared_error(y_val, y_pred, squared=False)

        # Log RMSE metric
        mlflow.log_metric("RMSE", rmse)

if __name__ == '__main__':
    run_train()