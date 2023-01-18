from autogluon.tabular import TabularDataset, TabularPredictor


class AutogluonModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.predictor = TabularPredictor.load(context.artifacts.get("predictor_path"))

    def predict(self, context, model_input):
        return self.predictor.predict(model_input)


def log_model():
    model = AutogluonModel()
    predictor_path = predictor.path + "models/" + predictor.get_model_best()
    artifacts = {"predictor_path": predictor_path}
    conda_env = {
        "channels": ["conda-forge"],
        "dependencies": [f"python={python_version()}", "pip"],
        "pip": [f"mlflow=={mlflow.__version__}", f'cloudpickle=="2.2.0"'],
        "name": "mlflow-env",
    }
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=model,
        artifacts=artifacts,
        conda_env=conda_env,
    )


def run_experiments(predictor):
    for i, model_name in enumerate(list(predictor.leaderboard(silent=True)["model"])):
        with mlflow.start_run(run_name=model_name):
            if i == 0:
                log_model()
            info = predictor.info()["model_info"][model_name]
            score = info["val_score"]
            model_type = info["model_type"]
            hyper_params = info["hyperparameters"]
            hyper_params["model_type"] = model_type
            mlflow.log_params(hyper_params)
            mlflow.log_metric("acc", score)


def create_autogluon_experiment(train_df):
    predictor = TabularPredictor(label="to_predict", eval_metric="accuracy").fit(
        train_data=train_df, verbosity=2, presets="medium_quality"
    )
    return predictor


if __name__ == "__main__":
    predictor = create_autogluon_experiment(train_data)

    run_experiments(predictor)

    predictor.leaderboard()
