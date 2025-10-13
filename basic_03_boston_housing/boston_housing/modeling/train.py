from pathlib import Path
from loguru import logger
import pandas as pd
import joblib
import typer
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error, r2_score

from boston_housing.config import PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR
from boston_housing.plots import plot_predictions, plot_residuals

app = typer.Typer()


# ========================
# Escolha dinâmica de modelo (regressão)
# ========================
def get_model(model_type: str):
    model_type = model_type.lower()
    if model_type == "random_forest":
        return RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "linear_regression":
        return LinearRegression()
    elif model_type == "svr":
        return SVR()
    elif model_type == "knn":
        return KNeighborsRegressor()
    else:
        raise ValueError(f"Tipo de modelo '{model_type}' não suportado para regressão.")


def save_model(model, model_path: Path, overwrite: bool = False):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not overwrite and model_path.exists():
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = MODELS_DIR / f"model_{timestamp}.pkl"
    joblib.dump(model, model_path)
    logger.info(f"💾 Modelo salvo em: {model_path}")
    return model_path


def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    overwrite: bool = False,
    experiment_name: str = "boston_housing_experiment",
    model_type: str = "random_forest",
):
    logger.info(f"📥 Carregando features de {features_path} e labels de {labels_path}")
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path).squeeze()  # Series

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"train_{model_type}"):
        logger.info(f"🏋️ Treinando modelo: {model_type}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = get_model(model_type)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # --- Métricas de regressão ---
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        metrics = {"mse": mse, "rmse": rmse, "r2": r2}
        logger.info(f"✅ Métricas (teste): {metrics}")

        # --- Salvar métricas em CSV ---
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        metrics_path = FIGURES_DIR / f"metrics_{model_type}.csv"
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

        # --- Gerar plots ---
        pred_plot_path = FIGURES_DIR / f"predictions_{model_type}.png"
        residual_plot_path = FIGURES_DIR / f"residuals_{model_type}.png"
        plot_predictions(y_test, y_pred, pred_plot_path)
        plot_residuals(y_test, y_pred, residual_plot_path)

        # --- Log no MLflow ---
        mlflow.log_param("model_type", model_type)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(pred_plot_path))
        mlflow.log_artifact(str(residual_plot_path))

        # --- Salvar modelo local ---
        model_path = save_model(model, model_path, overwrite)

    return model_path


@app.command()
def cli(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    overwrite: bool = True,
    model_type: str = typer.Option(
        "random_forest",
        help="Algoritmo a usar (random_forest, linear_regression, svr)"
    ),
):
    main(features_path, labels_path, model_path, overwrite=overwrite, model_type=model_type)


if __name__ == "__main__":
    app()
