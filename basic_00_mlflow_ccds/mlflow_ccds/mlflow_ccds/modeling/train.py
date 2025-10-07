# mlflow_ccds/modeling/train.py
from pathlib import Path
from loguru import logger
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import typer
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from mlflow_ccds.config import PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR
from mlflow_ccds.plots import plot_confusion_matrix

app = typer.Typer()


def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


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
    experiment_name: str = "mlflow_ccds_experiment",
):
    logger.info(f"📥 Carregando features de {features_path} e labels de {labels_path}")
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path).squeeze()  # Series

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        logger.info("🏋️ Treinando modelo RandomForestClassifier...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = train_model(X_train, y_train)
        y_pred = model.predict(X_test)

        # --- Métricas ---
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        metrics = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        }
        logger.info(f"✅ Métricas (teste): {metrics}")

        # Salvar métricas em CSV
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        metrics_path = FIGURES_DIR / "metrics.csv"
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

        # --- Gráfico matriz de confusão ---
        plot_path = FIGURES_DIR / "confusion_matrix.png"
        plot_confusion_matrix(y_test, y_pred, plot_path)

        # --- Log no MLflow ---
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(plot_path))

        # Salvar modelo local
        model_path = save_model(model, model_path, overwrite)

    return model_path


@app.command()
def cli(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    overwrite: bool = True,
):
    main(features_path, labels_path, model_path, overwrite=overwrite)


if __name__ == "__main__":
    app()
