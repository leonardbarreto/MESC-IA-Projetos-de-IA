# mlflow_ccds/modeling/predict.py
from pathlib import Path
import joblib
import pandas as pd
from loguru import logger
import typer
from mlflow_ccds.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


# -----------------------------------------------------------
# Função de predição pura (sem MLflow)
# -----------------------------------------------------------
def predict_model(model_path: Path, X: pd.DataFrame) -> pd.Series:
    """
    Carrega o modelo salvo e realiza predições.
    """
    logger.info(f"📥 Carregando modelo de: {model_path}")
    model = joblib.load(model_path)
    preds = model.predict(X)
    return pd.Series(preds, name="prediction")


# -----------------------------------------------------------
# Função principal de predição
# -----------------------------------------------------------
def predict(model_path: Path, features_path: Path, predictions_path: Path) -> Path:
    """
    Carrega features, gera predições e salva em CSV.
    """
    logger.info(f"📥 Carregando features de: {features_path}")
    X = pd.read_csv(features_path)

    # Predição
    preds = predict_model(model_path, X)

    # Salvar predições
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(predictions_path, index=False)
    logger.info(f"✅ Predições salvas em: {predictions_path}")
    return predictions_path


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------
@app.command()
def main(
    model_path: Path = MODELS_DIR / "model.pkl",
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    predictions_path: Path = PROCESSED_DATA_DIR / "predictions.csv",
):
    """
    Executa predição via CLI (sem MLflow).
    """
    predict(model_path, features_path, predictions_path)


if __name__ == "__main__":
    app()
