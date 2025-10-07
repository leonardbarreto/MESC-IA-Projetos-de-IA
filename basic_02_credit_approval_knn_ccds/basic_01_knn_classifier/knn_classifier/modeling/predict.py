# knn_classifier/modeling/predict.py
from pathlib import Path
import pandas as pd
from loguru import logger
import typer
import joblib

app = typer.Typer()

@app.command()
def main(features_path: Path, model_path: Path, predictions_path: Path):
    logger.info(f"Carregando features de {features_path}...")
    X = pd.read_csv(features_path).drop(columns=["target"], errors="ignore")

    logger.info(f"Carregando modelo de {model_path}...")
    model = joblib.load(model_path)

    logger.info("Realizando predições...")
    predictions = model.predict(X)

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions, columns=["prediction"]).to_csv(predictions_path, index=False)
    logger.success(f"Predições salvas em {predictions_path}")

if __name__ == "__main__":
    app()
