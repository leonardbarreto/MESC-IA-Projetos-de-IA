# mlflow_ccds/pipeline.py
from pathlib import Path
import logging
import typer
from mlflow_ccds import dataset
from mlflow_ccds import features as features_module
from mlflow_ccds.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from mlflow_ccds.modeling import train as train_module

# Inicializa Typer e Logger
app = typer.Typer()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

build_features_main = features_module.main

@app.command()
def main(
    loader_func: str = "iris",
    raw_filename: Path = RAW_DATA_DIR / "dataset_raw.csv",
    processed_filename: Path = PROCESSED_DATA_DIR / "dataset.csv",
):
    """
    Pipeline genérico:
    1️⃣ Carrega dataset e salva raw/processado
    2️⃣ Gera features e labels
    3️⃣ Treina modelo e loga no MLFlow
    """
    # --- 1️⃣ Carregar dataset ---
    logger.info("📥 Carregando dataset e salvando raw/processado...")
    if loader_func == "iris":
        from sklearn.datasets import load_iris
        df = dataset.load_dataset(load_iris)
    else:
        raise ValueError("Loader não suportado. Implemente seu loader_func personalizado.")

    # Salvar raw e processed
    dataset.save_raw(df, raw_filename.name)
    processed_path = dataset.save_processed(df, processed_filename.name)
    logger.info(f"✅ Dataset processado salvo em: {processed_path}")

    # --- 2️⃣ Gerar features e labels ---
    logger.info("🔧 Gerando features e labels...")
    try:
        features_path, labels_path = build_features_main(processed_path)
    except Exception as e:
        logger.warning(f"Falha ao gerar features: {e}")
        features_path, labels_path = None, None

    # --- 3️⃣ Treinar modelo ---
    logger.info("🏋️ Treinando modelo...")
    try:
        if features_path and labels_path:
            train_module.main(features_path=features_path, labels_path=labels_path, overwrite=False)
        else:
            logger.warning("Features ou labels não encontrados. Pulei treino.")
    except Exception as e:
        logger.warning(f"Falha no treino: {e}")

    logger.info("✅ Pipeline completo!")


if __name__ == "__main__":
    app()
