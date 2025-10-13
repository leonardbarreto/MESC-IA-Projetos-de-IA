# from pathlib import Path
# import logging
# import typer
# import pandas as pd
# from sklearn.datasets import fetch_openml

# from boston_housing import dataset
# from boston_housing.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
# from boston_housing.modeling import train as train_module

# # Inicializa Typer e Logger
# app = typer.Typer()
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# @app.command()
# def main(
#     raw_filename: Path = RAW_DATA_DIR / "boston_raw.csv",
#     processed_filename: Path = PROCESSED_DATA_DIR / "boston.csv",
#     model_type: str = "random_forest",
# ):
#     """
#     Pipeline Boston Housing:
#     1️⃣ Carrega dataset Boston Housing (OpenML) e salva raw/processado
#     2️⃣ Separa features e target
#     3️⃣ Treina modelo de regressão e loga no MLFlow
#     """
#     # --- 1️⃣ Carregar dataset ---
#     logger.info("📥 Carregando Boston Housing dataset...")
#     boston = fetch_openml(name="boston", version=1, as_frame=True)
#     df = boston.frame  # já vem como DataFrame

#     # Salvar raw e processed
#     dataset.save_raw(df, raw_filename.name)
#     processed_path = dataset.save_processed(df, processed_filename.name)
#     logger.info(f"✅ Dataset processado salvo em: {processed_path}")

#     # --- 2️⃣ Separar features e target ---
#     logger.info("🔧 Separando features e target...")
#     df_processed = pd.read_csv(processed_path)
#     if "target" not in df_processed.columns:
#         raise ValueError("Coluna 'target' não encontrada no dataset processado.")
    
#     features_path = PROCESSED_DATA_DIR / "features.csv"
#     labels_path = PROCESSED_DATA_DIR / "labels.csv"

#     df_processed.drop(columns=["target"]).to_csv(features_path, index=False)
#     df_processed["target"].to_csv(labels_path, index=False)
#     logger.info(f"✅ Features salvas em: {features_path}")
#     logger.info(f"✅ Labels salvas em: {labels_path}")

#     # --- 3️⃣ Treinar modelo ---
#     logger.info(f"🏋️ Treinando modelo de regressão: {model_type}")
#     train_module.main(
#         features_path=features_path,
#         labels_path=labels_path,
#         overwrite=True,
#         experiment_name="boston_housing_experiment",
#         model_type=model_type,
#     )

#     logger.info("✅ Pipeline completo!")


# if __name__ == "__main__":
#     app()

from pathlib import Path
import logging
import typer
import pandas as pd
from sklearn.datasets import fetch_openml

from boston_housing import dataset
from boston_housing.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from boston_housing.modeling import train as train_module

# Inicializa Typer e Logger
app = typer.Typer()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@app.command()
def main(
    raw_filename: Path = RAW_DATA_DIR / "boston_raw.csv",
    processed_filename: Path = PROCESSED_DATA_DIR / "boston.csv",
    model_type: str = typer.Option("random_forest", help="Modelo de regressão: random_forest, svr, knn"),
):
    """
    Pipeline Boston Housing:
    1️⃣ Carrega dataset Boston Housing (OpenML) e salva raw/processado
    2️⃣ Separa features e target
    3️⃣ Treina modelo de regressão e loga no MLFlow
    """
    # --- 1️⃣ Carregar dataset ---
    logger.info("📥 Carregando Boston Housing dataset...")
    boston = fetch_openml(name="boston", version=1, as_frame=True)
    df = boston.frame  # já vem como DataFrame

    # Salvar raw e processed
    dataset.save_raw(df, raw_filename.name)
    processed_path = dataset.save_processed(df, processed_filename.name)
    logger.info(f"✅ Dataset processado salvo em: {processed_path}")

    # --- 2️⃣ Separar features e target ---
    logger.info("🔧 Separando features e target...")
    df_processed = pd.read_csv(processed_path)
    if "target" not in df_processed.columns:
        raise ValueError("Coluna 'target' não encontrada no dataset processado.")
    
    features_path = PROCESSED_DATA_DIR / "features.csv"
    labels_path = PROCESSED_DATA_DIR / "labels.csv"

    df_processed.drop(columns=["target"]).to_csv(features_path, index=False)
    df_processed["target"].to_csv(labels_path, index=False)
    logger.info(f"✅ Features salvas em: {features_path}")
    logger.info(f"✅ Labels salvas em: {labels_path}")

    # --- 3️⃣ Treinar modelo ---
    logger.info(f"🏋️ Treinando modelo de regressão: {model_type}")
    train_module.main(
        features_path=features_path,
        labels_path=labels_path,
        overwrite=True,
        experiment_name="boston_housing_experiment",
        model_type=model_type,
    )

    logger.info("✅ Pipeline completo!")


if __name__ == "__main__":
    app()
