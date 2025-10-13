import os
from pathlib import Path

import pandas as pd
import typer
from loguru import logger
from sklearn.datasets import fetch_openml
from tqdm import tqdm

from boston_housing.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


def load_dataset(loader_func, **kwargs) -> pd.DataFrame:
    """
    Carrega um dataset usando uma função fornecida.
    """
    data = loader_func(**kwargs)

    # Se for um Bunch do sklearn
    if hasattr(data, "data") and hasattr(data, "feature_names"):
        df = pd.DataFrame(data.data, columns=data.feature_names)
        if hasattr(data, "target"):
            df["target"] = data.target
        return df

    # Se já for DataFrame
    if isinstance(data, pd.DataFrame):
        return data

    raise ValueError("O loader_func retornou um objeto desconhecido. Implemente o tratamento.")


def save_raw(df: pd.DataFrame, filename: str = "raw.csv") -> str:
    """
    Salva o DataFrame na pasta RAW_DATA_DIR.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    raw_path = os.path.join(RAW_DATA_DIR, filename)
    df.to_csv(raw_path, index=False)
    logger.success(f"✅ Dataset salvo (raw) em: {raw_path}")
    return raw_path


def save_processed(df: pd.DataFrame, filename: str = "processed.csv") -> str:
    """
    Aplica pré-processamento simples e salva na pasta PROCESSED_DATA_DIR.
    """
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    df_processed = df.copy()

    # Renomear colunas para minúsculas
    df_processed.columns = [col.lower() for col in df_processed.columns]

    # Renomear automaticamente a coluna do target do Boston Housing
    target_candidates = ["medv", "target"]
    for col in target_candidates:
        if col in df_processed.columns:
            df_processed.rename(columns={col: "target"}, inplace=True)
            break

    processed_path = os.path.join(PROCESSED_DATA_DIR, filename)
    df_processed.to_csv(processed_path, index=False)
    logger.success(f"✅ Dataset salvo (processed) em: {processed_path}")
    return processed_path


app = typer.Typer()


@app.command()
def main(
    raw_filename: str = "boston_raw.csv",
    processed_filename: str = "boston.csv",
):
    """
    Carrega o Boston Housing dataset via OpenML, salva versão raw e processed.
    """
    logger.info("📊 Carregando Boston Housing dataset...")
    boston = fetch_openml(name="boston", version=1, as_frame=True)
    df = boston.frame  # já vem como DataFrame

    save_raw(df, raw_filename)
    save_processed(df, processed_filename)

    # Log de progresso
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Metade do processamento concluída.")
    logger.success("✅ Processamento do dataset completo.")


if __name__ == "__main__":
    app()
