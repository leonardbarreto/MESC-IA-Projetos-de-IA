import os
from pathlib import Path

import pandas as pd
import typer
from loguru import logger
from sklearn.datasets import load_iris
from tqdm import tqdm

from mlflow_ccds.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

def load_dataset(loader_func, **kwargs) -> pd.DataFrame:
    """
    Carrega um dataset usando uma função fornecida.

    Parameters
    ----------
    loader_func : callable
        Função que retorna um objeto tipo Bunch ou DataFrame.
    **kwargs : dict
        Argumentos adicionais para passar à função loader_func.

    Returns
    -------
    pd.DataFrame
        DataFrame do dataset carregado.
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
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a ser salvo.
    filename : str
        Nome do arquivo CSV.
    
    Returns
    -------
    str
        Caminho do arquivo salvo.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    raw_path = os.path.join(RAW_DATA_DIR, filename)
    df.to_csv(raw_path, index=False)
    logger.success(f"✅ Dataset salvo (raw) em: {raw_path}")
    return raw_path

def save_processed(df: pd.DataFrame, filename: str = "processed.csv") -> str:
    """
    Aplica pré-processamento simples e salva na pasta PROCESSED_DATA_DIR.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a ser processado e salvo.
    filename : str
        Nome do arquivo CSV.
    
    Returns
    -------
    str
        Caminho do arquivo salvo.
    """
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    df_processed = df.copy()
    # Exemplo de pré-processamento: renomear colunas para minúsculas
    df_processed.columns = [col.lower() for col in df_processed.columns]

    processed_path = os.path.join(PROCESSED_DATA_DIR, filename)
    df_processed.to_csv(processed_path, index=False)
    logger.success(f"✅ Dataset salvo (processed) em: {processed_path}")
    return processed_path


app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
):

    logger.info("Processing dataset...")
    
    """
    Exemplo de execução: carregar dataset Iris do sklearn e salvar raw + processed
    """
    from sklearn.datasets import load_iris
    df = load_dataset(load_iris, as_frame=True)
    save_raw(df, "iris_raw.csv")
    save_processed(df, "iris.csv")

    
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")


if __name__ == "__main__":
    app()
