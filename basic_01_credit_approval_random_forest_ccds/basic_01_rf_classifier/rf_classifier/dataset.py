from pathlib import Path
import pandas as pd
import requests
from io import StringIO
from loguru import logger
from tqdm import tqdm
import typer

from rf_classifier.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


def download_dataset(url: str) -> pd.DataFrame:
    """Baixa dataset da web e retorna como DataFrame."""
    logger.info(f"Baixando dataset de {url} ...")
    response = requests.get(url)
    response.raise_for_status()  # levanta erro se houver falha
    df = pd.read_csv(StringIO(response.text))
    logger.info(f"Dataset baixado com shape {df.shape}")
    return df


@app.command()
def main(
    url: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
):
    """
    Baixa dataset da web e salva em `data/processed/dataset.csv`.
    """
    # Criar diretórios se não existirem
    output_path.parent.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Baixar dataset
    df = download_dataset(url)

    # Progresso simbólico para iniciantes
    for i in tqdm(range(len(df)), desc="Processing dataset"):
        if i == len(df) // 2:
            logger.info("Meio do processo alcançado...")

    # Salvar CSV
    df.to_csv(output_path, index=False)
    logger.success(f"Dataset salvo em {output_path}")


if __name__ == "__main__":
    app()
