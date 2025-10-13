# from pathlib import Path
# from loguru import logger
# from tqdm import tqdm
# import typer
# import pandas as pd

# from boston_housing.config import PROCESSED_DATA_DIR

# app = typer.Typer()


# def build_features(df: pd.DataFrame):
#     """
#     Retorna features e target separadas.
#     """
#     if "target" not in df.columns:
#         raise ValueError("Coluna 'target' não encontrada no dataset.")
#     X = df.drop(columns=["target"])
#     y = df["target"]
#     return X, y


# def main(processed_path: Path = PROCESSED_DATA_DIR / "dataset.csv"):
#     """
#     Gera features e labels a partir do dataset processado e salva os arquivos CSV.
#     Compatível para ser chamado pelo pipeline.
#     """
#     logger.info(f"📥 Carregando dataset de {processed_path}")
#     df = pd.read_csv(processed_path)

#     # Construir features e labels
#     logger.info("🔧 Separando features e labels...")
#     X, y = build_features(df)

#     # Caminhos de saída
#     features_path = PROCESSED_DATA_DIR / "features.csv"
#     labels_path = PROCESSED_DATA_DIR / "labels.csv"

#     # Salvar arquivos
#     logger.info(f"💾 Salvando features em {features_path}")
#     X.to_csv(features_path, index=False)
#     logger.info(f"💾 Salvando labels em {labels_path}")
#     y.to_csv(labels_path, index=False)

#     logger.success("✅ Features e labels gerados com sucesso!")
#     return features_path, labels_path


# @app.command()
# def cli(
#     input_path: Path = PROCESSED_DATA_DIR / "dataset.csv"
# ):
#     """
#     Typer CLI para executar features standalone.
#     """
#     main(input_path)


# if __name__ == "__main__":
#     app()

# boston_housing/features.py
from pathlib import Path
import pandas as pd
from loguru import logger
import typer

from boston_housing.config import PROCESSED_DATA_DIR

app = typer.Typer()


def build_features(df: pd.DataFrame):
    if "target" not in df.columns:
        raise ValueError("Coluna 'target' não encontrada no dataset.")
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def main(processed_path: Path = PROCESSED_DATA_DIR / "boston.csv"):
    logger.info(f"📥 Carregando dataset de {processed_path}")
    df = pd.read_csv(processed_path)
    X, y = build_features(df)

    features_path = PROCESSED_DATA_DIR / "features.csv"
    labels_path = PROCESSED_DATA_DIR / "labels.csv"

    logger.info(f"💾 Salvando features em {features_path}")
    X.to_csv(features_path, index=False)
    logger.info(f"💾 Salvando labels em {labels_path}")
    y.to_csv(labels_path, index=False)

    logger.success("✅ Features e labels gerados com sucesso!")
    return features_path, labels_path


@app.command()
def cli(input_path: Path = PROCESSED_DATA_DIR / "boston.csv"):
    main(input_path)


if __name__ == "__main__":
    app()
