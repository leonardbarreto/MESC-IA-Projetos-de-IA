from pathlib import Path
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from loguru import logger
import typer

# Lista padrão de colunas do dataset de crédito (UCI)
CRX_COLUMNS = [
    "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10",
    "A11", "A12", "A13", "A14", "A15", "A16"
]

app = typer.Typer()

def preprocess_features(df: pd.DataFrame, target_col: str):
    """
    Pré-processa o dataset:
    - Separa features e target.
    - Faz one-hot encoding de variáveis categóricas.
    - Faz padronização de variáveis numéricas.
    """

    # Caso a coluna alvo não exista, tentar usar a última coluna
    if target_col not in df.columns:
        logger.warning(f"Coluna alvo '{target_col}' não encontrada. "
                       f"Usando a última coluna '{df.columns[-1]}' como alvo.")
        target_col = df.columns[-1]

    if target_col not in df.columns:
        raise KeyError(f"A coluna alvo '{target_col}' não existe no dataset. "
                       f"Colunas disponíveis: {list(df.columns)}")

    # Separa X e y
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Identificar colunas categóricas e numéricas
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(exclude=['object']).columns

    logger.info(f"Colunas categóricas: {list(categorical_cols)}")
    logger.info(f"Colunas numéricas: {list(numerical_cols)}")

    # Pré-processador
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', StandardScaler(), numerical_cols)
        ]
    )

    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    X_processed = pipeline.fit_transform(X)

    # Retorna DataFrame processado para facilitar o salvamento
    X_df = pd.DataFrame(
        X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed
    )

    return X_df, y


@app.command()
def main(
    input_path: Path,
    output_path: Path,
    target_col: str = "A16",
):
    """
    Lê dataset bruto, pré-processa features e salva em output_path.
    """
    logger.info(f"Lendo dataset de {input_path} ...")
    df = pd.read_csv(input_path, header=None, names=CRX_COLUMNS, na_values='?')

    logger.info(f"Pré-processando features...")
    X_scaled, y = preprocess_features(df, target_col)

    df_features = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_path, index=False)
    logger.success(f"Features salvas em {output_path}")


if __name__ == "__main__":
    app()
