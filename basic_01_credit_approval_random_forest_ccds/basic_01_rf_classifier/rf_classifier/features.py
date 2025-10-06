# rf_classifier/features.py
from pathlib import Path
import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler
import typer

app = typer.Typer()

# Nomes das colunas do dataset CRX
CRX_COLUMNS = [
    "A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","target"
]

def preprocess_features(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Converter colunas categóricas em numéricas
    for col in X.select_dtypes(include="object").columns:
        X[col] = pd.factorize(X[col])[0]

    X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
    return X_scaled, y

@app.command()
def main(input_path: Path, output_path: Path, target_col: str = "target"):
    logger.info(f"Carregando dataset de {input_path}...")
    df = pd.read_csv(input_path, header=None, names=CRX_COLUMNS, na_values='?')
    
    logger.info("Pré-processando features...")
    X_scaled, y = preprocess_features(df, target_col)

    df_features = pd.concat([X_scaled, y], axis=1)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_path, index=False)
    logger.success(f"Features salvas em {output_path}")

if __name__ == "__main__":
    app()
