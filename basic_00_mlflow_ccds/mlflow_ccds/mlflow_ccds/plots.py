# mlflow_ccds/plots.py
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from mlflow_ccds.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


# -----------------------------------------------------------
# Função para gerar gráfico da matriz de confusão
# -----------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, output_path: Path):
    """
    Gera e salva um gráfico da matriz de confusão.
    """
    logger.info("Gerando matriz de confusão...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    logger.success(f"✅ Matriz de confusão salva em: {output_path}")
    return output_path


# -----------------------------------------------------------
# Nova função: plot de classification report com matriz
# -----------------------------------------------------------
def plot_classification_report(y_true, y_pred, output_path: Path):
    """
    Gera relatório de classificação (precision, recall, f1-score)
    e matriz de confusão, salvando como gráfico.
    """
    logger.info("📊 Gerando classification report e matriz de confusão...")

    # Relatório como DataFrame
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    # Salvar CSV do relatório
    csv_path = output_path.with_suffix(".csv")
    report_df.to_csv(csv_path)
    logger.info(f"📄 Relatório de métricas salvo em: {csv_path}")

    # Plot da matriz de confusão
    plot_confusion_matrix(y_true, y_pred, output_path)
    logger.success("✅ Gráfico de classification report e matriz de confusão gerado.")


# -----------------------------------------------------------
# CLI genérico para gerar qualquer plot
# -----------------------------------------------------------
@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
):
    """
    Gera plots a partir de dados CSV.
    Exemplo: matriz de confusão ou qualquer outro gráfico.
    """
    logger.info(f"📥 Carregando dados de: {input_path}")
    df = pd.read_csv(input_path)

    logger.info("🔧 Processando e gerando plot...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Iteração 5: exemplo de log durante plot.")

    # Exemplo: salvar dataset simples como plot de linha
    plt.figure(figsize=(6, 4))
    df.plot()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    logger.success(f"✅ Plot salvo em: {output_path}")


if __name__ == "__main__":
    app()
