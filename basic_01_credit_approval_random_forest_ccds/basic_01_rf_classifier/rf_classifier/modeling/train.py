"""
Treinamento de modelo de classificação (RandomForest inicial).

Referência:
- Géron (2019). Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow.
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from tqdm import tqdm
import typer

from rf_classifier.config import MODELS_DIR, PROCESSED_DATA_DIR, FIGURES_DIR

app = typer.Typer()

def plot_confusion_matrix(y_true, y_pred, save_path: Path):
    """
    Plota a matriz de confusão e salva em arquivo.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Matriz de Confusão")
    plt.ylabel("Classe Real")
    plt.xlabel("Classe Prevista")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Matriz de confusão salva em {save_path}")

def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Treina um modelo RandomForest e retorna o modelo treinado.
    """
    model = RandomForestClassifier(random_state=42)
    logger.info("Treinando RandomForest...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    logger.info("Avaliação do modelo:")
    logger.info("\n" + classification_report(y_test, y_pred))

    plot_confusion_matrix(y_test, y_pred, FIGURES_DIR / "confusion_matrix.png")
    return model

@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "labels.csv",
    model_path: Path = MODELS_DIR / "random_forest.pkl",
):
    """
    Treina RandomForest usando features e labels e salva o modelo.
    """
    logger.info("Carregando dados...")
    X = pd.read_csv(features_path)
    y = pd.read_csv(labels_path).squeeze()  # converte DataFrame para Series

    # Para simplificação didática: divisão treino/teste manual (80/20)
    train_size = int(0.8 * len(X))
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    # Progresso simbólico para iniciantes
    for i in tqdm(range(5), desc="Treinamento progressivo"):
        if i == 2:
            logger.info("Checkpoint no passo 2")
    
    model = train_random_forest(X_train, y_train, X_test, y_test)

    # Salvar modelo
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.success(f"Modelo RandomForest salvo em {model_path}")

if __name__ == "__main__":
    app()
