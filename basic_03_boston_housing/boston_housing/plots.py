# from pathlib import Path
# from loguru import logger
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import numpy as np

# from boston_housing.config import FIGURES_DIR

# # -----------------------------------------------------------
# # Plot de previsões vs valores reais
# # -----------------------------------------------------------
# def plot_predictions(y_true, y_pred, output_path: Path):
#     """
#     Plota valores preditos vs valores reais.
#     Aceita y_true e y_pred como pandas Series ou numpy arrays.
#     """
#     logger.info("📊 Gerando plot de previsões vs reais...")

#     # Converter para Series se for ndarray
#     if isinstance(y_true, np.ndarray):
#         y_true = pd.Series(y_true)
#     if isinstance(y_pred, np.ndarray):
#         y_pred = pd.Series(y_pred, index=y_true.index)

#     plt.figure(figsize=(6, 5))
#     plt.scatter(y_true, y_pred, alpha=0.7)
#     plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
#     plt.xlabel("Valores reais")
#     plt.ylabel("Valores previstos")
#     plt.title("Previsões vs Valores Reais")
#     plt.tight_layout()

#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     plt.savefig(output_path)
#     plt.close()
#     logger.success(f"✅ Plot de previsões salvo em: {output_path}")
#     return output_path

# # -----------------------------------------------------------
# # Plot de resíduos
# # -----------------------------------------------------------
# def plot_residuals(y_true, y_pred, output_path: Path):
#     """
#     Plota resíduos (erro real - previsto) vs valores previstos.
#     """
#     logger.info("📊 Gerando plot de resíduos...")

#     # Converter para Series se for ndarray
#     if isinstance(y_true, np.ndarray):
#         y_true = pd.Series(y_true)
#     if isinstance(y_pred, np.ndarray):
#         y_pred = pd.Series(y_pred, index=y_true.index)

#     residuals = y_true - y_pred
#     plt.figure(figsize=(6, 5))
#     plt.scatter(y_pred, residuals, alpha=0.7)
#     plt.axhline(0, color='r', linestyle='--')
#     plt.xlabel("Valores previstos")
#     plt.ylabel("Resíduos")
#     plt.title("Resíduos vs Valores Previstos")
#     plt.tight_layout()

#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     plt.savefig(output_path)
#     plt.close()
#     logger.success(f"✅ Plot de resíduos salvo em: {output_path}")
#     return output_path


from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from boston_housing.config import FIGURES_DIR

# -----------------------------------------------------------
# Plot de previsões vs valores reais
# -----------------------------------------------------------
def plot_predictions(y_true, y_pred, output_path: Path):
    """
    Plota valores preditos vs valores reais.
    Aceita y_true e y_pred como pandas Series ou numpy arrays.
    """
    logger.info("📊 Gerando plot de previsões vs reais...")

    # Converter para Series se for ndarray
    if isinstance(y_true, np.ndarray):
        y_true = pd.Series(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.Series(y_pred, index=y_true.index)

    plt.figure(figsize=(6, 5))
    plt.scatter(y_true, y_pred, alpha=0.7, label="Pontos previstos")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label="Linha ideal")
    plt.xlabel("Valores reais")
    plt.ylabel("Valores previstos")
    plt.title("Previsões vs Valores Reais")
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    logger.success(f"✅ Plot de previsões salvo em: {output_path}")
    return output_path

# -----------------------------------------------------------
# Plot de resíduos
# -----------------------------------------------------------
def plot_residuals(y_true, y_pred, output_path: Path):
    """
    Plota resíduos (erro real - previsto) vs valores previstos.
    """
    logger.info("📊 Gerando plot de resíduos...")

    # Converter para Series se for ndarray
    if isinstance(y_true, np.ndarray):
        y_true = pd.Series(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.Series(y_pred, index=y_true.index)

    residuals = y_true - y_pred
    plt.figure(figsize=(6, 5))
    plt.scatter(y_pred, residuals, alpha=0.7, label="Resíduos")
    plt.axhline(0, color='r', linestyle='--', label="Erro zero")
    plt.xlabel("Valores previstos")
    plt.ylabel("Resíduos")
    plt.title("Resíduos vs Valores Previstos")
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    logger.success(f"✅ Plot de resíduos salvo em: {output_path}")
    return output_path
