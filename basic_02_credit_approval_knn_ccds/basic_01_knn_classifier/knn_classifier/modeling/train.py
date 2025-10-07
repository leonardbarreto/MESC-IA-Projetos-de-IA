import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from knn_classifier.features import preprocess_features_main  # ou seu import correto

# Caminhos
dataset_path = "data/processed/dataset.csv"
features_path = "data/processed/features.csv"
target_col = "A16"

# 1️⃣ Pré-processar features
preprocess_features_main(
    input_path=dataset_path,
    output_path=features_path,
    target_col=target_col
)

# 2️⃣ Carregar features processadas
df_features = pd.read_csv(features_path)

# 3️⃣ Separar X e y
X = df_features.drop(columns=[target_col])
y = df_features[target_col]

# 4️⃣ Filtrar classes raras (<2 ocorrências)
counts = y.value_counts()
valid_classes = counts[counts >= 2].index
X = X[y.isin(valid_classes)]
y = y[y.isin(valid_classes)]

# 5️⃣ Escalar features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6️⃣ Dividir dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 7️⃣ Instanciar modelo KNN
knn_model = KNeighborsClassifier(
    n_neighbors=5,        # número de vizinhos
    weights='distance',   # pondera pelos vizinhos mais próximos
    metric='minkowski',   # padrão (equivale à Euclidiana para p=2)
    p=2
)

# 8️⃣ Treinar modelo
knn_model.fit(X_train, y_train)

# 9️⃣ Fazer previsões
y_pred = knn_model.predict(X_test)

# 10️⃣ Avaliar desempenho
print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nRelatório de classificação:\n", classification_report(y_test, y_pred))
