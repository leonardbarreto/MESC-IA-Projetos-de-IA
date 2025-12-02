# MESC IA - Projetos de Machine Learning

Repositório contendo a sequência didática de projetos de Machine Learning desenvolvidos para o mestrado.

## 🎯 Sobre este Repositório

Esta coleção segue uma progressão lógica do básico ao avançado:
- **Fase 1**: Algoritmos clássicos e métricas básicas
- **Fase 2**: Validação robusta e otimização de hiperparâmetros  
- **Fase 3**: Engenharia de features
- **Fase 4**: Redes Neurais Artificiais
- **Fase 5**: Introdução ao _Deep Learning_


## 🚀 Começando Aqui

### Para Estudantes
1. Comece pela **Fase 1** se é seu primeiro contato com ML
2. Siga a ordem numérica dos projetos dentro de cada fase
3. Use os projetos anteriores como base para os próximos

## Configuração rápida

# 1. Clonar o repositório 
Clone o repositório https://github.com/leonardbarreto/MESC-IA-Projetos-de-IA.git
```bash
git clone https://github.com/leonardbarreto/MESC-IA-Projetos-de-IA.git
```
# ⚠️ Aviso sobre o ambiente Conda

Todos os subprojetos deste repositório usam o mesmo ambiente Conda (`py3-11-13`).

## Como instalar o ambiente
Se ainda não tiver o ambiente, execute:
```bash
conda env create -f environment.yml
conda activate py3-11-13
```
## Como atualizar o ambiente
Se houver alterações no environment.yml (novos pacotes ou versões):
```bash
conda env update -f environment.yml --prune
conda activate py3-11-13
```

## ⚠️ Evite criar novos ambientes Conda por subprojeto, a menos que haja conflito de pacotes.

---

# 🎓 Disciplina de Inteligência Artificial - MESC

Este repositório reúne materiais, códigos e projetos desenvolvidos ao longo da disciplina de **Inteligência Artificial** no mestrado, seguindo uma estrutura organizada e reprodutível para aprendizado e experimentação.

O conteúdo está organizado em semanas, cada uma com temas específicos, atividades práticas e projetos aplicados.

---

## 📚 Conteúdo Programático

### **SEMANA 1 — Apresentação da Disciplina & Fundamentos de Estatística**
- Apresentação do curso, objetivos e metodologia
- Estatística descritiva: média, mediana, variância, desvio padrão
- Distribuições de probabilidade
- Testes de hipótese e intervalos de confiança
- Atividade: Análise exploratória de dataset real

---

### **SEMANA 2 — Python para ML & Ferramentas**
- Ambiente Python: Jupyter, VS Code, ambientes virtuais
- Bibliotecas essenciais: NumPy, Pandas, Matplotlib
- Introdução ao Scikit-learn
- MLflow: conceitos básicos e setup
- Atividade: Pipeline básico de ML com tracking

---
### **SEMANA 3 — Classificação Binária**
- Logistic Regression, SVM, Naive Bayes
- Métricas: Accuracy, Precision, Recall, F1, AUC-ROC
- Curvas de aprendizado
- Tarefa: Detecção de fraudes em transações

---

### **SEMANA 4 — Regressão**
- Algoritmos de regressão: Linear, Polinomial, Random Forest
- Métricas: MSE, RMSE, R², MAE
- Validação cruzada para regressão
- Tarefa: Previsão de preços de imóveis (Boston Housing)

---

### **SEMANA 5 — Classificação Multiclasse**
- Estratégias One-vs-Rest, One-vs-One
- Random Forest, XGBoost para multiclasse
- Matriz de confusão multiclasse
- Tarefa: Classificação de espécies de plantas (Iris)

---

### **SEMANA 6 — Otimização de Modelos**
- Hyperparameter tuning: Grid Search, Random Search
- Feature selection e importance
- Pipelines com Scikit-learn
- Tarefa: Otimização completa de pipeline de classificação

---

### **SEMANA 7 — Agrupamento (Clustering)**
- K-Means, DBSCAN, Hierarchical Clustering
- Métricas: Silhouette Score, Davies-Bouldin
- Determinação do número ideal de clusters
- Tarefa: Segmentação de clientes de e-commerce

---

### **SEMANA 8 — Redução de Dimensionalidade**
- PCA, t-SNE, UMAP
- Feature extraction vs feature selection
- Visualização de dados de alta dimensão
- Tarefa: Análise de componentes de produtos

---

### **SEMANA 9 — Redes Neurais Artificiais**
- Perceptron, MLP, Backpropagation
- Regularização: Dropout, Early Stopping
- CNNs para imagens
- Tarefa: Classificação de imagens de roupas (Fashion-MNIST)

---

### **SEMANA 10 — Transfer Learning & Técnicas Avançadas**
- Transfer Learning: conceitos e aplicações
- Modelos pré-treinados: ResNet, BERT
- Fine-tuning e feature extraction
- Tarefa: Reconhecimento de objetos com modelos pré-treinados

---

### **SEMANA 11 — Projeto Final - Desenvolvimento**
- Escolha do problema e dataset
- Análise exploratória e pré-processamento
- Implementação e experimentação
- MLflow: Tracking completo do projeto
- Entrega: Primeira versão do código e experimentos

---

### **SEMANA 12 — Projeto Final - Artigo Científico**
- Estrutura de artigo científico
- Redação de metodologia e resultados
- Análise crítica dos experimentos
- Preparação de apresentação
- Entrega:
  - Artigo científico completo
  - Código final com MLflow
  - Apresentação dos resultados

---

## 🛠 Ferramentas e Tecnologias

- **Python** (Jupyter, VS Code)
- **Bibliotecas:** NumPy, Pandas, Matplotlib, Scikit-learn, MLflow
- **Controle de versão:** Git e GitHub
- **Estruturação de projetos:** Cookiecutter Data Science (CCDS)

---

## 📂 Estrutura Geral dos Projetos

Cada projeto segue o padrão **Cookiecutter Data Science**: 
```bash
https://cookiecutter-data-science.drivendata.org/
```

## 🚀 Subprojetos Individuais

| 🧩 Projeto | 📘 Descrição | 🔗 Link |
|:-----------|:-------------|:--------|
| **basic_00_mlflow_ccds** | Estrutura base de MLOps com Cookiecutter e MLflow. | [🔗 GitHub](https://github.com/leonardbarreto/basic_00_mlflow_ccds) |
| **basic_01_credit_approval_random_forest_ccds** | Classificador Random Forest. | [🔗 GitHub](https://github.com/leonardbarreto/basic_01_credit_approval_random_forest_ccds.git) |
| **basic_02_credit_approval_knn_ccds** | Classificador kNN. | [🔗 GitHub](https://github.com/leonardbarreto/basic_02_credit_approval_knn_ccds.git) |
| **basic_03_boston_housing** | Regressores: Random Forest, Linear Regression e kNN com otimização de hiperparâmetros (Optuna) e MLflow. | [🔗 GitHub](https://github.com/leonardbarreto/basic_03_boston_housing.git) |
| **advanced_00_classifiy_regression_hyperOptimization_MLflow_ccds** | Classificadores e regressores com otimização de hiperparâmetros (Optuna) e MLflow | [🔗 GitHub](https://github.com/leonardbarreto/advanced_00_classifiy_regression_hyperOptimization_MLflow_ccds.git) |
| **basic_00_clustering** | Clustering com otimização de hiperparâmetros (Optuna) e MLflow | [🔗 GitHub](https://github.com/leonardbarreto/basic_00_clustering.git) |
| **basic_00_forecasting** | Técnicas de forecasting com otimização de hiperparâmetros (Optuna) e MLflow | [🔗 GitHub](https://github.com/leonardbarreto/basic_00_forecasting.git) |
| **basic_00_rna** | Redes Neurais (MLP) com otimização de hiperparâmetros (Optuna) e MLflow | [🔗 GitHub](https://github.com/leonardbarreto/basic_00_rna.git) |
| **basic_01_rna** | Redes Neurais (MLP) com otimização de hiperparâmetros (Optuna) e MLflow | [🔗 GitHub](https://github.com/leonardbarreto/basic_01_rna.git) |











