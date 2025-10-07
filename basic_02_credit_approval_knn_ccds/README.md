# Credit Approval Classification

Projeto de exemplo baseado na estrutura **Cookiecutter Data Science (CCDS)**, adaptado para execução local (VSCode).

## 🎯 Objetivo
Prever se um cliente será aprovado para crédito com base em atributos socioeconômicos e financeiros, utilizando **classificação supervisionada**.

## 🧠 Estrutura do Projeto
```
credit_approval_ccds/
│
├── data/
│   ├── raw/               <- Dados originais
│   ├── processed/         <- Dados tratados
│
├── notebooks/
│   └── 01_classificacao_credito.ipynb
│
├── src/
│   ├── data/
│   │   └── make_dataset.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   └── train_model.py
│   └── visualization/
│       └── visualize.py
│
├── models/
│   └── best_model.pkl
│
├── reports/
│   └── figures/
│       └── confusion_matrix.png
│
└── requirements.txt
```
## 🔬 Referências Científicas e Técnicas (Resumo)

### 1. Sobre k-Nearest Neighbors (k-NN)

O **k-Nearest Neighbors (k-NN)** é o principal algoritmo de classificação utilizado neste projeto. É um método não-paramétrico e de **aprendizado preguiçoso** (*lazy learning*), onde a função é apenas aproximada localmente, e todo o cálculo é adiado até que uma consulta seja feita. A classificação de um ponto é determinada pela maioria de votos de seus $k$ vizinhos mais próximos no espaço de *features*.

* O método é um dos mais antigos em reconhecimento de padrões, com o conceito fundamental datando de trabalhos em 1951.
* **Referências de Leitura:**
    * O trabalho clássico que popularizou o método para classificação: **Fix, E. & Hodges, J.L. (1951).** *Discriminatory analysis. Nonparametric discrimination: Consistency properties.*
    * Análise detalhada do algoritmo e suas aplicações em **"The Elements of Statistical Learning"** (Capítulo 13) por Hastie, Tibshirani e Friedman.

### 2. Sobre o Scikit-learn (sklearn)

A implementação prática do modelo é feita usando a biblioteca **Scikit-learn** (`sklearn`).

* O Scikit-learn é a biblioteca padrão em Python para Machine Learning, conhecida por sua uniformidade de API e vasta gama de algoritmos.
* A escolha do `sklearn` segue a documentação e os padrões apresentados no paper de **Pedregosa et al. (2011)** que introduziu a biblioteca à comunidade científica.

---
