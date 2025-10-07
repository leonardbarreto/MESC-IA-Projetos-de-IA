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

## ⚙️ Execução
1. Crie e ative um ambiente virtual:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # (Linux/Mac)
   .venv\Scripts\activate   # (Windows)
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

3. Execute o notebook principal em `notebooks/01_classificacao_credito.ipynb`.

## 📚 Referências
- Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow.*
- DrivenData. *Cookiecutter Data Science Template (2018).*
- Han, J., Kamber, M., & Pei, J. (2012). *Data Mining: Concepts and Techniques.*
