# mlflow_ccds

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A short description of the project.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         mlflow_ccds and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── mlflow_ccds   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mlflow_ccds a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```
# Instalar projeto
```bash
git clone
```
# Configurar o ambiente Conda
## Criar um novo ambiente
```bash
conda env create -f environment.yml
```
## Atualizar um ambiente existente
> Você pode atualizar o valor do parâmetro ´´´name:´´´ no arquivo _environment.yml_ para corresponder ao nome de um ambiente já existente na sua máquina

```bash
conda env update -f environment.yml --prune
```
## Ativar o ambiente
```bash
conda activate _nome_do_ambiente_
```
## 🔬 Referências Científicas e Técnicas (Resumo)

### 1. Sobre Random Forests (Floresta Aleatória)

O **Random Forest** é o principal algoritmo de classificação utilizado neste projeto. Ele é um método de aprendizado supervisionado que aumenta a precisão e a robustez ao construir múltiplas árvores de decisão e usar a média (para regressão) ou a moda (para classificação) de suas previsões.

* O conceito foi formalmente consolidado por **Leo Breiman** em 2001.
* **Referências de Leitura:**
    * O trabalho seminal de **Breiman (2004)** sobre Random Forests.
    * O Capítulo 8 de **"An Introduction to Statistical Learning"**, que discute métodos baseados em árvores.

### 2. Sobre o Scikit-learn (sklearn)

A implementação prática do modelo é feita usando a biblioteca **Scikit-learn** (`sklearn`).

* O Scikit-learn é a biblioteca padrão em Python para Machine Learning, conhecida por sua uniformidade de API e vasta gama de algoritmos.
* A escolha do `sklearn` segue a documentação e os padrões apresentados no paper de **Pedregosa et al. (2011)** que introduziu a biblioteca à comunidade científica.

--------

