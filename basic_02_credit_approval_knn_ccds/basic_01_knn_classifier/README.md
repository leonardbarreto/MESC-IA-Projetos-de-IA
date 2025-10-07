# knn_classifier

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
│                         knn_classifier and configuration for tools like black
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
└── knn_classifier   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes knn_classifier a Python module
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

*Estes itens fornecem o **embasamento teórico (k-NN)** e a **ferramenta de implementação (sklearn)** utilizadas no pipeline de classificação.*
--------

