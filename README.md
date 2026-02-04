# Diabetes Prediction: Comparative ML Study

Este repositório apresenta um estudo comparativo de modelos de Machine Learning para a predição de diabetes, utilizando dois datasets distintos: o **Indian Pima Diabetes** e dados do **Vigitel Brasil**.

## Objetivo do Projeto
O foco é demonstrar a **capacidade de generalização** de algoritmos de classificação em diferentes populações, utilizando um pipeline de pré-processamento robusto e métricas estatísticas rigorosas.

---

## Stack Técnica
* **Linguagem:** Python 3.10+
* **Bibliotecas:** * `scikit-learn`: Implementação de SVC, Random Forest e Regressão Logística.
  * `pandas` & `numpy`: Manipulação de dados e álgebra linear.
  * `matplotlib`: Visualização de métricas comparativas.

## Diferenciais do Projeto
* **Padronização:** Uso de `StandardScaler` para normalização de features.
* **Métricas de Saúde:** Foco em **Recall** e **F1-Score**, priorizando a redução de falsos negativos em diagnósticos médicos.
* **Validação Cruzada:** Implementação de `cross_validate` com 3-folds para assegurar a consistência estatística dos resultados.
* **Lidando com Desbalanceamento:** Aplicação de `class_weight='balanced'` para ajustar os modelos à realidade dos dados clínicos.

## 📁 Estrutura do Projeto
```text
├── data/               # Arquivos CSV originais
├── main.py             # Script principal de treino e avaliação
├── requirements.txt    # Dependências do projeto
└── README.md           # Documentação
