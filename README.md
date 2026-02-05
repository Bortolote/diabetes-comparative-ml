# Predição de Diabetes com Machine Learning: Uma Abordagem Comparativa

[cite_start]Este repositório contém o projeto de análise preditiva de diabetes mellitus desenvolvido como parte do currículo do **Departamento de Estatística (DEs)** e **Departamento de Computação (DC)** da **Universidade Federal de São Carlos (UFSCar)**[cite: 31, 35, 41, 45].

[cite_start]O objetivo principal é investigar o desempenho de diferentes algoritmos de Aprendizado de Máquina (ML) em identificar o risco de diabetes utilizando dois contextos de dados distintos: dados clínicos controlados e dados de vigilância populacional[cite: 64, 101].

## 📊 Sobre os Datasets

O projeto utiliza dois conjuntos de dados para avaliar a robustez dos modelos:
1.  [cite_start]**Indian Pima Diabetes:** Dados clínicos de 768 pacientes, focados em variáveis como Glicose, IMC e Insulina[cite: 102].
2.  [cite_start]**Vigitel 2023:** Dados do sistema de vigilância do Ministério da Saúde do Brasil, com 21.690 registros, representando uma amostra diversificada da população[cite: 104, 105].

## 🛠️ Metodologia e Tecnologias

A pipeline de dados foi construída para garantir rigor estatístico e replicabilidade:
* [cite_start]**Tratamento de Dados:** Imputação de valores ausentes pela mediana para o dataset Pima e preenchimento pela média para o Vigitel[cite: 107, 111].
* [cite_start]**Pré-processamento:** Normalização de atributos via `StandardScaler` e divisão estratificada dos dados (80/20) para manter a proporção das classes[cite: 112, 116, 118].
* [cite_start]**Validação:** Utilização de Validação Cruzada (*5-fold Cross-Validation*) e busca exaustiva de hiperparâmetros com `RandomizedSearchCV`[cite: 128, 132].

### Modelos Implementados
* [cite_start]Regressão Logística [cite: 122]
* [cite_start]Random Forest [cite: 123]
* [cite_start]Máquina de Vetores de Suporte (SVM) [cite: 124]
* [cite_start]Rede Neural Multicamadas (MLP) [cite: 125]



## 🚀 Como Executar o Projeto

### 1. Requisitos
Certifique-se de ter o Python 3.8+ instalado. Instale as dependências necessárias:
```bash
pip install -r requirements.txt
