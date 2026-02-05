# Predição de Diabetes com Machine Learning: Uma Abordagem Comparativa

Este repositório contém o projeto de análise preditiva de diabetes mellitus desenvolvido como parte do currículo do **Departamento de Estatística (DEs)** e **Departamento de Computação (DC)** da **Universidade Federal de São Carlos (UFSCar)**.

O objetivo principal é investigar o desempenho de diferentes algoritmos de Aprendizado de Máquina (ML) em identificar o risco de diabetes utilizando dois contextos de dados distintos: dados clínicos controlados e dados de vigilância populacional.

## 📊 Sobre os Datasets

O projeto utiliza dois conjuntos de dados para avaliar a robustez dos modelos:
1.  **Indian Pima Diabetes:** Dados clínicos de 768 pacientes, focados em variáveis como Glicose, IMC, Idade, Gravidezes, Função de Hereditariedade e Insulina.
2.  **Vigitel 2023:** Dados do sistema de vigilância do Ministério da Saúde do Brasil, contendo mais de 21.000 registros com atributos como glicemia normalizada e IMC normalizado.

## 🛠️ Metodologia e Tecnologias

A pipeline de dados foi construída para garantir rigor estatístico e replicabilidade:
* **Tratamento de Dados:** Imputação de valores ausentes pela mediana para o dataset Pima (focado na coluna de Insulina) e preenchimento pela média para o Vigitel.
* **Pré-processamento:** Normalização de atributos via `StandardScaler` e divisão estratificada dos dados (80/20) para manter a proporção das classes original.
* **Otimização:** Busca de hiperparâmetros utilizando `RandomizedSearchCV` para maximizar o desempenho dos modelos.
* **Validação:** Utilização de Validação Cruzada (*5-fold Cross-Validation*) para obtenção de métricas médias confiáveis.

### Modelos Implementados
* Regressão Logística
* Random Forest (Floresta Aleatória)
* SVM (Máquina de Vetores de Suporte)
* MLP (Rede Neural Multicamadas)

## 🚀 Como Executar o Projeto

### 1. Requisitos
Instale as dependências necessárias utilizando o gerenciador de pacotes pip:
```bash
pip install -r requirements.txt
