import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Função para treinar e avaliar modelos
def train_and_evaluate(X, y, description):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'SVM': SVC(probability=True, random_state=42, class_weight='balanced')
    }

    results = []
    for name, model in models.items():
        print(f" Treinando {description}: {name}")
        cv_results = cross_validate(model, X_train, y_train, cv=3, scoring=['accuracy', 'precision', 'recall', 'f1'], n_jobs=-1)
        results.append({
            'Modelo': name,
            'Dataset': description,
            'Acurácia Média': np.mean(cv_results['test_accuracy']),
            'Precisão Média': np.mean(cv_results['test_precision']),
            'Recall Médio': np.mean(cv_results['test_recall']),
            'F1-Score Médio': np.mean(cv_results['test_f1']),
        })
    return pd.DataFrame(results)

# 1. Dataset Indian Pima
path_pima = 'data/diabetes.csv'
if os.path.exists(path_pima):
    df_pima = pd.read_csv(path_pima)
    X_pima = df_pima[['Glucose', 'BMI']]
    y_pima = df_pima['Outcome']
    scaler = StandardScaler()
    X_pima = pd.DataFrame(scaler.fit_transform(X_pima), columns=X_pima.columns)
    X_pima.fillna(X_pima.mean(), inplace=True)
    results_pima = train_and_evaluate(X_pima, y_pima, "Indian Pima")
else:
    print(f"Erro: Arquivo {path_pima} não encontrado.")
    results_pima = pd.DataFrame()

# 2. Dataset Vigitel
path_vigitel = 'data/vigitel_principais_tratados_normalizados.csv'
if os.path.exists(path_vigitel):
    df_vigitel = pd.read_csv(path_vigitel)
    X_vigitel = df_vigitel[['Glicemia_Normalizada', 'IMC_Normalizado']]
    y_vigitel = df_vigitel['Histórico de Diabetes']
    X_vigitel.fillna(X_vigitel.mean(), inplace=True)
    results_vigitel = train_and_evaluate(X_vigitel, y_vigitel, "Vigitel Brasil")
else:
    print(f"Erro: Arquivo {path_vigitel} não encontrado.")
    results_vigitel = pd.DataFrame()

# Consolidação e Visualização
results_combined = pd.concat([results_pima, results_vigitel], ignore_index=True)
print("\n--- Comparação Final de Métricas ---")
print(results_combined)

# Plotagem
metrics = ['Acurácia Média', 'Recall Médio', 'F1-Score Médio']
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, metric in enumerate(metrics):
    for dataset in results_combined['Dataset'].unique():
        subset = results_combined[results_combined['Dataset'] == dataset]
        axes[i].bar(subset['Modelo'], subset[metric], label=dataset, alpha=0.7)
    axes[i].set_title(metric)
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].legend()
plt.tight_layout()
plt.show()
