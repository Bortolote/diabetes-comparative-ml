import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

def train_and_evaluate(X, y, description):
    # Split com estratificação para manter a proporção das classes
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'SVM': SVC(probability=True, random_state=42, class_weight='balanced')
    }

    results = []
    for name, model in models.items():
        # Validação Cruzada para garantir robustez estatística
        cv_results = cross_validate(model, X_train, y_train, cv=3, scoring=['accuracy', 'precision', 'recall', 'f1'], n_jobs=-1)
        results.append({
            'Modelo': name,
            'Dataset': description,
            'Acurácia Média': np.mean(cv_results['test_accuracy']),
            'Recall Médio': np.mean(cv_results['test_recall']),
            'F1-Score Médio': np.mean(cv_results['test_f1']),
        })
    return pd.DataFrame(results)

# Carregamento dos Datasets baseados nos seus arquivos enviados
try:
    # Dataset 1: Indian Pima
    df_pima = pd.read_csv('data/Base de dados 1 - Indian Pima Diabetes.csv')
    X_pima = df_pima[['Glucose', 'BMI']] # Seleção conforme README_indian_pima
    y_pima = df_pima['Outcome']
    scaler = StandardScaler()
    X_pima = pd.DataFrame(scaler.fit_transform(X_pima), columns=X_pima.columns)
    results_pima = train_and_evaluate(X_pima, y_pima, "Indian Pima")

    # Dataset 2: Vigitel Brasil
    df_vigitel = pd.read_csv('data/Base de dados 2 - Vigitel.csv')
    X_vigitel = df_vigitel[['Glicemia_Normalizada', 'IMC_Normalizado']]
    y_vigitel = df_vigitel['Histórico de Diabetes']
    results_vigitel = train_and_evaluate(X_vigitel, y_vigitel, "Vigitel Brasil")

    # Consolidação
    results_combined = pd.concat([results_pima, results_vigitel], ignore_index=True)
    print(results_combined)
    
except FileNotFoundError as e:
    print(f"Certifique-off que os arquivos CSV estão na pasta /data: {e}")

# Código para gerar gráfico comparativo de F1-Score (Métrica de balanço entre Precisão/Recall)
results_combined.pivot(index='Modelo', columns='Dataset', values='F1-Score Médio').plot(kind='bar')
plt.title('Comparação de Desempenho (F1-Score)')
plt.ylabel('F1-Score')
plt.show()
