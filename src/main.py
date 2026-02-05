import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class DiabetesPredictor:
    def __init__(self, dataset_type='pima'):
        self.dataset_type = dataset_type
        self.model_results = []
        self.best_models = {}

    def load_and_clean_data(self, filepath):
        """Carrega e trata os dados conforme a fonte (Pima ou Vigitel)"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Arquivo não encontrado em: {filepath}. Verifique a pasta 'datasets'.")
            
        df = pd.read_csv(filepath)
        
        if self.dataset_type == 'pima':
            # No Pima, valores 0 em insulina são tratados como nulos [cite: 14, 107]
            df['Insulin'] = df['Insulin'].replace(0, np.nan)
            df['Insulin'] = df['Insulin'].fillna(df['Insulin'].median())
            # Seleção de atributos conforme relevância clínica [cite: 15, 115]
            X = df[['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction', 'Pregnancies', 'Insulin']]
            y = df['Outcome']
        else:
            # No Vigitel, preenchemos nulos com a média [cite: 23, 111]
            df.fillna(df.mean(numeric_only=True), inplace=True)
            # Atributos específicos do dataset nacional [cite: 24, 117]
            X = df[['Glicemia_Normalizada', 'IMC_Normalizado']]
            y = df['Histórico de Diabetes']
            
        return X, y

    def preprocess_data(self, X, y):
        """Normalização e divisão estratificada dos dados [cite: 7, 112]"""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Divisão 80/20 com estratificação para manter proporção das classes [cite: 7, 21, 118]
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    def get_hyperparameters(self):
        """Define o espaço de busca para otimização [cite: 128, 129]"""
        return {
            "Random Forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5]
            },
            "SVM": {
                "C": [0.1, 1, 10],
                "kernel": ['rbf', 'linear']
            }
        }

    def run_pipeline(self, X_train, X_test, y_train, y_test):
        """Executa treinamento, ajuste e validação cruzada [cite: 68, 127]"""
        param_space = self.get_hyperparameters()
        
        # Modelos definidos no estudo [cite: 121, 160]
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(random_state=42),
            "SVM": SVC(probability=True, random_state=42),
            "Neural Network": MLPClassifier(max_iter=500, random_state=42)
        }

        for name, model in models.items():
            print(f"--- Treinando {name} ---")
            
            # Ajuste de hiperparâmetros com RandomizedSearchCV [cite: 18, 135]
            if name in param_space:
                search = RandomizedSearchCV(model, param_space[name], n_iter=10, cv=3, random_state=42, n_jobs=1)
                search.fit(X_train, y_train)
                model = search.best_estimator_
            else:
                model.fit(X_train, y_train)
            
            self.best_models[name] = model

            # Validação Cruzada (5-fold) para métricas robustas [cite: 132, 142]
            cv_results = cross_validate(model, X_train, y_train, cv=5, 
                                        scoring=['accuracy', 'precision', 'recall', 'f1'],n_jobs=1)
            
            # Extração de métricas conforme metodologia [cite: 133, 134]
            self.model_results.append({
                "Modelo": name,
                "Acurácia": cv_results['test_accuracy'].mean(),
                "Precisão": cv_results['test_precision'].mean(),
                "Recall": cv_results['test_recall'].mean(),
                "F1-Score": cv_results['test_f1'].mean()
            })

    def plot_results(self):
        """Gera visualizações comparativas [cite: 9, 137]"""
        results_df = pd.DataFrame(self.model_results)
        plt.figure(figsize=(12, 7))
        sns.barplot(data=results_df.melt(id_vars="Modelo"), x="Modelo", y="value", hue="variable")
        plt.title(f"Desempenho dos Modelos - Dataset {self.dataset_type.upper()}")
        plt.ylim(0, 1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    def print_summary_table(self):
        df = pd.DataFrame(self.model_results)
        df_sorted = df.sort_values(by="F1-Score", ascending=False)

        print("\n" + "="*70)
        print("RESUMO FINAL DOS MODELOS (Validação Cruzada)")
        print("="*70)
        print(df_sorted.to_string(index=False, float_format="%.4f"))
        print("="*70)

    def plot_confusion_matrices(self, X_test, y_test):
        for name, model in self.best_models.items():
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)

            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.title(f"Matriz de Confusão - {name}")
            plt.tight_layout()
            plt.show()

    def plot_feature_importance(self, X, feature_names):
        if "Random Forest" not in self.best_models:
            print("Random Forest não foi treinado.")
            return

        model = self.best_models["Random Forest"]
        importances = model.feature_importances_

        imp_df = pd.DataFrame({
            "Variável": feature_names,
            "Importância": importances
        }).sort_values(by="Importância", ascending=True)

        plt.figure(figsize=(8, 5))
        plt.barh(imp_df["Variável"], imp_df["Importância"])
        plt.title("Importância das Variáveis - Random Forest")
        plt.xlabel("Importância")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Configuração de argumentos via terminal
    parser = argparse.ArgumentParser(description="Pipeline de Ciência de Dados para Predição de Diabetes")
    parser.add_argument('--dataset', type=str, default='pima', choices=['pima', 'vigitel'], 
                        help='Escolha o dataset para análise (pima ou vigitel)')
    args = parser.parse_args()

    # Lógica de diretórios resiliente
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_FILENAME = 'diabetes.csv' if args.dataset == 'pima' else 'vigitel_2023.csv'
    
    # Procura na pasta datasets que deve estar no mesmo nível da pasta src
    DATA_PATH = os.path.join(BASE_DIR, '..', 'datasets', DATASET_FILENAME)

    # Início da Pipeline
    predictor = DiabetesPredictor(dataset_type=args.dataset)
    
    try:
        X, y = predictor.load_and_clean_data(DATA_PATH)
        X_train, X_test, y_train, y_test = predictor.preprocess_data(X, y)
        predictor.run_pipeline(X_train, X_test, y_train, y_test)
        predictor.print_summary_table()
        predictor.plot_results()
        predictor.plot_confusion_matrices(X_test, y_test)
        predictor.plot_feature_importance(X, X.columns)
    except Exception as e:
        print(f"\nERRO NA EXECUÇÃO: {e}")
