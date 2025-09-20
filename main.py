import joblib
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from scipy.sparse import csr_matrix
from typing import Any

from src.data.load import load_data
from src.utils.logger import (print_debug, print_info, print_error, print_warning, print_cross_validation_results, \
                              set_global_debug_mode)


class BibleStylePipeline:

    def __init__(self, config: dict[str, Any], debug: bool = False):
        self.config = config
        self.debug = debug
        self.results = {}
        self.vectorizer = None
        self.model = None
        self.best_params = None

        set_global_debug_mode(debug)

    def load_data(self) -> tuple[list, list]:
        dataset_map = {
            'arcaico_moderno': 'train_arcaico_moderno.csv',
            'complexo_simples': 'train_complexo_simples.csv',
            'literal_dinamico': 'train_literal_dinamico.csv'
        }

        dataset_file = dataset_map[self.config["dataset"]]
        print_info(f"Carregando dados do arquivo: {dataset_file}")

        df = load_data(dataset_file)
        print_info(f"Dataframe: {df.shape[0]} linhas, {df.shape[1]} colunas ")

        text_column = "text"
        label_column = "style"

        if self.debug:
            print_debug(f"   🔍 Colunas do DataFrame: {df.columns.tolist()}")

        texts = df[text_column].astype(str).tolist()

        labels = df[label_column].values

        print_info(f"Extraídos: {len(texts)} textos e {len(labels)} labels")
        print_info(f"Classes únicas: {np.unique(labels)}")

        return texts, labels

    def extract_features(self, texts: list) -> csr_matrix:
        method = self.config["features"]["method"]
        params = self.config["features"]["params"]

        print_info(f"extraindo features com metódo: {method}")

        if method == "bag_of_words":
            from src.features.bag_of_words import create_bag_of_words_vect
            x_vectorized, self.vectorizer = create_bag_of_words_vect(texts, **params)

        elif method == "tfidf":
            raise NotImplementedError

        else:
            raise ValueError(f"Método de features não encontrado: {method}")

        if self.debug:
            print_debug(f"parametros usados{params}")

        return x_vectorized

    def get_model_instance(self, params: dict = None) -> BaseEstimator:
        model_name = self.config["model"]["name"]
        model_params = params or self.config["model"]["params"]

        if self.debug:
            print_debug(f"   🤖 Criando modelo {model_name} com parâmetros: {model_params}")

        if model_name == "logistic_regression":
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(**model_params)

        elif model_name == "mlp":
            from sklearn.neural_network import MLPClassifier
            return MLPClassifier(**model_params)

        else:
            raise ValueError(f"Modelo não suportado: {model_name}")

    def run_grid_search(self, x, y) -> tuple[BaseEstimator, dict]:
        if "grid_search" not in self.config or not self.config["grid_search"]["enable"]:
            print_info("Grid search desabilitado")
            return self.get_model_instance(), {}

        print_info("Executando grid search")

        grid_config = self.config["grid_search"]
        model = self.get_model_instance()

        if self.debug:
            print_debug(f"parametros do grid: {grid_config['param_grid']}")
            print_debug(f"   scoring: {grid_config['scoring']}")
            print_debug(f"   num_folds: {grid_config['num_folds']}")

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=grid_config["param_grid"],
            scoring=grid_config["scoring"],
            cv=grid_config["num_folds"],
            n_jobs=grid_config.get("n_jobs", 1),
            verbose=grid_config.get("verbose", 1)
        )
        grid_search.fit(x, y)

        print_info("Grid Search concluído!")
        print_info(f"Melhores parâmetros: {grid_search.best_params_}")
        print_info(f"Melhor score: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_, grid_search.best_params_

    def evaluate(self, x, y) -> dict[str, Any]:
        from src.evaluation.cross_validation import evaluate_cross_validation

        print_info("Executando validação cruzada...")

        eval_params = self.config['model']['params'].copy()
        if 'n_jobs' in eval_params:
            eval_params['n_jobs'] = 1

        model = self.get_model_instance(eval_params)

        results = evaluate_cross_validation(
            model,
            x,
            y,
            num_folds=self.config['evaluation']['folds'],
            random_state=self.config['evaluation']['random_state']
        )

        return results

    def train_final_model(self, x, y):
        print_info("Treinando modelo final...")

        if self.config['grid_search']['enable']:
            self.model, self.best_params = self.run_grid_search(x, y)
        else:
            self.model = self.get_model_instance()
            self.model.fit(x, y)

            # Avaliação rápida
        train_accuracy = self.model.score(x, y)
        print_info(f"Acurácia no treino completo: {train_accuracy:.3f}")

        return self.model

    def save_model(self):
        if self.model is None:
            print_warning("Nada para salvar - modelo não disponível")
            return

        model_dir = Path("models/saved")
        model_dir.mkdir(exist_ok=True, parents=True)

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{self.config['dataset']}_{self.config['features']['method']}_{self.config['model']['name']}_{timestamp}"

        model_path = model_dir / f"{base_name}_model.pkl"
        joblib.dump(self.model, model_path)

        config_path = model_dir / f"{base_name}_config.json"
        import json
        with open(config_path, "w", encoding="utf-8") as f:
            serializable_config = self._make_config_serializable(self.config)
            json.dump(serializable_config, f, indent=2, ensure_ascii=False)

        results_path = model_dir / f"{base_name}_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            results_data = {
                'mean_accuracy': float(self.results['mean_accuracy']),
                'std_accuracy': float(self.results['std_accuracy']),
                'mean_f1': float(self.results['mean_f1']),
                'std_f1': float(self.results['std_f1']),
                'timestamp': timestamp,
                'dataset': self.config['dataset']
            }
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        if self.vectorizer is not None:
            vectorizer_path = model_dir / f"{base_name}_vectorizer.pkl"
            joblib.dump(self.vectorizer, vectorizer_path)
            if self.debug:
                print_debug(f"Vectorizer salvo: {vectorizer_path}")

        if self.best_params:
            best_params_path = model_dir / f"{base_name}_best_params.json"
            with open(best_params_path, "w", encoding="utf-8") as f:
                json.dump(self.best_params, f, indent=2, ensure_ascii=False)

        print_info(f" Modelo salvos com timestamp: {timestamp}")
        print_info(f" Modelo: {model_path}")
        print_info(f" Configuração: {config_path}")
        print_info(f" Resultados: {results_path}")

        return model_path

    def _make_config_serializable(self, config):
        import numpy as np
        from pathlib import Path

        def convert_value(value):
            if isinstance(value, (np.integer, np.floating)):
                return float(value)
            elif isinstance(value, (np.ndarray, list, tuple)):
                return [convert_value(v) for v in value]  # Chamada recursiva
            elif isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}  # Chamada recursiva
            elif isinstance(value, (Path, set)):
                return str(value)
            else:
                return value

        return convert_value(config)

    def run(self):
        print_info(f"Pipeline: {self.config['dataset']} | "
                   f"{self.config['features']['method']} | "
                   f"{self.config['model']['name']}"
                   )

        if self.debug:
            print_debug(f"Configuração completa: {self.config}")

        try:
            texts, labels = self.load_data()
            x = self.extract_features(texts)

            results = self.evaluate(x, labels)
            self.results = results

            print_cross_validation_results(results, f"{self.config['model']['name']} - {self.config['dataset']}")

            if results['mean_accuracy'] >= self.config['evaluation']['accuracy_threshold']:
                self.train_final_model(x, labels)

                if self.config['evaluation']['save_model']:
                    self.save_model()
            else:
                print_warning(f"Modelo final não treinado - acurácia abaixo do threshold")

            return self.results

        except Exception as e:
            print_error(f"Erro no pipeline {e}")
            if self.debug:
                import traceback
                print_debug(f"Stack trace: {traceback.format_exc()}")
            raise


def main():
    debug_mode = True
    set_global_debug_mode(debug_mode)

    config = {
        'dataset': 'arcaico_moderno',
        'features': {
            'method': 'bag_of_words',
            'params': {'ngram': (1, 3)}
        },
        'model': {
            'name': 'logistic_regression',
            'params': {'max_iter': 1200, 'class_weight': 'balanced', 'random_state': 99}
        },
        'evaluation': {
            'folds': 10,
            'random_state': 99,
            'accuracy_threshold': 0.6,
            'save_model': True
        },
        'grid_search': {
            'enable': False
        }
    }

    pipeline = BibleStylePipeline(config, debug=debug_mode)
    results = pipeline.run()

    print_info(f"Pipeline concluído! Acurácia: {results['mean_accuracy']:.3f}")


if __name__ == "__main__":
    main()


def template_config():
    """
    DOCUMENTAÇÃO DE TODAS AS CONFIGURAÇÕES DE PARÂMETROS DO CONFIG
    """
    return {
        # --- CONFIGURAÇÕES PRINCIPAIS ---
        'dataset': 'arcaico_moderno',  # Possibilidades: 'arcaico_moderno', 'complexo_simples', 'literal_dinamico'

        # --- CONFIGURAÇÕES DE FEATURES ---
        'features': {
            'method': 'bag_of_words',  # Possibilidades: 'bag_of_words', 'tfidf'
            'params': {
                # Parâmetros comuns para ambos os métodos:
                'ngrams': (1, 1),  # Possibilidades: (1,1), (1,2), (1,3), (2,2), (2,3)
                'max_features': None,  # Possibilidades: None, 1000, 5000, 10000 (número máximo de features)
                'min_df': 1,  # Possibilidades: 1, 2, 5 (mínimo de documentos para feature)
                'max_df': 1.0,  # Possibilidades: 0.7, 0.8, 0.9, 1.0 (máximo de documentos para feature)

                # Parâmetros específicos para TF-IDF:
                'use_idf': True,  # Possibilidades: True, False (usar IDF)
                'smooth_idf': True,  # Possibilidades: True, False (suavizar IDF)
                'sublinear_tf': False,  # Possibilidades: True, False (TF sublinear)
            }
        },

        # --- CONFIGURAÇÕES DO MODELO ---
        'model': {
            'name': 'logistic_regression', # Possibilidades: 'logistic_regression', 'mlp', 'random_forest', 'svm'
            # TODO verificar modelos após implementacao

            'params': {
                # --- PARÂMETROS GERAIS ---
                'random_state': 42,  # Para reprodutibilidade

                # --- LOGISTIC REGRESSION ---
                'penalty': 'l2',  # Possibilidades: 'l1', 'l2', 'elasticnet', 'none'
                'C': 1.0,  # Possibilidades: 0.001, 0.01, 0.1, 1.0, 10.0, 100.0
                'solver': 'lbfgs',  # Possibilidades: 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'
                'max_iter': 1000,  # Possibilidades: 100, 500, 1000, 2000
                'class_weight': None,  # Possibilidades: None, 'balanced'
                'n_jobs': -1,  # Possibilidades: -1 (todos cores), 1, 2, 4

                # --- MLP (Multi-Layer Perceptron) ---
                'hidden_layer_sizes': (100,),  # Possibilidades: (50,), (100,), (50, 30), (100, 50), (100, 50, 25)
                'activation': 'relu',  # Possibilidades: 'identity', 'logistic', 'tanh', 'relu'
                'alpha': 0.0001,  # Possibilidades: 0.0001, 0.001, 0.01, 0.1
                'learning_rate': 'constant',  # Possibilidades: 'constant', 'invscaling', 'adaptive'
                'learning_rate_init': 0.001,  # Possibilidades: 0.001, 0.01, 0.1

            }
        },

        # --- CONFIGURAÇÕES DE AVALIAÇÃO ---
        'evaluation': {
            'folds': 5,  # Possibilidades: 3, 5, 10 (número de folds na validação cruzada)
            'random_state': 42,  # Seed para reprodutibilidade
            'accuracy_threshold': 0.6,  # Possibilidades: 0.5, 0.6, 0.7, 0.8 (mínimo para salvar modelo)
            'save_model': True,  # Possibilidades: True, False
        },

        # --- CONFIGURAÇÕES DE GRID SEARCH ---
        'grid_search': {
            'enable': False,  # Possibilidades: True, False
            'scoring': 'accuracy',  # Possibilidades: 'accuracy', 'f1', 'precision', 'recall', 'roc_auc'
            'num_folds': 3,  # Possibilidades: 3, 5 (folds no Grid Search)
            'n_jobs': -1,  # Possibilidades: -1, 1, 2, 4 (paralelização)
            'verbose': 1,  # Possibilidades: 0, 1, 2, 3 (nível de log)
            'refit': True,  # Possibilidades: True, False (re-treinar com melhores parâmetros)

            # Grade de parâmetros para busca (depende do modelo)
            'param_grid': {
                # Exemplo para Logistic Regression:
                'C': [0.1, 1, 10],
                'solver': ['liblinear', 'saga'],
                'class_weight': [None, 'balanced']

            }
        }
    }
