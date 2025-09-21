import joblib
from pathlib import Path
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.base import ClassifierMixin
from scipy.sparse import csr_matrix
from typing import Any

from src.data.load import load_data
from src.utils.logger import (print_debug, print_info, print_error, print_warning, print_cross_validation_results,
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

    def get_model_instance(self, params: dict = None) -> ClassifierMixin:
        model_name = self.config["model"]["name"]
        model_params = params or self.config["model"]["params"]

        if self.debug:
            print_debug(f"   🤖 Criando modelo {model_name} com parâmetros: {model_params}")

        if model_name == "logistic_regression":
            from src.models.logistic_regression import create_logistic_regression
            return create_logistic_regression(**model_params)

        elif model_name == "mlp":
            from src.models.mlp import create_mlp
            return create_mlp(**model_params)

        elif model_name == 'naive_bayes':
            from src.models.naive_bayes import create_naive_bayes
            return create_naive_bayes(**model_params)

        elif model_name == 'random_forest':
            from src.models.random_forest import create_random_forest
            return create_random_forest(**model_params)

        elif model_name == 'svm':
            from src.models.svm import create_svm
            return create_svm(**model_params)

        else:
            raise ValueError(f"Modelo não suportado: {model_name}")

    def run_grid_search(self, x, y) -> tuple[ClassifierMixin, dict]:
        if "grid_search" not in self.config or not self.config["grid_search"]["enable"]:
            print_info("Grid search desabilitado")
            return self.get_model_instance(), {}

        print_info("Executando grid search")

        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        grid_config = self.config["grid_search"]
        model = self.get_model_instance()

        if self.config['model']['name'] == 'mlp' and hasattr(x, 'toarray'):
            print_info("Convertendo dados para denso (Grid Search MLP)")
            x_processed = x.toarray()
        else:
            x_processed = x

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
        grid_search.fit(x_processed, y_encoded)

        print_info("Grid Search concluído!")
        print_info(f"Melhores parâmetros: {grid_search.best_params_}")
        print_info(f"Melhor score: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_, grid_search.best_params_

    def evaluate(self, x, y) -> dict[str, Any]:
        from src.evaluation.cross_validation import evaluate_cross_validation

        print_info("Executando validação cruzada...")

        if self.best_params is not None:
            print_info("Usando melhores parâmetros do grid search")
            model = self.get_model_instance(self.best_params)
        else:
            model = self.get_model_instance()

        if self.config['model']['name'] == 'mlp' and hasattr(x, 'toarray'):
            print_info("Convertendo dados para denso (Validação MLP)")
            x_processed = x.toarray()
        else:
            x_processed = x

        results = evaluate_cross_validation(
            model,
            x_processed,
            y,
            num_folds=self.config['evaluation']['folds'],
            random_state=self.config['evaluation']['random_state']
        )

        return results

    def train_final_model(self, x, y):
        print_info("Treinando modelo final...")

        if self.best_params is not None:
            print_info("Usando melhores parâmetros do grid search")
            self.model = self.get_model_instance(self.best_params)
        else:
            self.model = self.get_model_instance()

        if (self.config['model']['name'] == 'mlp' and
                hasattr(x, 'toarray') and
                hasattr(self.model, 'early_stopping') and
                self.model.early_stopping):

            print_info("Convertendo dados para denso (MLP com early stopping)")
            x_dense = x.toarray()
            self.model.fit(x_dense, y)

        else:
            self.model.fit(x, y)

        self._print_model_specific_info()

        train_accuracy = self.model.score(x, y)
        print_info(f"Acurácia no treino completo: {train_accuracy:.3f}")

        return self.model

    def save_model(self):
        if self.model is None:
            print_warning("Nada para salvar - modelo não disponível")
            return

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = \
            f"{self.config['dataset']}_{self.config['features']['method']}_{self.config['model']['name']}_{timestamp}"

        model_dir = Path("models/saved") / base_name
        model_dir.mkdir(exist_ok=True, parents=True)

        model_path = model_dir / "model.pkl"
        joblib.dump(self.model, model_path)

        config_path = model_dir / "config.json"
        import json
        with open(config_path, "w", encoding="utf-8") as f:
            serializable_config = self._make_config_serializable()
            json.dump(serializable_config, f, indent=2, ensure_ascii=False)

        results_path = model_dir / "results.json"
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
            vectorizer_path = model_dir / "vectorizer.pkl"
            joblib.dump(self.vectorizer, vectorizer_path)
            if self.debug:
                print_debug(f"Vectorizer salvo: {vectorizer_path}")

        if self.best_params:
            best_params_path = model_dir / "best_params.json"
            with open(best_params_path, "w", encoding="utf-8") as f:
                json.dump(self.best_params, f, indent=2, ensure_ascii=False)

        print_info(f" Modelo salvos com timestamp: {timestamp}")
        print_info(f" Modelo: {model_path}")
        print_info(f" Configuração: {config_path}")
        print_info(f" Resultados: {results_path}")

        return model_path

    def _print_model_specific_info(self):
        model_name = self.config['model']['name']
        try:
            if model_name == 'random_forest' and hasattr(self, 'vectorizer'):
                from src.models.random_forest import print_random_forest_info
                feature_names = self.vectorizer.get_feature_names_out() if self.vectorizer else None
                print_random_forest_info(self.model, feature_names)
            elif model_name == 'naive_bayes' and hasattr(self, 'vectorizer'):
                from src.models.naive_bayes import print_naive_bayes_info
                feature_names = self.vectorizer.get_feature_names_out() if self.vectorizer else None
                print_naive_bayes_info(self.model, feature_names)

            elif model_name == 'mlp':
                from src.models.mlp import print_mlp_info
                print_mlp_info(self.model)

            elif model_name == 'svm':
                from src.models.svm import print_svm_info
                print_svm_info(self.model)

        except Exception as e:
            if self.debug:
                print_warning(f"Could not print model info: {e}")

    def _make_config_serializable(self):
        import numpy
        from pathlib import Path

        def convert_value(value):
            if isinstance(value, (numpy.integer, numpy.floating)):
                return float(value)
            elif isinstance(value, (numpy.ndarray, list, tuple)):
                return [convert_value(v) for v in value]  # Chamada recursiva
            elif isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}  # Chamada recursiva
            elif isinstance(value, (Path, set)):
                return str(value)
            else:
                return value

        return convert_value(self.config)

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

            if self.config['grid_search']['enable']:
                print_info("Executando grid search para parâmetros ótimos...")
                self.model, self.best_params = self.run_grid_search(x, labels)

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
            'params': {'ngrams': (1, 3)}
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


def teste():
    debug_mode = True
    set_global_debug_mode(debug_mode)

    # Lista de datasets para testar
    datasets = ['arcaico_moderno', 'complexo_simples', 'literal_dinamico']

    # 🔧 CONFIGURAÇÃO 1: Logistic Regression (Baseline)
    config_lr = {
        'dataset': None,  # Será preenchido para cada dataset
        'features': {
            'method': 'bag_of_words',
            'params': {'ngrams': (1, 2), 'max_features': 5000}
        },
        'model': {
            'name': 'logistic_regression',
            'params': {
                'max_iter': 1000,
                'class_weight': 'balanced',
                'random_state': 42,
                'solver': 'liblinear'
            }
        },
        'evaluation': {
            'folds': 5,
            'random_state': 42,
            'accuracy_threshold': 0.6,
            'save_model': True
        },
        'grid_search': {
            'enable': False  # Sem grid search para baseline
        }
    }

    # 🔧 CONFIGURAÇÃO 2: Naive Bayes (Simples + Grid Search)
    config_nb = {
        'dataset': None,
        'features': {
            'method': 'bag_of_words',
            'params': {'ngrams': (1, 3), 'max_features': 8000}
        },
        'model': {
            'name': 'naive_bayes',
            'params': {
                'alpha': 1.0,
                'fit_prior': True
            }
        },
        'evaluation': {
            'folds': 5,
            'random_state': 42,
            'accuracy_threshold': 0.55,  # Mais baixo para Naive Bayes
            'save_model': True
        },
        'grid_search': {
            'enable': True,
            'scoring': 'f1',
            'num_folds': 3,
            'n_jobs': -1,
            'verbose': 1,
            'param_grid': {
                'alpha': [0.1, 0.5, 1.0, 2.0, 5.0],
                'fit_prior': [True, False]
            }
        }
    }

    # 🔧 CONFIGURAÇÃO 3: Random Forest (Complexo + Grid Search)
    config_rf = {
        'dataset': None,
        'features': {
            'method': 'bag_of_words',
            'params': {'ngrams': (1, 2), 'max_features': 10000, 'min_df': 2}
        },
        'model': {
            'name': 'random_forest',
            'params': {
                'n_estimators': 100,
                'max_depth': None,
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1
            }
        },
        'evaluation': {
            'folds': 5,
            'random_state': 42,
            'accuracy_threshold': 0.65,
            'save_model': True
        },
        'grid_search': {
            'enable': True,
            'scoring': 'accuracy',
            'num_folds': 3,
            'n_jobs': -1,
            'verbose': 1,
            'param_grid': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        }
    }

    # 🔧 CONFIGURAÇÃO 4: MLP (Redes Neurais + Grid Search)
    config_mlp = {
        'dataset': None,
        'features': {
            'method': 'bag_of_words',
            'params': {'ngrams': (1, 2), 'max_features': 8000, 'max_df': 0.9}
        },
        'model': {
            'name': 'mlp',
            'params': {
                'hidden_layer_sizes': (100,),
                'activation': 'relu',
                'solver': 'adam',
                'max_iter': 1000,
                'random_state': 42,
                'early_stopping': True
            }
        },
        'evaluation': {
            'folds': 5,
            'random_state': 42,
            'accuracy_threshold': 0.63,
            'save_model': True
        },
        'grid_search': {
            'enable': True,
            'scoring': 'accuracy',
            'num_folds': 3,
            'n_jobs': 1,  # MLP não paraleliza bem
            'verbose': 1,
            'param_grid': {
                'hidden_layer_sizes': [(50,), (100,), (50, 30)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.0005]
            }
        }
    }

    # 🔧 CONFIGURAÇÃO 5: SVM (Kernel Linear + Grid Search)
    config_svm = {
        'dataset': None,
        'features': {
            'method': 'bag_of_words',
            'params': {'ngrams': (1, 2), 'max_features': 12000, 'min_df': 3}
        },
        'model': {
            'name': 'svm',
            'params': {
                'C': 1.0,
                'kernel': 'linear',
                'class_weight': 'balanced',
                'random_state': 42,
                'probability': True
            }
        },
        'evaluation': {
            'folds': 5,
            'random_state': 42,
            'accuracy_threshold': 0.67,  # SVM geralmente tem boa acurácia
            'save_model': True
        },
        'grid_search': {
            'enable': True,
            'scoring': 'accuracy',
            'num_folds': 3,
            'n_jobs': -1,
            'verbose': 1,
            'param_grid': {
                'C': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        }
    }

    # 📊 LISTA DE TODAS AS CONFIGURAÇÕES
    all_configs = [
        ('Logistic_Regression', config_lr),
        ('Naive_Bayes', config_nb),
        ('Random_Forest', config_rf),
        ('MLP', config_mlp),
        ('SVM', config_svm)
    ]

    # 🚀 EXECUTAR TODOS OS EXPERIMENTOS
    results_summary = {}

    for config_name, config in all_configs:
        print(f"\n{'🚀' * 3} INICIANDO {config_name.upper()} {'🚀' * 3}")

        dataset_results = {}
        for dataset in datasets:
            print(f"\n{'📊' * 2} DATASET: {dataset.upper()} {'📊' * 2}")

            # Configurar para este dataset
            current_config = config.copy()
            current_config['dataset'] = dataset

            try:
                pipeline = BibleStylePipeline(current_config, debug=debug_mode)
                results = pipeline.run()

                dataset_results[dataset] = {
                    'mean_accuracy': results['mean_accuracy'],
                    'mean_f1': results['mean_f1'],
                    'best_params': getattr(pipeline, 'best_params', None)
                }

                print_info(f"✅ {config_name} - {dataset}: {results['mean_accuracy']:.3f}")

            except Exception as e:
                print_error(f"❌ Erro em {config_name} - {dataset}: {e}")
                dataset_results[dataset] = {'error': str(e)}

        results_summary[config_name] = dataset_results

    # 📈 IMPRIMIR RELATÓRIO FINAL
    print(f"\n{'🎯' * 5} RELATÓRIO FINAL {'🎯' * 5}")
    print("=" * 80)

    for config_name, dataset_results in results_summary.items():
        print(f"\n📊 {config_name.upper()}:")
        for dataset, results in dataset_results.items():
            if 'error' in results:
                print(f"   {dataset}: ❌ {results['error']}")
            else:
                print(f"   {dataset}: ✅ Accuracy={results['mean_accuracy']:.3f}, F1={results['mean_f1']:.3f}")


if __name__ == "__main__":
    teste()


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
            'name': 'logistic_regression',  # Possibilidades: 'logistic_regression', 'mlp', 'random_forest', 'svm'
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
