from sklearn.linear_model import LogisticRegression


def train_logistic_regression(x_train,
                              y_train,
                              max_iter: int,
                              class_weight: str = "balanced",
                              multi_class: str = "multinomial",
                              n_jobs: int = 1,
                              random_state: int = 99
                              ) -> LogisticRegression:
    model = LogisticRegression(max_iter=max_iter, multi_class=multi_class, class_weight=class_weight, n_jobs=n_jobs, random_state=random_state)
    model.fit(x_train, y_train)
    return model
