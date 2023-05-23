import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression

from sklearn.feature_selection import SelectFromModel


class DM:
    def __init__(self, data_path="data/Expanded_data_with_more_features.csv"):
        self.train_data, self.test_data = preprocessing(data_path)
        self.num_features = len(self.train_data.columns) - 1

        self.models = {
            "LDA": {
                "model": LinearDiscriminantAnalysis(),
                "param": {},
                "type": "classification",
            },
            "QDA": {
                "model": QuadraticDiscriminantAnalysis(),
                "param": {"reg_param": [0.2 * i for i in range(5)]},
                "type": "classification",
            },
            "Logistic Regression": {
                "model": LogisticRegression(),
                "param": {},
                "type": "classification",
            },
            "Tree": {
                "model": DecisionTreeClassifier(),
                "param": {},
                "type": "classification",
            },
            "LinearRegression": {
                "model": LinearRegression(),
                "param": {},
                "type": "regression",
            },
            "Ridge": {
                "model": Ridge(),
                "param": {"alpha": [1e-8, 0.0001, 0.001, 0.01, 0.1, 1, 10]},
                "type": "regression",
            },
            "LASSO": {
                "model": Lasso(),
                "param": {"alpha": [1e-8, 0.0001, 0.001, 0.01, 0.1, 1, 10]},
                "type": "regression",
            },
            "ElasticNet": {
                "model": ElasticNet(),
                "param": {
                    "alpha": [1e-8, 0.0001, 0.001, 0.01, 0.1, 1, 10],
                    "l1_ratio": [0.3, 0.5, 0.7],
                },
                "type": "regression",
            },
            "PLS": {
                "model": PLSRegression(),
                "param": {"n_components": list(range(1, self.num_features))},
                "type": "regression",
            },
            "PCR": {
                "model": Pipeline(
                    steps=[("pca", PCA()), ("regression", LinearRegression())]
                ),
                "param": {
                    "pca__n_components": [None] + list(range(1, self.num_features))
                },
                "type": "regression",
            },
        }

    def kfold_grid_serch(self, model, param, type="classification"):
        target = "MathGrade" if type == "classification" else "MathScore"
        if param:
            grid = GridSearchCV(model, param, cv=5, return_train_score=True)
            grid.fit(
                self.train_data.drop(["MathGrade", "MathScore"], axis=1),
                self.train_data[target],
            )
            return {
                "model": grid.best_estimator_,
                "best_param": grid.best_params_,
                "cv_result": grid.cv_results_,
            }
        else:
            model.fit(
                self.train_data.drop(["MathGrade", "MathScore"], axis=1),
                self.train_data[target],
            )
            return {"model": model}

    def best_subset_selection(self, model, type):
        target = "MathGrade" if type == "classification" else "MathScore"

        features_name = [
            col for col in self.train.columns if col not in ["MathGrade", "MathScore"]
        ]

        X_train = self.train[features_name].to_numpy()
        y_train = self.train[target].to_numpy()

        selector = SelectFromModel(estimator=model, threshold="median").fit(
            X_train, y_train
        )

        try:
            selected_features = list(np.array(features_name)[selector.get_support()])
        except ValueError as e:
            print(f"{model} 해당 모델은 개별 특성의 중요도를 직접 추정할 수 없습니다.")
            return None

        n_features = len(selected_features)

        print(f"{n_features} features are selected.")
        print(f"Selected features : {selected_features}")

        return n_features, selected_features


def preprocessing(data_path):
    """
    1. 결측치 제거

    2. 변수 변환
     a) Gender, EthnicGroup, ParentEduc, LunchType, TestPrep, ParentMaritalStatus,
        PracticeSport, WklyStudyHours : 더미화
     b) IsFirstChild, NrSiblings, TransportMeans 변수 제거 (by ttest)
     c) WritingScore 변수 제거 (다중 공선성)
     d) 수치형 데이터 정규화

    3. train, test 분할 (8:2)
    Returns:
        data_path: 데이터 주소
    """

    def dummy(Data, col):
        Data = pd.concat([Data, pd.get_dummies(data[col], prefix=col[:3])], axis=1)
        Data.drop(col, axis=1, inplace=True)
        return Data

    data = pd.read_csv(data_path, index_col=0)
    data.dropna(inplace=True)  # 11398 삭제
    data.reset_index(drop=True, inplace=True)

    data["MathGrade"] = pd.cut(
        data.MathScore,
        data.MathScore.quantile([0, 0.04, 0.11, 0.23, 0.40, 0.60, 0.88, 0.89, 0.96, 1]),
        labels=[f"{i}" for i in range(9, 0, -1)],
    )

    data["MathGrade"] = data["MathGrade"].fillna("9")

    for col in [
        "Gender",
        "EthnicGroup",
        "ParentEduc",
        "LunchType",
        "TestPrep",
        "ParentMaritalStatus",
        "PracticeSport",
        "WklyStudyHours",
    ]:
        data = dummy(data, col)

    data.drop(
        ["WritingScore", "IsFirstChild", "NrSiblings", "TransportMeans"],
        axis=1,
        inplace=True,
    )

    train_data = data.sample(frac=0.8, random_state=42)
    train_numeric = train_data.loc[:, ["ReadingScore"]]
    mean = train_numeric.mean()
    std = train_numeric.std(ddof=1)
    train_data.loc[:, ["ReadingScore"]] = (train_numeric - mean) / std

    test_data = data.drop(train_data.index)
    test_numeric = test_data.loc[:, ["ReadingScore"]]
    test_data.loc[:, ["ReadingScore"]] = (test_numeric - mean) / std
    return train_data, test_data


if __name__ == "__main__":
    A = DM()
    # A.kfold_grid_serch(**A.models["LDA"])
    model = QuadraticDiscriminantAnalysis()
    data_path = "data/Expanded_data_with_more_features.csv"
    A.best_subset_selection(model, data_path, type="classification")
