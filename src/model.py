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
from sklearn.svm import SVC

from sklearn.feature_selection import SelectFromModel

import time
import warnings

warnings.filterwarnings(action="ignore")


class DM:
    def __init__(self, data_path="data/Expanded_data_with_more_features.csv"):
        # variables이 서로 선형관계를 띄고 있어서 UserWarning:Variables are collinear발생
        self.train_data, self.test_data = preprocessing(data_path, "classification")
        self.num_features = len(self.train_data.columns) - 1

        self.models = {
            "LDA": {
                "model": LinearDiscriminantAnalysis(),
                "param": {"solver": ["svd", "lsqr", "eigen"]},
                "type": "classification",
            },
            "QDA": {
                "model": QuadraticDiscriminantAnalysis(),
                "param": {"reg_param": list(np.linspace(0.0, 1.0, 11, endpoint=True))},
                "type": "classification",
            },
            "Logistic Regression": {
                "model": LogisticRegression(),
                "param": {
                    "penalty": ["l1", "l2", "elasticnet", "none"],
                    "C": list(np.logspace(-4, 4, 20)),
                    "solver": ["lbfgs", "newton-cg", "liblinear", "sag", "saga"],
                },
                "type": "classification",
            },
            "Tree": {
                "model": DecisionTreeClassifier(),
                "param": {
                    "max_depth": [3, 5, 7, 10, 15],
                    "min_samples_leaf": [3, 5, 10, 15, 20],
                    "min_samples_split": [8, 10, 12, 18, 20, 16],
                    "criterion": ["gini", "entropy"],
                },
                "type": "classification",
            },
            "SVC": {
                "model": SVC(),
                "param": {
                    "C": [0.1, 1, 10, 100],
                    "gamma": [1, 0.1, 0.01, 0.001],
                    "kernel": ["rbf", "poly", "sigmoid"],
                },
                "type": "classification",
            },
            "LinearRegression": {
                "model": LinearRegression(),
                "param": {},
                "type": "regression",
            },
            "Ridge": {
                "model": Ridge(),
                "param": {"alpha": list(np.arange(0, 1, 0.01))},
                "type": "regression",
            },
            "LASSO": {
                "model": Lasso(),
                "param": {"alpha": list(np.arange(0, 1, 0.01))},
                "type": "regression",
            },
            "ElasticNet": {
                "model": ElasticNet(),
                "param": {
                    "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0],
                    "l1_ratio": list(np.arange(0, 1, 0.01)),
                },
                "type": "regression",
            },
            "PLS": {
                "model": PLSRegression(),
                "param": {"n_components": list(range(1, 10))},
                "type": "regression",
            },
            "PCR": {
                "model": Pipeline(
                    steps=[("pca", PCA()), ("regression", LinearRegression())]
                ),
                "param": {"n_components": list(range(1, 10))},
                "type": "regression",
            },
        }

    def kfold_grid_serch(self, model, param, type="classification"):
        target = "MathGrade" if type == "classification" else "MathScore"
        if param:
            # max_iter가 충분하지 않아 결과가 수렴하지 않아서 ConvergenceWarning 발생
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


def best_subset_selection(model, data_path, type):
    # data
    train, test = preprocessing(data_path, "classification")
    target = "MathGrade" if type == "classification" else "MathScore"

    features_name = [
        col for col in train.columns if col not in ["MathGrade", "MathScore"]
    ]

    X_train = train[features_name].to_numpy()
    y_train = train[target].to_numpy()

    selector = SelectFromModel(estimator=model, threshold="median").fit(
        X_train, y_train
    )

    try:
        selected_features = list(np.array(features_name)[selector.get_support()])
    except ValueError as e:
        print("해당 모델은 개별 특성의 중요도를 직접 추정할 수 없습니다.")
        return None

    n_features = len(selected_features)

    print(f"{n_features} features are selected.")
    print(f"Selected features : {selected_features}")

    return n_features, selected_features


def preprocessing(data_path, data_mode):
    """
    1. 결측치 제거

    2. 변수 변환
     a) Gender : male : 0, female : 1
     b) EthnicGroup : 더미화 5그룹
     c) ParentEduc : 더미화 6그룹
     d) LunchType : standard : 0, free/reduced : 1
     e) TestPrep : none : 0, completed : 1
     f) ParentMaritalStatus : 더미화 4그룹
     g) PracticeSport : 더미화 3그룹
     h) IsFirstChild : yes : 0, no = 1
     i) TransportMeans: school_bus : 0, private : 1
     j) WklyStudyHours : 더미화 3그룹
     k) 수치형 데이터 정규화

    3. train, test 분할 (8:2)
    Returns:
        data_path: 데이터 주소
    """
    data = pd.read_csv(data_path, index_col=0)
    data.dropna(inplace=True)  # 11398 삭제
    data.reset_index(drop=True, inplace=True)

    if data_mode == "classification":
        data["MathGrade"] = pd.cut(
            data.MathScore,
            data.MathScore.quantile(
                [0, 0.04, 0.11, 0.23, 0.40, 0.60, 0.88, 0.89, 0.96, 1]
            ),
            labels=[f"{i}" for i in range(9, 0, -1)],
        )
        data["MathGrade"] = data["MathGrade"].fillna("9")

    data.Gender = np.where(data.Gender == "male", 0.0, 1)

    data = pd.concat([data, pd.get_dummies(data.EthnicGroup, prefix="Ethnic")], axis=1)

    data = pd.concat([data, pd.get_dummies(data.ParentEduc, prefix="Educ")], axis=1)

    data.LunchType = np.where(data.LunchType == "standard", 0.0, 1)

    data.TestPrep = np.where(data.TestPrep == "none", 0.0, 1)

    data = pd.concat(
        [data, pd.get_dummies(data.ParentMaritalStatus, prefix="PM")], axis=1
    )

    data = pd.concat([data, pd.get_dummies(data.PracticeSport, prefix="sport")], axis=1)

    data.IsFirstChild = np.where(data.IsFirstChild == "yes", 0.0, 1)

    data.TransportMeans = np.where(data.TransportMeans == "school_bus", 0.0, 1)

    data = pd.concat(
        [data, pd.get_dummies(data.WklyStudyHours, prefix="study")], axis=1
    )

    data.drop(
        [
            "EthnicGroup",
            "ParentEduc",
            "ParentMaritalStatus",
            "PracticeSport",
            "WklyStudyHours",
        ],
        axis=1,
        inplace=True,
    )

    train_data = data.sample(frac=0.8, random_state=42)
    train_numeric = train_data.loc[:, ["ReadingScore", "WritingScore"]]
    mean = train_numeric.mean()
    std = train_numeric.std(ddof=1)
    train_data.loc[:, ["ReadingScore", "WritingScore"]] = (train_numeric - mean) / std

    test_data = data.drop(train_data.index)
    test_numeric = test_data.loc[:, ["ReadingScore", "WritingScore"]]
    test_data.loc[:, ["ReadingScore", "WritingScore"]] = (test_numeric - mean) / std
    return train_data, test_data


if __name__ == "__main__":
    A = DM()
    d = A.kfold_grid_serch(**A.models["LDA"])
    start = time.time()
    for k in A.models.keys():
        A.kfold_grid_serch(**A.models[k])
    end = time.time()

    print(f"{end - start:.5f} sec")
    # model = QuadraticDiscriminantAnalysis()
    # data_path = "data/Expanded_data_with_more_features.csv"
    # best_subset_selection(model, data_path, type="classification")
