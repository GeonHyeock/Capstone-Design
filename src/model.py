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
        self.train_data, self.test_data = preprocessing(data_path, "classification")
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


def best_subset_selection(model, X, y):
    selector = SelectFromModel(estimator=model, threshold="median")
    n_features = selector.transform(X)
    selected_features = list(X.columns[selector.get_support()])
    print(f"{n_features}개의 feature가 선택됨")


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
    A.kfold_grid_serch(**A.models["LDA"])
