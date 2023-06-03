from model import DM
from sklearn.metrics import confusion_matrix
import numpy as np
import pickle
import pandas as pd
import time
from tqdm import tqdm
import statsmodels.api as sm
from scipy import stats
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.linalg import pinv
import graphviz
from sklearn import tree
import pydot


def get_statistics(model, intercept=0, coef=[], train_df=None, target_df=None):
    params = np.append(intercept, coef)
    prediction = model.predict(train_df.values)

    if len(prediction.shape) == 1:
        prediction = np.expand_dims(prediction, axis=1)
    print(train_df.columns)

    new_trainset = pd.DataFrame({"Constant": np.ones(len(train_df.values))}).join(
        pd.DataFrame(train_df.values)
    )

    new_trainset[0] = new_trainset[0].astype("float")
    o_col = [c for c in new_trainset.columns if new_trainset[c].dtypes == "O"]
    for c in o_col:
        new_trainset[c] = new_trainset[c].apply(lambda x: 1.0 if x == True else 0.0)

    MSE = mean_squared_error(prediction, target_df.values)

    # Calculate p-value
    residuals = target_df.to_numpy() - predict
    residual_sum_of_squares = np.sum(residuals**2)
    variance = np.diagonal(pinv(np.dot(new_trainset.T, new_trainset)))

    std_error = np.sqrt(variance)
    t_values = params / std_error
    p_values = [
        2
        * (
            1
            - stats.t.cdf(
                np.abs(i), (len(new_trainset) - len(new_trainset.columns) - 1)
            )
        )
        for i in t_values
    ]

    std_error = np.round(std_error, 4)
    t_values = np.round(t_values, 4)
    p_values = np.round(p_values, 4)
    params = np.round(params, 4)

    statistics = pd.DataFrame()
    (
        statistics["Coefficients"],
        statistics["Standard Errors"],
        statistics["t-values"],
        statistics["p-values"],
    ) = [params, std_error, t_values, p_values]

    return statistics


if __name__ == "__main__":
    with open("./result.pickle", "rb") as fr:
        result = pickle.load(fr)

    m = [result[k]["model"] for k in result.keys()]
    for k in ["LinearRegression", "Ridge", "LASSO", "PLS"]:
        lm = result[k]["model"]
        X_test, target, predict = result[k]["metric"]
        statistics = get_statistics(
            lm,
            intercept=lm.intercept_,
            coef=lm.coef_,
            train_df=X_test,
            target_df=pd.DataFrame(target),
        )
        print("-" * 80)
        print(k)
        print(statistics)

        # statistics.to_csv(f"{k}.csv")
