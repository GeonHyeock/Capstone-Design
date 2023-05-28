from model import DM
from sklearn.metrics import confusion_matrix
import numpy as np
import pickle
import time
from tqdm import tqdm


def metric(model, type):
    assert type in ["classification", "regression"], "type을 확인해 주세요."
    X_test = dm.test_data.drop(["MathGrade", "MathScore"], axis=1)
    predict = model.predict(X_test)
    if type == "classification":
        target = dm.test_data["MathGrade"]
        return confusion_matrix(target, predict)
    elif type == "regression":
        target = dm.test_data["MathScore"]
        if isinstance(predict, np.ndarray) and len(predict.shape) > 1:
            predict = predict.squeeze()
        return X_test, target, predict


if __name__ == "__main__":
    dm, result = DM(), {}

    start = time.time()

    for k, v in tqdm(dm.models.items()):
        print(k)
        if k not in ["Logistic Regression", "LinearRegression", "Ridge", "LASSO"]:
            continue
        kfold = dm.kfold_grid_serch(**v)
        my_metric = metric(kfold["model"], v["type"])
        result[k] = {**kfold, "metric": my_metric}

    end = time.time()

    print(f"{end - start:.5f} sec")

    with open("result_reg.pickle", "wb") as data:
        pickle.dump(result, data)
