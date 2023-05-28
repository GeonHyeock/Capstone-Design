from model import DM
from sklearn.metrics import confusion_matrix
import numpy as np
import pickle


def metric(model, type):
    assert type in ["classification", "regression"], "type을 확인해 주세요."
    predict = model.predict(dm.test_data.drop(["MathGrade", "MathScore"], axis=1))
    if type == "classification":
        target = dm.test_data["MathGrade"]
        return confusion_matrix(target, predict)

    elif type == "regression":
        target = dm.test_data["MathScore"]
        if isinstance(predict, np.ndarray) and len(predict.shape) > 1:
            predict = predict.squeeze()
        return np.sqrt(((target - predict) ** 2).mean())


if __name__ == "__main__":
    dm, result = DM(), {}
    for idx, value in enumerate(dm.models.items()):
        print(f"{idx}/{len(dm.models.items())}")
        k, v = value
        kfold = dm.kfold_grid_serch(**v)
        my_metric = metric(kfold["model"], v["type"])
        result[k] = {**kfold, "metric": my_metric}

    with open("result4.pickle", "wb") as data:
        pickle.dump(result, data)
