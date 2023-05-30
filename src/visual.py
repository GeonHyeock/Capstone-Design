import pickle
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import DM
import statsmodels.api as sm
from itertools import combinations
from scipy.stats import probplot
from scipy.stats import shapiro


def EDA(data_path):
    data = pd.read_csv(data_path, index_col=0)
    data.dropna(inplace=True)  # 11398 삭제
    data.reset_index(drop=True, inplace=True)

    probplot(data.ReadingScore, plot=plt)
    plt.ylim(0, 100)
    plt.savefig(f"./images/EDA_Mathscore_qq_plot.png")
    plt.clf()

    sns.kdeplot(data.ReadingScore, fill=True)
    plt.savefig(f"./images/EDA_Mathscore_density_plot.png")
    plt.clf()

    sns.scatterplot(
        data=data, x="WritingScore", y="MathScore", hue="Gender", s=4, alpha=0.5
    )
    plt.savefig(f"./images/EDA_gender_MathScore.png")
    plt.clf()

    for col in data.drop(["ReadingScore", "WritingScore", "MathScore"], axis=1).columns:
        sns.boxplot(data=data, x=col, y="MathScore").set(title=f"{col}에 따른 수학점수")
        plt.savefig(f"./images/EDA_{col}.png")
        plt.clf()

    for a, b in combinations(["ReadingScore", "WritingScore", "MathScore"], 2):
        sns.regplot(x=a, y=b, data=data, scatter_kws={"s": 0.5, "alpha": 0.5})
        plt.savefig(f"./images/{a[0]},{b[0]}_scatter.png")
        plt.clf()


def classification_result(result):
    for col in set(result.keys()) & set(
        ["LDA", "QDA", "Logistic Classification", "Tree"]
    ):
        data = result[col]
        plot = sns.heatmap(
            data["metric"] / data["metric"].sum(axis=1),
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=[f"{i} 등급" for i in range(1, 10)],
            yticklabels=[f"{i} 등급" for i in range(1, 10)],
        )
        best_param = {
            k: v if isinstance(v, str) else f"{v:.2f}"
            for k, v in data["best_param"].items()
        }
        print(f"{col} : {best_param}")
        plot.set_title(f"{col}")
        plt.savefig(f"./images/{col}_confusion.png")
        plt.clf()
        print(f"{col}분류율 : {np.trace(data['metric']) / np.sum(data['metric']):.4f}")

        if col in ["QDA", "Logistic Classification"]:
            if col == "QDA":
                param = "param_reg_param"
                data_dict = {
                    "train_score": data["cv_result"]["mean_train_score"],
                    "test_score": data["cv_result"]["mean_test_score"],
                    f"{param}": data["cv_result"][param].data,
                }
                pd.DataFrame(data_dict).plot.line(x=f"{param}")
            else:
                idx = (
                    data["cv_result"]["param_penalty"] == data["best_param"]["penalty"]
                ) & (data["cv_result"]["param_solver"] == data["best_param"]["solver"])
                param = "param_C"
                data_dict = {
                    "train_score": data["cv_result"]["mean_train_score"][idx],
                    "test_score": data["cv_result"]["mean_test_score"][idx],
                    f"{param}": data["cv_result"][param].data[idx],
                }
                pd.DataFrame(data_dict).plot.line(x=f"{param}")
                plt.xlim(0, 1)
            plt.savefig(f"./images/{col}_{param}.png")
            plt.clf()


def regression_result(result):
    for col in set(result.keys()) & set(["LinearRegression", "Ridge", "LASSO", "PLS"]):
        data = result[col]
        _, b, c = data["metric"]
        print(f"{col} : {np.sqrt(((b-c)**2).mean()):.4f}")
        if col in ["Ridge", "LASSO", "PLS"]:
            print(f"{col} : {data['best_param']}")
            param = "param_n_components" if col == "PLS" else "param_alpha"
            data_dict = {
                "train_score": data["cv_result"]["mean_train_score"],
                "test_score": data["cv_result"]["mean_test_score"],
                f"{param}": data["cv_result"][param].data,
            }
            pd.DataFrame(data_dict).plot.line(x=f"{param}")
            plt.ylim(0.5, 1)
            plt.savefig(f"./images/{col}_{param}.png")
            plt.clf()


if __name__ == "__main__":
    plt.rcParams["font.family"] = "AppleGothic"
    my_dm = DM()
    with open("./result_no_expension.pickle", "rb") as fr:
        result = pickle.load(fr)

    EDA("data/Expanded_data_with_more_features.csv")
    classification_result(result)
    regression_result(result)
