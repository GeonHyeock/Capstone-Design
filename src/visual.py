import pickle
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def confusion_png(result):
    for col in list(result.keys())[:4]:
        data = result[col]["metric"]
        plot = sns.heatmap(data / data.sum(axis=1), annot=True, fmt=".2f", cmap="Blues")
        plot.set_title(col)
        plt.savefig(f"./images/{col}_confusion.png")
        plt.clf()


if __name__ == "__main__":
    with open("./result.pickle", "rb") as fr:
        result = pickle.load(fr)

    confusion_png(result)
