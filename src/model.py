import pandas as pd


class Regression:
    ## TODO : 선형회귀
    ## TODO : Ridge
    ## TODO : Lasso
    ## TODO : PCR
    ## TODO : PLS
    ## TODO : subset selection
    pass


class Classification:
    ## TODO : LDA & QDA & RDA
    ## TODO : Logistic Regression
    ## TODO : SVM
    ## TODO : TREE
    pass


def preprocessing(data_path="../data/Expanded_data_with_more_features.csv"):
    """결측치 제거

    Returns:
        data_path: 데이터 주소
    """
    data = pd.read_csv(data_path, index_col=0)
    data.dropna(inplace=True)  # 11398 삭제
    data.reset_index(drop=True, inplace=True)
    return data


if __name__ == "__main__":
    pass
