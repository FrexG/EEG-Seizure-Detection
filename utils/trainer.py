import json
import joblib
from joblib import Parallel, delayed
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    log_loss,
    f1_score,
    precision_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from custom_loader import XGBLoader


def get_data(dataset: list, window: int = 1):
    x = []
    y = []
    x_19 = []
    y_19 = []

    bar = tqdm(dataset, total=len(dataset), desc="Generating Matrix")

    for i, eeg_annot in enumerate(bar):
        montage = eeg_annot["montage"]
        dg = XGBLoader(eeg_annot, window_size=window)
        j = 0
        for _x, _y in dg:
            if _y.sum() == 0:
                continue
            if np.isnan(_x).sum() == 0:
                if montage not in ["01_tcp_ar", "02_tcp_le"]:
                    x_19.append(_x)  # _x[[3, 5, 12, 13], :])
                    y_19.append(_y)  # _y[[3, 5, 12, 13]])
                    continue
                x.append(_x)  # _x[[3, 5, 14, 15], :])
                y.append(_y)  # _y[[3, 5, 14, 15]])

    x_np = np.array(x)
    y_np = np.array(y)

    x_np_19 = np.array(x_19)
    y_np_19 = np.array(y_19)

    x_np = x_np.reshape(-1, x_np.shape[-1])
    y_np = y_np.reshape(-1)

    x_np_19 = x_np_19.reshape(-1, x_np_19.shape[-1])
    y_np_19 = y_np_19.reshape(-1)

    x_np = np.concatenate((x_np, x_np_19), axis=0)
    y_np = np.concatenate((y_np, y_np_19), axis=0)

    return x_np, y_np


def train_xgb(lr=0.01, max_depth=6, n_estimator=200):
    """Train an XGboost model on the gpu

    Args:
        dataset (list): a list containing annotations
        lr (_type_, optional): _description_. Defaults to 1e-3.
        max_depth (int, optional): _description_. Defaults to 10.
    """
    # Set the tree_method parameter to gpu_hist
    params = {"tree_method": "gpu_hist"}
    # Set the gpu_id parameter to the ID of your GPU
    params["gpu_id"] = 0
    params["predictor"] = "gpu_predictor"

    xgb_clf = xgb.XGBClassifier(
        booster="gbtree",
        tree_method="gpu_hist",
        n_estimators=n_estimator,
        eta=lr,
        gamma=0.015,
        max_depth=max_depth,
        eval_metric="logloss",
        objective="binary:logistic",
    )

    # start training
    print(f"Training on {len(x_train_np)} samples")
    xgb_clf.fit(x_train_np, y_train_np, verbose=True)
    # save the model
    print("Saving model")
    xgb_clf.save_model(f"xgb_model_n_{n_estimator}_lr_{lr}.json")

    print(f"Score on training set: ")
    preds = xgb_clf.predict(x_train_np)
    scores = score_xg(y_train_np, preds)
    print(
        f"\tAcc = {scores[0]}\n\tRecall = {scores[1]}\n\tF1 = {scores[2]}\n\tPrecision = {scores[3]}"
    )

    print(f"Score on testing set: ")
    preds = xgb_clf.predict(x_test_np)
    scores = score_xg(y_test_np, preds)
    print(
        f"\tAcc = {scores[0]}\n\tRecall = {scores[1]}\n\tF1 = {scores[2]}\n\tPrecision = {scores[3]}"
    )

    """ 
    test_scores_dict = dict(zip(("acc", "recall", "f1"), scores))

    with open("test_scores.json", "w") as f:
        json.dump(test_scores_dict, f) """


def score_xg(y, preds):
    acc = accuracy_score(y, preds)
    recall = recall_score(y, preds, average="binary")
    f1 = f1_score(y, preds, average="binary")
    precision = precision_score(y, preds, average="binary")

    return (acc, recall, f1, precision)


def train_rf(n_estimators: int = 1000):
    clf = RandomForestClassifier(n_estimators, n_jobs=-1, max_depth=15, verbose=1)
    clf.fit(x_train_np, y_train_np)

    preds = clf.predict(x_train_np)
    scores = score_xg(y_train_np, preds)
    print(
        f"\tAcc = {scores[0]}\n\tRecall = {scores[1]}\n\tF1 = {scores[2]}\n\tPrecision = {scores[3]}"
    )
    preds = clf.predict(x_test_np)
    scores = score_xg(y_test_np, preds)
    print(
        f"\tAcc = {scores[0]}\n\tRecall = {scores[1]}\n\tF1 = {scores[2]}\n\tPrecision = {scores[3]}"
    )
    # save the trained RF model
    joblib.dump(clf, f"random_forest_est_{n_estimators}.joblib")


if __name__ == "__main__":
    training_data_path = "../bipolar_eeg_dataset/train_filtred_channel_based.json"
    testing_data_path = "../bipolar_eeg_dataset/dev_filtred_channel_based.json"

    with open(training_data_path, "r") as f:
        training_data = json.load(f)

    with open(testing_data_path, "r") as f:
        testing_data = json.load(f)

    """ Global variables """

    print("Generating Training Data")
    x_train_np, y_train_np = get_data(training_data, window=1)
    print(x_train_np.shape, y_train_np.shape)

    print("Generating Testing Data")
    x_test_np, y_test_np = get_data(testing_data, window=1)
    print(x_test_np.shape, y_test_np.shape)

    # train xgboost
    print("#" * 50)
    print("Training XGBoost")
    train_xgb(lr=0.001, max_depth=10, n_estimator=1000)
    print("#" * 50)
    print("Training Random forest")
    train_rf(1000)
    print("#" * 50)
