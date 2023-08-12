import json
import numpy as np
import torch
import xgboost as xgb
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, log_loss, f1_score
from sklearn.preprocessing import LabelEncoder
from custom_loader import XGBLoader


le = LabelEncoder()
labels = [0.0, 1.0]
le.fit_transform(labels)


def train_xgb(dataset: list, test_dataset: list, lr=3e-4, max_depth=10, window=20):
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
        tree_method="gpu_hist",
        learning_rate=lr,
        nthread=16,
        max_depth=max_depth,
        eval_metric="logloss",
        objective="binary:logistic",
    )
    x_train = []
    y_train = []

    bar = tqdm(dataset, total=len(dataset), desc="Generating Matrix")

    for i, eeg_annot in enumerate(bar):
        dg = XGBLoader(eeg_annot, window_size=window)
        j = 0
        for x, y in dg:
            if y.sum() == 0:
                continue

            x_train.append(x)
            y_train.append(y)

    # Evaluate when training ends
    x_train_np = np.stack(x_train)
    y_train_np = np.stack(y_train)

    x_train_np = x_train_np.reshape(-1, x_train_np.shape[-1])
    y_train_np = y_train_np.reshape(-1)
    # start training
    print(f"Training on {len(x_train)} samples")
    xgb_clf.fit(x_train_np, y_train_np, verbose=True)
    # save the model
    print("Saving model")
    xgb_clf.save_model("xgb_model.json")

    del x_train
    del x_train_np
    del y_train
    del y_train_np
    scores = test_xg(xgb_clf, test_dataset, window)

    test_scores_dict = dict(zip(("acc", "recall", "f1"), scores))

    print(scores)

    with open("test_scores.json", "w") as f:
        json.dump(test_scores_dict, f)


def test_xg(xgb_model, dataset, window):
    bar = tqdm(dataset, total=len(dataset), desc="Evaluating Matrix")

    x_test = []
    y_test = []
    print(f"Evaluating")
    for i, eeg_annot in enumerate(bar):
        dg = XGBLoader(eeg_annot, window_size=window)
        for x, y in dg:
            if y.sum() == 0:
                continue
            x_test.append(x)
            y_test.append(y)

    x_test_np = np.stack(x_test)
    y_test_np = np.stack(y_test)

    x_test_np = x_test_np.reshape(-1, x_test_np.shape[-1])
    y_test_np = y_test_np.reshape(-1)

    preds = xgb_model.predict(x_test_np)

    acc = accuracy_score(y_test_np, preds)
    recall = recall_score(y_test_np, preds, average="binary")
    f1 = f1_score(y_test_np, preds, average="binary")

    del x_test
    del x_test_np
    del y_test
    del y_test_np

    return acc, recall, f1


if __name__ == "__main__":
    training_data_path = "../bipolar_eeg_dataset/train_filtred_channel_based.json"
    testing_data_path = "../bipolar_eeg_dataset/dev_filtred_channel_based.json"

    with open(training_data_path, "r") as f:
        training_data = json.load(f)

    with open(testing_data_path, "r") as f:
        testing_data = json.load(f)

    train_xgb(training_data, testing_data, lr=1e-3, window=30)
