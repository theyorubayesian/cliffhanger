import os

import pandas as pd

from cliffhanger.ml.data import process_data
from cliffhanger.ml.model import compute_model_metrics
from cliffhanger.utils import load_asset


def evaluate_model(
    data,
    cat_cols: list,
    output_dir: str,
    model=None,
    encoder=None,
    lb=None,
):
    model = model or load_asset("trained_model.pkl")
    encoder = encoder or load_asset("encoder.pkl")
    lb = lb or load_asset("lb.pkl")

    performance_df = pd.DataFrame(columns=["feature", "category", "precision", "recall", "fbeta"])
    for feature in cat_cols:
        feature_performance = []
        for category in data[feature].unique():
            subset = data.loc[data[feature] == category]
            X_test, y_test, *_ = process_data(
                subset,
                categorical_features=cat_cols,
                label="salary",
                training=False,
                encoder=encoder,
                lb=lb,
            )
            y_pred = model.predict(X_test)
            precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

            feature_performance.append(
                {
                    "feature": feature,
                    "category": category,
                    "precision": precision,
                    "recall": recall,
                    "fbeta": fbeta,
                }
            )
        performance_df = performance_df.append(feature_performance, ignore_index=True)
    output_file = os.path.join(output_dir, "slice_output.txt")
    performance_df.to_csv(output_file, index=False)
