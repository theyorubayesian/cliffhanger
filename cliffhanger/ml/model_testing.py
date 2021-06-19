import pandas as pd

from cliffhanger.ml.data import process_data
from cliffhanger.utils import load_asset
from cliffhanger.ml.model import compute_model_metrics
from cliffhanger.ml.train_model import cat_features


def test_model(data, cat_cols: list = None):
    model = load_asset("trained_model.pkl")
    encoder = load_asset("encoder.pkl")
    lb = load_asset("lb.pkl")
    cat_cols = cat_cols or cat_features

    performance_df = pd.DataFrame(
        columns=["feature", "category", "precision", "recall", "fbeta"]
    )
    for feature in cat_cols:
        feature_performance = []
        for category in data[feature].unique():
            subset = data.loc[data.feature == category]
            X_test, y_test, *_ = process_data(
                subset,
                categorical_features=cat_cols,
                label="salary",
                training=False,
                encoder=encoder,
                lb=lb
            )
            y_pred = model.predict(X_test)
            precision, recall, fbeta = compute_model_metrics(y_test, y_pred)

            feature_performance.append({
                "feature": feature,
                "category": category,
                "precision": precision,
                "recall": recall,
                "fbeta": fbeta
            })
        performance_df.append(feature_performance, ignore_index=True)
    performance_df.to_csv("Model_Performance.csv", index=False)
