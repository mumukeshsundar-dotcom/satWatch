
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "dataset.csv"

df = pd.read_csv(DATA_PATH)

# Preserve the dataset's official split.
train_df = df.loc[df["train"] == 1].copy()
test_df = df.loc[df["train"] == 0].copy()

TARGET = "anomaly"

# Never use labels, split indicators, or segment IDs as features.
excluded = {"anomaly", "train", "segment"}
feature_columns = [
    col for col in df.columns if col not in excluded
]

X_train = train_df[feature_columns]
y_train = train_df[TARGET]

X_test = test_df[feature_columns]
y_test = test_df[TARGET]

categorical = ["channel"]
numeric = [
    col for col in feature_columns
    if col not in categorical
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            SimpleImputer(strategy="median"),
            numeric,
        ),
        (
            "categorical",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]),
            categorical,
        ),
    ]
)

models = {
    "Dummy baseline": DummyClassifier(
        strategy="most_frequent"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}

print(f"Training rows: {len(X_train)}")
print(f"Test rows: {len(X_test)}")
print(f"Features: {feature_columns}")
print()

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model),
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).ravel()

    false_alarm_rate = (
        fp / (fp + tn) if (fp + tn) else 0.0
    )

    print(f"\n=== {name} ===")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"F1-score: {f1_score(y_test, predictions, zero_division=0):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")
    print(
        "Average precision: "
        f"{average_precision_score(y_test, probabilities):.4f}"
    )
    print(f"False-alarm rate: {false_alarm_rate:.4f}")
    print(f"Confusion matrix:\n{confusion_matrix(y_test, predictions, labels=[0, 1])}")
    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Normal", "Anomaly"],
        zero_division=0,
    ))
