import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from joblib import dump

dataset_path = "dataset_37_diabetes.csv"
model_path = "knn_model.joblib"
scaler_path = "scaler.joblib"
accuracy_path = "accuracy.joblib"
train_test_ratio = 0.05
n_neighbors = 3


def get_raw_dataset():
    return pd.read_csv(dataset_path)


def get_encoded_dataset():
    dataset_encoded = get_raw_dataset()

    dataset_encoded["class"] = ("tested_positive" == dataset_encoded["class"]).astype(
        int
    )

    median_zeros = ["plas", "pres", "skin", "insu", "mass", "pedi", "age"]

    for col in median_zeros:
        med = dataset_encoded[col].median()
        dataset_encoded[col] = dataset_encoded[col].replace(0.0, med)

    return dataset_encoded


def train_model():
    dataset_encoded = get_encoded_dataset()

    Y = dataset_encoded["class"]
    X = dataset_encoded.copy().drop("class", axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=train_test_ratio, random_state=42
    )

    print("Train size:", X_test.shape[1])
    print("Test size:", X_test.shape[1])

    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn_reg = KNeighborsClassifier(n_neighbors=n_neighbors)

    knn_reg.fit(X_train_scaled, y_train)

    y_pred = knn_reg.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {acc*100:.2f}")

    dump(knn_reg, model_path)
    dump(scaler, scaler_path)
    dump(acc, accuracy_path)


if __name__ == "__main__":
    train_model()
