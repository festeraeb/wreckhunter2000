import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def train_random_forest(features_csv, model_output):
    """
    Train a Random Forest classifier to separate Wrecks from Wells.

    Args:
        features_csv (str): Path to the CSV file containing features and labels.
        model_output (str): Path to save the trained model.
    """
    data = pd.read_csv(features_csv)
    X = data.drop(columns=['label'])
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    joblib.dump(clf, model_output)
    print(f"Model saved to {model_output}")

if __name__ == "__main__":
    features_csv = "../sample_data/features.csv"  # Replace with actual path
    model_output = "../models/random_forest_model.pkl"

    train_random_forest(features_csv, model_output)