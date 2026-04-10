#!/usr/bin/env python3
"""Train a random forest classifier on wreck_dna_library features."""
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / 'wreck_dna_library.csv'
OUT_MODEL = ROOT / 'wreck_classifier_v1.bin'

if not CSV_PATH.exists():
    raise FileNotFoundError(CSV_PATH)

# Load data
raw = pd.read_csv(CSV_PATH)

# Map type from name / add a label: steel/iron=1, wood=0
label_map = {
    'Cedarville': 1,
    'Acme': 1,
    'Sandusky': 0,
    'Atlantic': 0,
    'Sport': 1,
    'Philadelphia': 1,
}
raw['label'] = raw['name'].map(label_map).fillna(0).astype(int)

# Feature engineering
raw['b01b03_med'] = raw['b01b03_med'].replace({np.nan: 0.0})
raw['fog_pulse'] = raw['fog_pulse'].astype(int)
raw['sdb_deviation'] = raw['sdb_deviation'].astype(int)
raw['ecostress_zscore'] = raw['ecostress_zscore'].fillna(0.0)
raw['sar_coherence'] = raw['sar_coherence'].fillna(0.0)

features = ['sdb_delta_m', 'b01b03_med', 'fog_pulse', 'sdb_deviation', 'ecostress_zscore', 'sar_coherence']
X = raw[features].fillna(0.0)
y = raw['label']

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123, test_size=0.33, stratify=y)

clf = RandomForestClassifier(n_estimators=100, random_state=123, n_jobs=4)
clf.fit(X_train, y_train)

acc = clf.score(X_test, y_test)
cv_splits = min(3, len(X))
if cv_splits < 2:
    cv_scores = [acc]
else:
    cv_scores = cross_val_score(clf, X, y, cv=cv_splits)

# Save model
with open(OUT_MODEL, 'wb') as f:
    pickle.dump(clf, f)

print(f"Trained model saved to {OUT_MODEL}")
print(f"Test accuracy: {acc:.3f}")
print(f"Cross-validated accuracy: {np.mean(cv_scores):.3f} +/- {np.std(cv_scores):.3f}")
