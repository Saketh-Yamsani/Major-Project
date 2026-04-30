# =============================================================
# SMART BULB - MODEL COMPARISON SCRIPT
# CONSISTENT PREPROCESSING PIPELINE
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam


# ============================
# 1️⃣ LOAD DATA
# ============================

normal_df = pd.read_csv("normal_data.csv", low_memory=False)
attack_df = pd.read_csv("attack_data.csv", low_memory=False)

normal_df["Label"] = 0
attack_df["Label"] = 1

data = pd.concat([normal_df, attack_df], ignore_index=True)

print("Initial Shape:", data.shape)


# ============================
# 2️⃣ REMOVE FULLY EMPTY COLUMNS
# ============================

data = data.loc[:, data.notna().any()]
print("After Removing Empty Columns:", data.shape)


# ============================
# 3️⃣ ENCODE CATEGORICAL FEATURES
# ============================

categorical_cols = data.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    if col not in ["Attack_Type", "Attack_Family"]:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))


# ============================
# 4️⃣ SEPARATE FEATURES & LABEL
# ============================

y = data["Label"]

drop_cols = ["Label", "Attack_Type", "Attack_Family"]
X = data.drop(columns=[col for col in drop_cols if col in data.columns])

X = X.select_dtypes(include=[np.number])


# ============================
# 5️⃣ REMOVE HIGHLY CORRELATED FEATURES
# ============================

corr_matrix = pd.DataFrame(X).corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr_features = [col for col in upper.columns if any(upper[col] > 0.90)]

X = pd.DataFrame(X).drop(columns=high_corr_features)

print("Removed Correlated Features:", len(high_corr_features))


# ============================
# 6️⃣ IMPUTE MISSING VALUES
# ============================

imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)


# ============================
# 7️⃣ STANDARDIZE FEATURES
# ============================

scaler = StandardScaler()
X = scaler.fit_transform(X)


# ============================
# 8️⃣ FEATURE SELECTION
# ============================

selector = SelectKBest(score_func=mutual_info_classif, k=25)
X = selector.fit_transform(X, y)

print("Final Feature Count:", X.shape[1])


# ============================
# 9️⃣ SPLIT NORMAL & ATTACK
# ============================

X_normal = X[y == 0]
X_attack = X[y == 1]

X_test = np.concatenate([X_normal, X_attack])
y_test = np.concatenate([np.zeros(len(X_normal)), np.ones(len(X_attack))])

results = []


# =========================================================
# 1️⃣ Isolation Forest
# =========================================================

iso = IsolationForest(contamination=0.1, random_state=42)
iso.fit(X_normal)
preds = iso.predict(X_test)
preds = np.where(preds == -1, 1, 0)

results.append([
    "IsolationForest",
    accuracy_score(y_test, preds),
    precision_score(y_test, preds),
    recall_score(y_test, preds),
    f1_score(y_test, preds)
])


# =========================================================
# 2️⃣ One-Class SVM
# =========================================================

svm = OneClassSVM()
svm.fit(X_normal)
preds = svm.predict(X_test)
preds = np.where(preds == -1, 1, 0)

results.append([
    "OneClassSVM",
    accuracy_score(y_test, preds),
    precision_score(y_test, preds),
    recall_score(y_test, preds),
    f1_score(y_test, preds)
])


# =========================================================
# 3️⃣ LOF
# =========================================================

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
lof.fit(X_normal)
preds = lof.predict(X_test)
preds = np.where(preds == -1, 1, 0)

results.append([
    "LOF",
    accuracy_score(y_test, preds),
    precision_score(y_test, preds),
    recall_score(y_test, preds),
    f1_score(y_test, preds)
])


# =========================================================
# 4️⃣ Elliptic Envelope
# =========================================================

ell = EllipticEnvelope(contamination=0.1, random_state=42)
ell.fit(X_normal)
preds = ell.predict(X_test)
preds = np.where(preds == -1, 1, 0)

results.append([
    "EllipticEnvelope",
    accuracy_score(y_test, preds),
    precision_score(y_test, preds),
    recall_score(y_test, preds),
    f1_score(y_test, preds)
])


# =========================================================
# 5️⃣ AUTOENCODER
# =========================================================

X_train, X_val = train_test_split(X_normal, test_size=0.3, random_state=42)

input_dim = X_train.shape[1]

input_layer = Input(shape=(input_dim,))
encoded = Dense(32, activation="relu")(input_layer)
encoded = Dense(16, activation="relu")(encoded)
encoded = Dense(8, activation="relu")(encoded)

decoded = Dense(16, activation="relu")(encoded)
decoded = Dense(32, activation="relu")(decoded)
decoded = Dense(input_dim, activation="linear")(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer=Adam(0.001), loss="mse")

autoencoder.fit(
    X_train, X_train,
    epochs=30,
    batch_size=128,
    validation_data=(X_val, X_val),
    shuffle=True,
    verbose=0
)

reconstructions = autoencoder.predict(X_test, verbose=0)
mse = np.mean(np.square(X_test - reconstructions), axis=1)

threshold = np.percentile(mse[:len(X_normal)], 95)
preds = (mse > threshold).astype(int)

results.append([
    "Autoencoder",
    accuracy_score(y_test, preds),
    precision_score(y_test, preds),
    recall_score(y_test, preds),
    f1_score(y_test, preds)
])


# =========================================================
# RESULTS TABLE
# =========================================================

results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1"]
)

print("\nModel Comparison:\n")
print(results_df)


# =========================================================
# BAR CHART: ACCURACY VS F1 SCORE
# =========================================================

x = np.arange(len(results_df["Model"]))
width = 0.35

plt.figure(figsize=(10,6))

plt.bar(x - width/2, results_df["Accuracy"], width, label="Accuracy")
plt.bar(x + width/2, results_df["F1"], width, label="F1 Score")

plt.xticks(x, results_df["Model"], rotation=45)
plt.ylabel("Score")
plt.title("Model Comparison (Accuracy vs F1 Score)")
plt.legend()

plt.tight_layout()
plt.show()


print("\nComparison completed successfully!")