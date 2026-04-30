# =============================================================
# SMART BULB - FINAL MODEL TRAINING WITH VISUALIZATION
# LOF + RANDOM FOREST + CONFUSION MATRICES + PIPELINE FIX
# =============================================================

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
from sklearn.decomposition import PCA

# ============================
# 1️⃣ LOAD DATA
# ============================

normal_df = pd.read_csv(r"C:\saketh\Major Project\normal_data.csv")
attack_df = pd.read_csv(r"C:\saketh\Major Project\attack_data.csv")

normal_df["Label"] = 0
attack_df["Label"] = 1

data = pd.concat([normal_df, attack_df], ignore_index=True)

print("Initial Shape:", data.shape)

# ============================
# 2️⃣ REMOVE EMPTY COLUMNS
# ============================

data = data.loc[:, data.notna().any()]

# ============================
# 3️⃣ ENCODE CATEGORICAL FEATURES
# ============================

categorical_cols = data.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    if col not in ["Attack_Type", "Attack_Family"]:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))

# ============================
# 4️⃣ FEATURE SEPARATION
# ============================

y_detection = data["Label"]

X = data.drop(columns=[col for col in ["Label", "Attack_Type", "Attack_Family"] if col in data.columns])
X = X.select_dtypes(include=[np.number])

# ============================
# 5️⃣ REMOVE CORRELATED FEATURES
# ============================

corr = X.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
drop_cols = [col for col in upper.columns if any(upper[col] > 0.90)]

X = X.drop(columns=drop_cols)

print("Removed Correlated:", len(drop_cols))

# =============================================================
# 🔥 IMPORTANT FIX (ONLY ADDITION)
# =============================================================

feature_columns = X.columns.tolist()
joblib.dump(feature_columns, "feature_columns.pkl")

# ============================
# 6️⃣ IMPUTE + SCALE
# ============================

imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# ============================
# 7️⃣ PCA
# ============================

pca = PCA(n_components=0.99)
X = pca.fit_transform(X)

print("After PCA:", X.shape)

# ============================# ============================
# SAVE PREPROCESSED DATA (WITH COLUMN NAMES)
# ============================

# Create meaningful PCA column names
pca_columns = [f"PC_{i+1}" for i in range(X.shape[1])]

# Convert numpy array to DataFrame
X_df = pd.DataFrame(X, columns=pca_columns)

# Add label column
X_df["Label"] = y_detection.values

# ----------------------------
# Save full dataset
# ----------------------------
X_df.to_csv("preprocessed_dataset.csv", index=False)

# ----------------------------
# Save only NORMAL data
# ----------------------------
normal_processed = X_df[X_df["Label"] == 0]
normal_processed.to_csv("preprocessed_normal.csv", index=False)

# ----------------------------
# Save only ATTACK data
# ----------------------------
attack_processed = X_df[X_df["Label"] == 1]
attack_processed.to_csv("preprocessed_attack.csv", index=False)

print("Preprocessed datasets saved successfully with proper column names!")

# =============================================================
# 🔟 LOF ANOMALY DETECTION
# =============================================================

X_normal = X[y_detection == 0]
X_attack = X[y_detection == 1]

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
lof.fit(X_normal)

# ============================
# LOF EVALUATION
# ============================

X_test = np.vstack([X_normal, X_attack])
y_test = np.hstack([np.zeros(len(X_normal)), np.ones(len(X_attack))])

preds = lof.predict(X_test)
y_pred = np.where(preds == -1, 1, 0)

print("\n=== LOF Detection Performance ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"]))

# ============================
# LOF CONFUSION MATRIX
# ============================

cm_lof = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(cm_lof, display_labels=["Normal", "Attack"])
disp.plot(ax=ax, cmap="Blues", values_format='d')

plt.title("LOF Confusion Matrix")
plt.tight_layout()
plt.show()

# =============================================================
# 🔟 RANDOM FOREST CLASSIFICATION
# =============================================================

if "Attack_Type" in data.columns:

    attack_data = X[y_detection == 1]
    attack_labels = data.loc[y_detection == 1, "Attack_Type"]

    le_attack = LabelEncoder()
    y_attack = le_attack.fit_transform(attack_labels)

    print("\nAttack Distribution:\n", attack_labels.value_counts())

    # ============================
    # BALANCE DATA
    # ============================

    df = pd.DataFrame(attack_data)
    df["label"] = y_attack

    max_size = df["label"].value_counts().max()

    balanced = []
    for label in df["label"].unique():
        subset = df[df["label"] == label]
        up = resample(subset, replace=True, n_samples=max_size, random_state=42)
        balanced.append(up)

    df_bal = pd.concat(balanced)

    X_bal = df_bal.drop("label", axis=1).values
    y_bal = df_bal["label"].values

    # ============================
    # CLASS WEIGHTS
    # ============================

    weights = compute_class_weight("balanced", classes=np.unique(y_bal), y=y_bal)
    class_weights = dict(enumerate(weights))

    # ============================
    # TRAIN TEST SPLIT
    # ============================

    X_train, X_test_clf, y_train, y_test_clf = train_test_split(
        X_bal,
        y_bal,
        test_size=0.3,
        stratify=y_bal,
        random_state=42
    )

    # ============================
    # RANDOM FOREST MODEL
    # ============================

    clf = RandomForestClassifier(
        n_estimators=1000,
        max_features="sqrt",
        min_samples_split=3,
        min_samples_leaf=1,
        class_weight=class_weights,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )

    clf.fit(X_train, y_train)

    ##print("OOB Score:", clf.oob_score_)

    y_pred_clf = clf.predict(X_test_clf)

    ##print("\n=== Random Forest Performance ===")
    ##print("Accuracy:", accuracy_score(y_test_clf, y_pred_clf))
    ##print("Precision (weighted):", precision_score(y_test_clf, y_pred_clf, average="weighted"))
    ##print("Recall (weighted):", recall_score(y_test_clf, y_pred_clf, average="weighted"))
    ##print("F1 (weighted):", f1_score(y_test_clf, y_pred_clf, average="weighted"))
    ##print("F1 (macro):", f1_score(y_test_clf, y_pred_clf, average="macro"))

    # ============================
    # TOP-3 ACCURACY
    # ============================

    probs = clf.predict_proba(X_test_clf)
    top3 = np.argsort(probs, axis=1)[:, -3:]

    correct = sum(y_test_clf[i] in top3[i] for i in range(len(y_test_clf)))
    print("Top-3 Accuracy:", correct / len(y_test_clf))

    # ============================
    # RF CONFUSION MATRIX
    # ============================

    cm_rf = confusion_matrix(y_test_clf, y_pred_clf)

    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(cm_rf, display_labels=le_attack.classes_)

    disp.plot(ax=ax, cmap="Blues")

    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)

    plt.title("Random Forest Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # ============================
    # SAVE MODELS
    # ============================

    joblib.dump(clf, "attack_classifier.pkl")
    joblib.dump(le_attack, "attack_label_encoder.pkl")

# =============================================================
# SAVE PIPELINE
# =============================================================

joblib.dump(lof, "lof_model.pkl")
joblib.dump(clf, "attack_classifier.pkl")
joblib.dump(le_attack, "attack_label_encoder.pkl")
joblib.dump(imputer, "imputer.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca.pkl")

print("\nTraining Completed Successfully")