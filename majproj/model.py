# =============================================================
# SMART BULB - FINAL MODEL TRAINING WITH EVALUATION TECHNIQUES
# LOF + RANDOM FOREST + ROC + CROSS VALIDATION
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
    classification_report, roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
from sklearn.decomposition import PCA

# ============================
# 1 LOAD DATA
# ============================

normal_df = pd.read_csv(r"C:\saketh\Major Project\normal_data.csv")
attack_df = pd.read_csv(r"C:\saketh\Major Project\attack_data.csv")

normal_df["Label"] = 0
attack_df["Label"] = 1

data = pd.concat([normal_df, attack_df], ignore_index=True)

print("Initial Shape:", data.shape)

# ============================
# 2 CLEAN DATA
# ============================

data = data.loc[:, data.notna().any()]

# ============================
# 3 ENCODE
# ============================

for col in data.select_dtypes(include=["object"]).columns:
    if col not in ["Attack_Type", "Attack_Family"]:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))

# ============================
# 4 FEATURES
# ============================

y_detection = data["Label"]

X = data.drop(columns=[col for col in ["Label", "Attack_Type", "Attack_Family"] if col in data.columns])
X = X.select_dtypes(include=[np.number])

# ============================
# 5 REMOVE CORRELATED
# ============================

corr = X.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
drop_cols = [col for col in upper.columns if any(upper[col] > 0.90)]

X = X.drop(columns=drop_cols)

print("Removed Correlated:", len(drop_cols))

# SAVE FEATURE NAMES
feature_columns = X.columns.tolist()
joblib.dump(feature_columns, "feature_columns.pkl")

# ============================
# 6 IMPUTE + SCALE
# ============================

imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# ============================
# 7 PCA
# ============================

pca = PCA(n_components=0.99)
X = pca.fit_transform(X)

print("After PCA:", X.shape)

# ============================
# SAVE PREPROCESSED DATA
# ============================

pca_columns = [f"PC_{i+1}" for i in range(X.shape[1])]
X_df = pd.DataFrame(X, columns=pca_columns)
X_df["Label"] = y_detection.values

X_df.to_csv("preprocessed_dataset.csv", index=False)
X_df[X_df["Label"] == 0].to_csv("preprocessed_normal.csv", index=False)
X_df[X_df["Label"] == 1].to_csv("preprocessed_attack.csv", index=False)

# =============================================================
# LOF
# =============================================================

X_normal = X[y_detection == 0]
X_attack = X[y_detection == 1]

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
lof.fit(X_normal)

# TEST SET
X_test = np.vstack([X_normal, X_attack])
y_test = np.hstack([np.zeros(len(X_normal)), np.ones(len(X_attack))])

preds = lof.predict(X_test)
y_pred = np.where(preds == -1, 1, 0)

print("\n=== LOF Performance ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))

# ============================
# ROC-AUC (Evaluation Technique)
# ============================

lof_scores = -lof.decision_function(X_test)

auc = roc_auc_score(y_test, lof_scores)
print("LOF ROC-AUC:", auc)

fpr, tpr, _ = roc_curve(y_test, lof_scores)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title("LOF ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()

# ============================
# CONFUSION MATRIX
# ============================

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Attack"])
disp.plot()
plt.title("LOF Confusion Matrix")
plt.show()

# =============================================================
# RANDOM FOREST
# =============================================================

attack_data = X[y_detection == 1]
attack_labels = data.loc[y_detection == 1, "Attack_Type"]

le_attack = LabelEncoder()
y_attack = le_attack.fit_transform(attack_labels)

# BALANCE DATA
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

# CLASS WEIGHTS
weights = compute_class_weight("balanced", classes=np.unique(y_bal), y=y_bal)
class_weights = dict(enumerate(weights))

# SPLIT
X_train, X_test_clf, y_train, y_test_clf = train_test_split(
    X_bal, y_bal, test_size=0.3, stratify=y_bal, random_state=42
)

# MODEL
clf = RandomForestClassifier(
    n_estimators=1000,
    max_features="sqrt",
    class_weight=class_weights,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

clf.fit(X_train, y_train)

print("OOB Score:", clf.oob_score_)

y_pred_clf = clf.predict(X_test_clf)

print("\n=== RF Performance ===")
print("Accuracy:", accuracy_score(y_test_clf, y_pred_clf))
print("F1:", f1_score(y_test_clf, y_pred_clf, average="weighted"))

# ============================
# CROSS VALIDATION (Evaluation Technique)
# ============================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(clf, X_bal, y_bal, cv=cv, scoring='accuracy', n_jobs=-1)

print("\nCross Validation Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))

# ============================
# CONFUSION MATRIX RF
# ============================

cm_rf = confusion_matrix(y_test_clf, y_pred_clf)
disp = ConfusionMatrixDisplay(cm_rf, display_labels=le_attack.classes_)

disp.plot()
plt.xticks(rotation=45)
plt.title("RF Confusion Matrix")
plt.show()

# ============================
# SAVE MODELS
# ============================

joblib.dump(lof, "lof_model.pkl")
joblib.dump(clf, "attack_classifier.pkl")
joblib.dump(le_attack, "attack_label_encoder.pkl")
joblib.dump(imputer, "imputer.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca.pkl")

print("\nTraining Completed Successfully")