# =============================================================
# SMART BULB - IMPROVED MODEL TRAINING (FIXED VERSION)
# =============================================================

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
# 1. LOAD DATA
# ============================

normal_df = pd.read_csv(r"C:\saketh\Major Project\new_normal_data.csv")
attack_df = pd.read_csv(r"C:\saketh\Major Project\new_attack_data.csv")

normal_df["Label"] = 0
attack_df["Label"] = 1

print("Normal shape:", normal_df.shape)
print("Attack shape:", attack_df.shape)

# ============================
# 2. ALIGN COMMON COLUMNS
# ============================

META_COLS = ["Label", "Attack_Type", "Attack_Family"]

normal_feat_cols = [c for c in normal_df.columns if c not in META_COLS]
attack_feat_cols = [c for c in attack_df.columns if c not in META_COLS]

common_feat_cols = list(set(normal_feat_cols) & set(attack_feat_cols))

normal_aligned = normal_df[common_feat_cols + ["Label"]].copy()
attack_aligned = attack_df[common_feat_cols + ["Label", "Attack_Family"]].copy()

# Add missing meta cols
if "Attack_Family" not in normal_aligned.columns:
    normal_aligned["Attack_Family"] = "Normal"

data = pd.concat([normal_aligned, attack_aligned], ignore_index=True)

print("\nCombined shape:", data.shape)
print("Label distribution:\n", data["Label"].value_counts())

# ============================
# 3. CLEAN DATA
# ============================

data = data.loc[:, data.notna().any()]

# ============================
# 4. ENCODE OBJECTS
# ============================

for col in data.select_dtypes(include=["object"]).columns:
    if col != "Attack_Family":
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))

# ============================
# 5. FEATURES
# ============================

y = data["Label"].values

X = data.drop(columns=["Label", "Attack_Family"], errors="ignore")
X = X.select_dtypes(include=[np.number])

# ============================
# 6. CORRELATION (REMOVED AGGRESSIVE DROPPING)
# ============================

# We skip dropping highly correlated features here because it was destroying
# too much variance, leading to poor LOF performance. We let PCA handle it.
# corr = X.corr().abs()
# upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
# drop_cols = [col for col in upper.columns if any(upper[col] > 0.98)]
# X = X.drop(columns=drop_cols)

print("Remaining features:", X.shape[1])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

feature_columns = X.columns.tolist()
joblib.dump(feature_columns, os.path.join(BASE_DIR, "feature_columns.pkl"))

# ============================
# 7. IMPUTE + SCALE
# ============================

imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# ============================
# 8. PCA (RETAIN 99% VARIANCE)
# ============================

USE_PCA = True

if USE_PCA:
    pca = PCA(n_components=0.99, random_state=42)  # FIXED back to variance retention
    X = pca.fit_transform(X)
    joblib.dump(pca, os.path.join(BASE_DIR, "pca.pkl"))
    print("After PCA:", X.shape)
else:
    print("PCA Skipped")

# ============================
# LOF (ANOMALY DETECTION)
# ============================

idx = np.arange(len(X))
idx_normal = idx[y == 0]
idx_attack = idx[y == 1]

# Split
idx_normal_train, idx_normal_test = train_test_split(
    idx_normal, test_size=0.3, random_state=42
)

X_normal_train = X[idx_normal_train]
X_normal_test = X[idx_normal_test]
X_attack_test = X[idx_attack]

X_test = np.vstack([X_normal_test, X_attack_test])
y_test = np.hstack([np.zeros(len(X_normal_test)), np.ones(len(X_attack_test))])

# FIXED contamination
lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.1,   # FIXED
    novelty=True
)

lof.fit(X_normal_train)

preds = lof.predict(X_test)
y_pred = np.where(preds == -1, 1, 0)

print("\n=== LOF Performance ===")
lof_acc = accuracy_score(y_test, y_pred)
lof_prec = precision_score(y_test, y_pred, zero_division=0)
lof_rec = recall_score(y_test, y_pred, zero_division=0)
lof_f1 = f1_score(y_test, y_pred, zero_division=0)

print("Accuracy :", lof_acc)
print("Precision:", lof_prec)
print("Recall   :", lof_rec)
print("F1       :", lof_f1)
print("\nLOF Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"]))

# ROC
lof_scores = -lof.decision_function(X_test)
auc = roc_auc_score(y_test, lof_scores)
print("LOF ROC-AUC:", auc)

# === PLOT: LOF Confusion Matrix ===
plt.figure(figsize=(6, 5))
cm_lof = confusion_matrix(y_test, y_pred)
sns.heatmap(cm_lof, annot=True, fmt='d', cmap='Blues', xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
plt.title('LOF Anomaly Detection Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'lof_confusion_matrix.png'), dpi=300)
plt.close()

# === PLOT: LOF ROC Curve ===
fpr, tpr, _ = roc_curve(y_test, lof_scores)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (LOF)')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'lof_roc_curve.png'), dpi=300)
plt.close()

# ============================
# RANDOM FOREST (ATTACK CLASSIFICATION)
# ============================

attack_mask = (y == 1)
X_attack = X[attack_mask]
attack_labels = data.loc[attack_mask, "Attack_Family"]

le_attack = LabelEncoder()
y_attack = le_attack.fit_transform(attack_labels)

# Balance
df = pd.DataFrame(X_attack)
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

# Train/Test
X_train, X_test_rf, y_train, y_test_rf = train_test_split(
    X_bal, y_bal, test_size=0.3, stratify=y_bal, random_state=42
)

# FIXED RF (stronger)
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

clf.fit(X_train, y_train)

y_pred_rf = clf.predict(X_test_rf)

print("\n=== RF Performance ===")
rf_acc = accuracy_score(y_test_rf, y_pred_rf)
rf_prec = precision_score(y_test_rf, y_pred_rf, average="weighted", zero_division=0)
rf_rec = recall_score(y_test_rf, y_pred_rf, average="weighted", zero_division=0)
rf_f1 = f1_score(y_test_rf, y_pred_rf, average="weighted")

print("Accuracy :", rf_acc)
print("F1       :", rf_f1)
print("\nRF Classification Report:")
target_names = le_attack.classes_
print(classification_report(y_test_rf, y_pred_rf, target_names=target_names))

# === PLOT: RF Confusion Matrix ===
plt.figure(figsize=(10, 8))
cm_rf = confusion_matrix(y_test_rf, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Random Forest Attack Family Confusion Matrix')
plt.ylabel('True Family')
plt.xlabel('Predicted Family')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'rf_confusion_matrix.png'), dpi=300)
plt.close()

# === PLOT: RF Feature Importance ===
plt.figure(figsize=(10, 6))
importances = clf.feature_importances_
# Since we used PCA, the features are just components
indices = np.argsort(importances)[::-1][:20] # Top 20
plt.bar(range(len(indices)), importances[indices], align="center", color='teal')
plt.xticks(range(len(indices)), [f"PCA_{i+1}" for i in indices], rotation=45, ha='right')
plt.title("Random Forest Top 20 Feature Importances (PCA Components)")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'rf_feature_importance.png'), dpi=300)
plt.close()

# === PLOT: Model Comparison Bar Chart ===
plt.figure(figsize=(8, 5))
metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
lof_metrics = [lof_acc, lof_prec, lof_rec, lof_f1]
rf_metrics = [rf_acc, rf_prec, rf_rec, rf_f1]

x = np.arange(len(metrics_labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, lof_metrics, width, label='LOF (Anomaly)', color='cornflowerblue')
rects2 = ax.bar(x + width/2, rf_metrics, width, label='RF (Classification)', color='lightcoral')

ax.set_ylabel('Scores')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(metrics_labels)
ax.set_ylim([0, 1.1])
ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2)

# Add text labels
for rects in [rects1, rects2]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'model_performance_comparison.png'), dpi=300)
plt.close('all')

# ============================
# SAVE MODELS
# ============================

joblib.dump(lof, os.path.join(BASE_DIR, "lof_model.pkl"))
joblib.dump(clf, os.path.join(BASE_DIR, "attack_classifier.pkl"))
joblib.dump(le_attack, os.path.join(BASE_DIR, "attack_label_encoder.pkl"))
joblib.dump(imputer, os.path.join(BASE_DIR, "imputer.pkl"))
joblib.dump(scaler, os.path.join(BASE_DIR, "scaler.pkl"))

try:
    print("\n✅ FIXED TRAINING COMPLETED")
except UnicodeEncodeError:
    print("\nFIXED TRAINING COMPLETED")