# =============================================================
# detect.py
# FINAL VERSION: CORRECT OUTPUT LOGIC
# =============================================================

import pandas as pd
import numpy as np
import joblib
import datetime
import smtplib
from email.mime.text import MIMEText

CONFIDENCE_THRESHOLD = 0.60
ALERT_LOG_FILE = "security_alerts.log"

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

SENDER_EMAIL = "5sakethyamsani@gmail.com"
SENDER_PASSWORD = "gpkz uznw xjem zvkf"
RECEIVER_EMAIL = "abhishekgoudgadivenuka@gmail.com"

# ================= EMAIL =================

def send_summary_email(original_df, n_attacks):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    attack_rows = original_df[original_df["Final_Prediction"] != "Normal"]
    attack_summary = attack_rows[["Final_Prediction", "Predicted_Attack_Family", "Confidence_Score"]].to_string(index=False)

    message = f"""
IoT SECURITY SUMMARY ALERT

Time: {timestamp}
Total Attacks Detected: {n_attacks}

--- Attack Details ---
{attack_summary}
"""

    msg = MIMEText(message)
    msg["Subject"] = f"Smart Bulb Security Alert — {n_attacks} threat(s) detected"
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        server.quit()
        print("Summary email alert sent!")
    except Exception as e:
        print("Email failed:", e)

# ================= LOG =================

def log_alert(prediction, attack_type, confidence):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(ALERT_LOG_FILE, "a") as f:
        f.write(f"{timestamp} | {prediction} | {attack_type} | Confidence: {confidence}\n")

import os

# Use absolute path based on script location so it always loads the right models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

lof = joblib.load(os.path.join(BASE_DIR, "lof_model.pkl"))
clf = joblib.load(os.path.join(BASE_DIR, "attack_classifier.pkl"))
le_attack = joblib.load(os.path.join(BASE_DIR, "attack_label_encoder.pkl"))

imputer = joblib.load(os.path.join(BASE_DIR, "imputer.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
pca = joblib.load(os.path.join(BASE_DIR, "pca.pkl"))

feature_columns = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))

print("Models + pipeline loaded successfully!")

# ================= INPUT =================

# Load the test.csv from the same directory as the script
test_path = os.path.join(BASE_DIR, "Datasets", "test.csv")
new_data = pd.read_csv(test_path)

original = new_data.copy()

if "Label" in new_data.columns:
    X = new_data.drop(columns=["Label"])
else:
    X = new_data.copy()

# ================= FEATURE ALIGNMENT =================

X = X.select_dtypes(include=[np.number])
X = X.reindex(columns=feature_columns, fill_value=0)
X = X.copy()

# ================= PIPELINE =================

X = imputer.transform(X)
X = scaler.transform(X)
X = pca.transform(X)

# ================= LOF =================
# Optimized batch prediction
lof_preds = lof.predict(X)
attack_flags = np.where(lof_preds == -1, 1, 0)

original["Final_Prediction"] = "Normal"
original["Predicted_Attack_Family"] = "None"
original["Confidence_Score"] = 0.0

# ================= NORMAL =================

for idx in np.where(attack_flags == 0)[0]:
    print("\nNORMAL TRAFFIC DETECTED")
    print("Prediction: Normal")

# ================= CLASSIFICATION =================

if attack_flags.sum() > 0:

    attack_indices = np.where(attack_flags == 1)[0]
    attack_rows = X[attack_flags == 1]

    probs = clf.predict_proba(attack_rows)
    max_probs = probs.max(axis=1)

    preds = clf.predict(attack_rows)
    decoded = le_attack.inverse_transform(preds)

    for i, idx in enumerate(attack_indices):

        confidence = float(max_probs[i])

        if confidence >= CONFIDENCE_THRESHOLD:
            prediction = "Known Attack"
            attack_type = decoded[i]
        else:
            prediction = "Abnormal Pattern Detected"
            attack_type = "None"

        original.loc[idx, "Final_Prediction"] = prediction
        original.loc[idx, "Predicted_Attack_Family"] = attack_type
        original.loc[idx, "Confidence_Score"] = round(confidence, 4)

        print("\nSECURITY ALERT")
        print("Prediction:", prediction)

        if prediction == "Known Attack":
            print("Attack Family:", attack_type)

        log_alert(prediction, attack_type, confidence)

    # Send one summary email instead of per-sample spam
    send_summary_email(original, attack_flags.sum())

# ================= SAVE =================

original.to_csv("final_output_with_zero_day.csv", index=False)

print("\nDetection completed.")