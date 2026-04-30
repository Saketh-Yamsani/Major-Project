# 📘 Smart Home Security System (Anomaly, Attack Detection & Classification)

This project implements a hybrid intrusion detection system for IoT smart bulb environments by combining anomaly detection and supervised attack classification. The system identifies normal traffic, detects suspicious or unknown behavior, and classifies known attack types with confidence scoring.

The overall pipeline follows a structured approach: raw network data is preprocessed (cleaning, encoding, scaling), transformed using PCA for dimensionality reduction, passed through a Local Outlier Factor (LOF) model for anomaly detection, and then routed to a Random Forest classifier for attack family identification. This layered design enables both detection of zero-day anomalies and classification of known threats.

Project structure:

Major Project/
│
├── majproj/
│   ├── train_model_fixed.py
│   ├── detect.py
│   ├── pkl/              (stored models and preprocessing artifacts)
│   ├── outputs/          (generated plots, CSVs, logs)
│
├── Datasets/
│   ├── new_normal_data.csv
│   ├── new_attack_data.csv
│   ├── test.csv
│
└── README.md

During training, the system loads normal and attack datasets, aligns common feature spaces, applies preprocessing (imputation, scaling), and reduces dimensionality using PCA while retaining 99% of variance. The LOF model is trained on normal traffic to learn baseline behavior, while the Random Forest classifier is trained on attack data (balanced via resampling) to classify attack families. All trained models and preprocessing components are stored in the majproj/pkl/ directory.

To train the models:
python majproj/train_model_fixed.py

For detection, the system loads the trained models, processes the test dataset, applies the same preprocessing pipeline, and performs anomaly detection followed by classification. The results include prediction labels, attack family classification, and confidence scores.

To run detection:
python majproj/detect.py

All generated outputs are saved in majproj/outputs/, including:

* final_output_with_zero_day.csv containing predictions, attack families, and confidence scores
* security_alerts.log with timestamped alert records
* Visualization plots such as confusion matrices, ROC curves, feature importance, and model comparison charts

The system includes an alert mechanism that logs suspicious activities and sends summary email notifications with detected threats, their types, and associated confidence levels.

Saved model artifacts in majproj/pkl/:

* lof_model.pkl
* attack_classifier.pkl
* attack_label_encoder.pkl
* imputer.pkl
* scaler.pkl
* pca.pkl
* feature_columns.pkl

A confidence threshold of 0.60 is used to determine classification certainty. Predictions below this threshold are treated as abnormal patterns, while higher-confidence predictions are classified as known attacks.

Dependencies:
pip install pandas numpy scikit-learn matplotlib seaborn joblib

To prevent large files from being pushed to version control, add the following to .gitignore:
majproj/pkl/
majproj/outputs/
*.pkl

Future enhancements include integrating advanced anomaly detection techniques, enabling real-time data streaming, building a monitoring dashboard, deploying the system as an API, and improving feature engineering.

Author: Saketh Yamsani
