import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Use os.path.join so it always finds the files inside majproj
true_df = pd.read_csv(os.path.join(BASE_DIR, "test.csv"))
pred_df = pd.read_csv(os.path.join(BASE_DIR, "final_output_with_zero_day.csv"))

true_labels = true_df['Label'].values
pred_labels = (pred_df['Final_Prediction'] != 'Normal').astype(int).values

total = len(true_labels)
lof_correct = (true_labels == pred_labels).sum()

attack_mask = (true_labels == 1)
attack_total = attack_mask.sum()
normal_total = total - attack_total

attack_caught = (pred_labels[attack_mask] == 1).sum()
normal_passed = (pred_labels[~attack_mask] == 0).sum()

print(f"--- LOF (Anomaly Detection) ---")
print(f"Total Samples Tested: {total}")
print(f"Normal Traffic Correctly Passed: {normal_passed} out of {normal_total} ({normal_passed/normal_total:.2%})")
print(f"Attacks Successfully Detected: {attack_caught} out of {attack_total} ({attack_caught/attack_total:.2%})")
print(f"Overall Accuracy: {lof_correct/total:.2%}")

true_fam = true_df['Attack_Family'].values
pred_fam = pred_df['Predicted_Attack_Family'].values

clf_mask = (pred_labels == 1) & (true_labels == 1)
if clf_mask.sum() > 0:
    fam_correct = (true_fam[clf_mask] == pred_fam[clf_mask]).sum()
    print(f"\n--- Random Forest (Family Classification) ---")
    print(f"Successfully Classified Families: {fam_correct} out of {clf_mask.sum()} attacks detected ({fam_correct/clf_mask.sum():.2%})")
