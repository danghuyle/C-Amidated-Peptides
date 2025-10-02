import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    balanced_accuracy_score, roc_auc_score, average_precision_score,
    matthews_corrcoef, ConfusionMatrixDisplay, roc_curve, auc,
    precision_recall_curve
)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--group', required=True)
args = parser.parse_args()
group = args.group

x_train = pd.read_csv(f"output/{group}/x_train_{group}.csv")
x_test = pd.read_csv(f"output/{group}/x_test_{group}.csv")
y_train = pd.read_csv(f"output/{group}/y_train_{group}.csv").squeeze()
y_test = pd.read_csv(f"output/{group}/y_test_{group}.csv").squeeze()

group_counts = y_train.value_counts().sort_index()
print(group_counts)

os.makedirs("plots_sfs_ebm", exist_ok=True)
os.makedirs(f"plots_sfs_ebm/{group}", exist_ok=True)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)

ebm_model = ExplainableBoostingClassifier(random_state=100, n_jobs=-1)

sfs = SequentialFeatureSelector(
    estimator=ebm_model,
    direction='forward',
    scoring='f1',
    cv=cv.split(x_train, y_train),
    n_jobs=-1,
    tol=0.001
)
sfs.fit(x_train, y_train)

selected_features = sfs.get_support()
selected_feature_names = x_train.columns[selected_features]
print("Selected Features (Names):", list(selected_feature_names))

cv_f1_scores = cross_val_score(
    ebm_model, x_train[selected_feature_names], y_train,
    cv=cv, scoring="f1", n_jobs=-1
)
mean_cv_f1 = np.mean(cv_f1_scores)
print(f"\nBest F1 Score: {mean_cv_f1:.4f}")

x_train_final = x_train[selected_feature_names]
x_test_final = x_test[selected_feature_names]

final_ebm = ExplainableBoostingClassifier(random_state=100, n_jobs=-1)
final_ebm.fit(x_train_final, y_train)

y_train_pred = final_ebm.predict(x_train_final)
y_test_pred = final_ebm.predict(x_test_final)

train_f1 = f1_score(y_train, y_train_pred)
test_f1 = f1_score(y_test, y_test_pred)

print(f"\nTraining F1-Score: {train_f1:.4f}")
print(f"Test F1-Score: {test_f1:.4f}")

term_importances = final_ebm.term_importances()
term_names = final_ebm.term_names_
ranked_term_names = [name for name, _ in sorted(zip(term_names, term_importances), key=lambda x: x[1])]
print("\nRanked Features:")
print(ranked_term_names)

y_train_pred_proba = final_ebm.predict_proba(x_train_final)
y_test_pred_proba = final_ebm.predict_proba(x_test_final)

def binary_metrics(y_true, y_pred, y_pred_proba):
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred_proba[:, 1])
    auc_pr = average_precision_score(y_true, y_pred_proba[:, 1])
    mcc = matthews_corrcoef(y_true, y_pred)
    return accuracy, balanced_acc, f1, precision, recall, auc_roc, auc_pr, mcc

def plot_confusion_matrix(y_true, y_pred, labels=None, title="Confusion Matrix", save_path="plots_sfs_ebm/confusion_matrix.png"):
    if labels is None:
        labels = sorted(list(set(y_true)))
    folder = os.path.dirname(save_path)
    os.makedirs(folder, exist_ok=True)
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=labels, cmap="Blues", colorbar=True
    )
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_combined_roc_curve(y_train, y_train_proba, y_test, y_test_proba, save_path="plots_sfs_ebm/roc_curve_combined.png"):
    folder = os.path.dirname(save_path)
    os.makedirs(folder, exist_ok=True)
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba[:, 1])
    roc_auc_train = auc(fpr_train, tpr_train)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba[:, 1])
    roc_auc_test = auc(fpr_test, tpr_test)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_train, tpr_train, label=f"Train ROC (AUC = {roc_auc_train:.2f})")
    plt.plot(fpr_test, tpr_test, label=f"Test ROC (AUC = {roc_auc_test:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Train vs Test)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_combined_pr_curve(y_train, y_train_proba, y_test, y_test_proba, save_path="plots_sfs_ebm/pr_curve_combined.png"):
    folder = os.path.dirname(save_path)
    os.makedirs(folder, exist_ok=True)
    precision_train, recall_train, _ = precision_recall_curve(y_train, y_train_proba[:, 1])
    ap_score_train = average_precision_score(y_train, y_train_proba[:, 1])
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_proba[:, 1])
    ap_score_test = average_precision_score(y_test, y_test_proba[:, 1])
    plt.figure(figsize=(8, 6))
    plt.plot(recall_train, precision_train, label=f"Train PR (AP = {ap_score_train:.2f})")
    plt.plot(recall_test, precision_test, label=f"Test PR (AP = {ap_score_test:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Train vs Test)")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

plot_confusion_matrix(y_train, y_train_pred, save_path=f"plots_sfs_ebm/{group}/conf_matrix_train.png")
plot_confusion_matrix(y_test, y_test_pred, save_path=f"plots_sfs_ebm/{group}/conf_matrix_test.png")
plot_combined_roc_curve(y_train, y_train_pred_proba, y_test, y_test_pred_proba, save_path=f"plots_sfs_ebm/{group}/roc_curve_combined.png")
plot_combined_pr_curve(y_train, y_train_pred_proba, y_test, y_test_pred_proba, save_path=f"plots_sfs_ebm/{group}/pr_curve_combined.png")

train_metrics = binary_metrics(y_train, y_train_pred, y_train_pred_proba)
test_metrics = binary_metrics(y_test, y_test_pred, y_test_pred_proba)

print("\nTraining Metrics:")
print(f"Accuracy:             {train_metrics[0]:.4f}")
print(f"Balanced Accuracy:    {train_metrics[1]:.4f}")
print(f"F1 Score:             {train_metrics[2]:.4f}")
print(f"Precision:            {train_metrics[3]:.4f}")
print(f"Recall:               {train_metrics[4]:.4f}")
print(f"AUC ROC:              {train_metrics[5]:.4f}")
print(f"AUC PR:               {train_metrics[6]:.4f}")
print(f"MCC:                  {train_metrics[7]:.4f}")

print("\nTest Metrics:")
print(f"Accuracy:             {test_metrics[0]:.4f}")
print(f"Balanced Accuracy:    {test_metrics[1]:.4f}")
print(f"F1 Score:             {test_metrics[2]:.4f}")
print(f"Precision:            {test_metrics[3]:.4f}")
print(f"Recall:               {test_metrics[4]:.4f}")
print(f"AUC ROC:              {test_metrics[5]:.4f}")
print(f"AUC PR:               {test_metrics[6]:.4f}")
print(f"MCC:                  {test_metrics[7]:.4f}")
