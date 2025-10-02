import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import *
from joblib import dump

x_train = pd.read_csv("output/composition_CTD_global/x_train_composition_CTD_global.csv")
x_test = pd.read_csv("output/composition_CTD_global/x_test_composition_CTD_global.csv")
y_train = pd.read_csv("output/composition_CTD_global/y_train_composition_CTD_global.csv").squeeze()
y_test = pd.read_csv("output/composition_CTD_global/y_test_composition_CTD_global.csv").squeeze()

group_counts = y_train.value_counts().sort_index()
print(group_counts)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)
ebm_model = ExplainableBoostingClassifier(random_state=100, n_jobs=-1)

selected_feature_names = [
    'AAC_A',
    'CKSAAGP_alphaticr.postivecharger.gap1',
    'CKSAAGP_alphaticr.uncharger.gap2',
    'CKSAAGP_uncharger.uncharger.gap3',
    'CTDC_hydrophobicity_PONP930101.G3',
    'CTDD_hydrophobicity_CASG920101.3.residue0',
    'CTDD_hydrophobicity_FASG890101.3.residue75',
    'CTDD_hydrophobicity_ZIMJ680101.3.residue50',
    'Charge'
]

cv_f1_scores = cross_val_score(ebm_model, x_train[selected_feature_names], y_train, cv=cv, scoring="f1", n_jobs=-1)
mean_cv_f1 = np.mean(cv_f1_scores)
print(f"\nBest F1 Score: {mean_cv_f1:.4f}")

x_train_final = x_train[selected_feature_names]
x_test_final = x_test[selected_feature_names]

final_ebm = ExplainableBoostingClassifier(random_state=100, n_jobs=-1)
final_ebm.fit(x_train_final, y_train)

dump(final_ebm, "ebm_model.pkl")

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
