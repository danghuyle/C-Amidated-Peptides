import glob
import copy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import *
from transformers import EvalPrediction
from transformers.modeling_outputs import SequenceClassifierOutput
from esm.model.esm2 import ESM2
from esm import Alphabet
from torch.serialization import safe_globals


patterns = [
    "save_folds/fold1_pytorch_model.bin",
    "save_folds/fold2_pytorch_model.bin",
    "save_folds/fold3_pytorch_model.bin",
    "save_folds/fold4_pytorch_model.bin",
    "save_folds/fold5_pytorch_model.bin",
]
fold_model_paths = [sorted(glob.glob(pat))[-1] for pat in patterns if glob.glob(pat)]

pretrained_path = "./esm2_t6_8M_UR50D.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

def compute_metrics(pred: EvalPrediction):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    balanced_acc = balanced_accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='binary')
    recall = recall_score(labels, preds, average='binary')
    f1 = f1_score(labels, preds, average='binary')
    mcc = matthews_corrcoef(labels, preds)

    try:
        probs = pred.predictions[:, 1]
        auc_roc = roc_auc_score(labels, probs)
        auc_pr = average_precision_score(labels, probs)
        brier = brier_score_loss(labels, probs)
    except Exception as e:
        auc_roc = None
        auc_pr = None
        brier = None
        print("Could not calculate AUC or Brier:", e)

    return {
        'balanced_acc': balanced_acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'brier_score': brier,
        'mcc': mcc
    }

class amp_data(Dataset):
    def __init__(self, df):
        self.ids = list(df.index.astype(str))
        self.seqs = list(df['sequence'])
        self.labels = list(df['target'].astype(int))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'id': self.ids[idx],
            'sequence': self.seqs[idx],
            'labels': self.labels[idx]
        }

def esm_collate_fn(batch):
    data = [(item['id'], item['sequence']) for item in batch]
    labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
    _, _, batch_tokens = batch_converter(data)
    return {
        'input_ids': batch_tokens,
        'labels': labels
    }

class ESM2WithCustomHead(nn.Module):
    def __init__(self, backbone, num_labels=2, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(backbone.embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_labels)
        )

    def forward(self, input_ids, labels=None):
        outputs = self.backbone(input_ids, repr_layers=[self.backbone.num_layers], return_contacts=False)
        cls_repr = outputs["representations"][self.backbone.num_layers][:, 0, :]
        logits = self.classifier(cls_repr)
        return SequenceClassifierOutput(logits=logits)

test_df = pd.read_csv("../data/test_set.csv", index_col=None)
test_dataset = amp_data(test_df)
test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=esm_collate_fn)

def load_esm2_backbone(pretrained_path):
    with safe_globals([ESM2]):
        model_data = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    model = model_data["model"]
    alphabet = model_data["alphabet"]
    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter

backbone_model, alphabet, batch_converter = load_esm2_backbone(pretrained_path)
models = []
for path in fold_model_paths:
    cloned_backbone = copy.deepcopy(backbone_model)
    model = ESM2WithCustomHead(cloned_backbone, num_labels=2)
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    models.append(model)

all_preds = []
all_probs = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Inference with ensemble"):
        input_ids = batch['input_ids'].to(device)
        logits_list = [model(input_ids=input_ids).logits for model in models]
        avg_logits = torch.stack(logits_list).mean(dim=0)
        probs = torch.softmax(avg_logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

true_labels = test_df["target"].values
test_y_true = np.array(true_labels)
test_y_prob = np.array(all_probs)
test_y_pred = np.array(all_preds)

pred_logits = np.vstack([1 - test_y_prob, test_y_prob]).T
eval_pred = EvalPrediction(predictions=pred_logits, label_ids=test_y_true)

metrics_test = compute_metrics(eval_pred)

print("\nExternal Test Set Evaluation Results:")
for k, v in metrics_test.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

cm = confusion_matrix(test_y_true, test_y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\nConfusion Matrix:")
print(f"         Pred 0   Pred 1")
print(f"True 0   {tn:6}   {fp:6}")
print(f"True 1   {fn:6}   {tp:6}")
