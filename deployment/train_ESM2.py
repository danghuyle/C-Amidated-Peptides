import os
import shutil
import math
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    balanced_accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score, average_precision_score
)
import argparse
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Trainer, TrainingArguments, EarlyStoppingCallback,
    EvalPrediction
)
from transformers.modeling_outputs import SequenceClassifierOutput
from esm.model.esm2 import ESM2
from torch.serialization import safe_globals

def get_args():
    defaults = {
        'batch_size': 8,
        'grad_accum_steps': 8,
        'epochs': 50,
        'lr': 1e-5,
        'k_folds': 5,
        'weight_decay': 0.01,
        "smoothing": 0,
        'checkpoints_dir': None,
        'dropout': 0.1
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", "-b", type=int)
    parser.add_argument("--grad_accum_steps", "-g", type=int)
    parser.add_argument("--epochs", "-e", type=int)
    parser.add_argument("--lr", "-l", type=float)
    parser.add_argument("--k_folds", "-k", type=int)
    parser.add_argument("--weight_decay", "-w", type=float)
    parser.add_argument("--smoothing", "-s", type=float)    
    parser.add_argument("--checkpoints_dir", "-c", type=str)
    parser.add_argument("--dropout", "-d", type=float)

    parser.set_defaults(**defaults)
    args = parser.parse_args()

    changed = {
        k: getattr(args, k)
        for k in sorted(defaults)
        if getattr(args, k) != defaults[k]
    }

    base_dir = f"esm2_t6_checkpoints_dir"
    if args.checkpoints_dir is None:
        if changed:
            suffix = "~".join(f"{k[0]}_{changed[k]}" for k in changed)
            args.checkpoints_dir = os.path.join(base_dir, suffix)
        else:
            args.checkpoints_dir = os.path.join(base_dir, "default")

    return args

args = get_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pretrained_path = "./esm2_t6_8M_UR50D.pt"
if os.path.exists(args.checkpoints_dir):
    shutil.rmtree(args.checkpoints_dir)
os.makedirs(args.checkpoints_dir, exist_ok=True)

df = pd.read_csv("../data/model_set.csv")
labels = df['target'].astype(int).values
test_ratio = 0.1
num_samples = math.ceil((1 - 1 / args.k_folds) * (1 - test_ratio) * len(df))
steps_per_epoch = num_samples // args.batch_size
warmup_steps = int(1 * steps_per_epoch)

with safe_globals([ESM2]):
    model_data = torch.load(pretrained_path, map_location="cpu", weights_only=False)
alphabet = model_data["alphabet"]
batch_converter = alphabet.get_batch_converter()

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

def compute_metrics(pred: EvalPrediction):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    try:
        probs = pred.predictions[:, 1]
        auc_roc = roc_auc_score(labels, probs)
        auc_pr = average_precision_score(labels, probs)
    except:
        auc_roc = None
        auc_pr = None

    return {
        'balanced_acc': balanced_accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='binary'),
        'precision': precision_score(labels, preds, average='binary'),
        'recall': recall_score(labels, preds, average='binary'),
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'mcc': matthews_corrcoef(labels, preds)
    }

class ESM2WithCustomHead(nn.Module):
    def __init__(self, model, num_labels=2, dropout=0.1):
        super().__init__()
        self.backbone = model
        hidden_size = self.backbone.embed_dim

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_labels)
        )

    def forward(self, input_ids, labels=None):
        outputs = self.backbone(input_ids, repr_layers=[self.backbone.num_layers], return_contacts=False)
        token_representations = outputs["representations"][self.backbone.num_layers]
        pooled_output = token_representations[:, 0, :]
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)

def custom_collate_fn(batch):
    data = [(item['id'], item['sequence']) for item in batch]
    labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
    _, _, batch_tokens = batch_converter(data)
    return {'input_ids': batch_tokens, 'labels': labels}

class CustomTrainer(Trainer):
    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        return DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=10)
folds = list(skf.split(df, labels))
fold_model_paths = []

for fold, (train_idx, val_idx) in enumerate(folds):
    df_train = df.iloc[train_idx].sample(frac=1, random_state=fold).reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)

    with safe_globals([ESM2]):
        model_data = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    backbone_model = model_data["model"]

    train_dataset = amp_data(df_train)
    val_dataset = amp_data(df_val)

    model = ESM2WithCustomHead(
        model=backbone_model,
        num_labels=2,
        dropout=args.dropout
    ).to(device)

    fold_output_dir = os.path.join(args.checkpoints_dir, f"fold_{fold + 1}")
    os.makedirs(fold_output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=fold_output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir=os.path.join(fold_output_dir, "logs"),
        report_to="tensorboard",
        fp16=True,
        logging_strategy="epoch",
        save_total_limit=1,
        label_smoothing_factor=args.smoothing,
        save_safetensors=False,
        seed=fold
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=custom_collate_fn,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()
    trainer.evaluate()
    fold_model_paths.append(fold_output_dir)
