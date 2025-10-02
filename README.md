# C-Amidated AMP Classification Project

This repository includes two main types of models for antimicrobial peptide (AMP) classification:

- **Design-oriented model** (based on EBM - Explainable Boosting Machine)
- **Deployment model** (based on pretrained ESM2)

The data used in this project is located in the `data/` folder.

---

## Folder Structure

```
data/
├── all_C_amidated_thres_10.csv
├── all_C_amidated_thres_15.csv
├── model_set.csv
└── test_set.csv

deployment/
├── esm2_t6_8M_UR50D.pt
├── save_folds/
├── train_ESM2.py
└── inference_esm2_ensemble.py

design_oriented/
├── iFeature/
├── threshold_10/
└── threshold_15/
    ├── logs_sfs_ebm/
    ├── output/
    ├── plots_sfs_ebm/
    ├── preprocessing.py
    ├── sfs_ebm_group.py
    ├── save_best_ebm.py
    ├── interpret.ipynb
    └── ebm_model.pkl
```

---

## Design-Oriented Model (EBM)

- **Code**: `design_oriented/` 
- **Features**: interpretable (Composition, CTD, Global descriptors)
- **Selection**: via Sequential Feature Selector (SFS)
- **Model**: `ExplainableBoostingClassifier` from `interpret` package

### Example (run one trial for feature group, with a chosen threshold of 15)
```bash
python design_oriented/threshold_15/sfs_ebm_group.py --group composition_CTD_global
```

- There are 7 feature group combinations already stored in `output/`
- You can also regenerate feature sets with:
```bash
python design_oriented/threshold_15/preprocessing.py
```

- **Results** (metrics & plots): `logs_sfs_ebm/`, `plots_sfs_ebm/`
- **Saved Model**: `ebm_model.pkl`
- **Interpretability**: via `interpret.ipynb`

---

## Deployment Model (ESM2)

- **Code**: `deployment/`
- **Model**: pretrained transformer (ESM2) with custom classification head
- **Training**: 5-fold cross-validation with checkpoint saving

### Checkpoints & Ensemble

- `deployment/save_folds/`: stores the best model from each fold during 5-fold training.
- These saved models are later used for ensemble prediction in `inference_esm2_ensemble.py`.

### Train model
```bash
python deployment/train_ESM2.py --help
```

### Supported Arguments
```
usage: train_ESM2.py [-h]
                     [--batch_size BATCH_SIZE]
                     [--grad_accum_steps GRAD_ACCUM_STEPS]
                     [--epochs EPOCHS]
                     [--lr LR]
                     [--k_folds K_FOLDS]
                     [--weight_decay WEIGHT_DECAY]
                     [--smoothing SMOOTHING]
                     [--checkpoints_dir CHECKPOINTS_DIR]
                     [--dropout DROPOUT]

Train ESM2 model with k-fold cross-validation on AMP data.

optional arguments:
  -h, --help            Show this help message and exit
  --batch_size, -b      Batch size per device (default: 8)
  --grad_accum_steps, -g
                        Gradient accumulation steps (default: 8)
  --epochs, -e          Number of epochs (default: 50)
  --lr, -l              Learning rate (default: 1e-5)
  --k_folds, -k         Number of CV folds (default: 5)
  --weight_decay, -w    Weight decay (default: 0.01)
  --smoothing, -s       Label smoothing factor (default: 0.0)
  --checkpoints_dir, -c Directory to save model checkpoints
  --dropout, -d         Dropout rate (default: 0.1)
```

### Inference
Use the trained ensemble of models for inference on `test_set.csv`:
```bash
python deployment/inference_esm2_ensemble.py
```

### CAmidPred Web Tool

An easy-to-use, browser-based version of CAmidPred is freely available at: [https://huggingface.co/spaces/danghuyle/CAmidPred](https://huggingface.co/spaces/danghuyle/CAmidPred)


---

## Requirements

Install required packages:
```bash
pip install -r requirements.txt
```