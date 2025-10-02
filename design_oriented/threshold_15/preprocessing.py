import os
import glob
import subprocess
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.model_selection import train_test_split
from modlamp.descriptors import GlobalDescriptor

threshold = 0.8
new_df = pd.read_csv("../../data/all_C_amidated_thres_15.csv")
print(new_df['target'].count())

with open('feature_cal.fasta', 'w') as fasta_file:
    for _, row in new_df.iterrows():
        name = row['Name']
        sequence = row['sequence']
        fasta_file.write(f'>{name}\n{sequence}\n')

print("Done!")

os.makedirs('feature', exist_ok=True)
for feature_type in ['AAC','CKSAAP','DPC','TPC','GAAC','CKSAAGP','GDPC','GTPC','AAINDEX','ZSCALE','BLOSUM62','NMBroto','Moran','Geary','CTDC','CTDT','CTDD','CTriad','KSCTriad','SOCNumber','QSOrder','PAAC','APAAC']:
    output_file = f"feature/{feature_type}.tsv"
    subprocess.run(['python', '../iFeature/iFeature.py', '--file', 'feature_cal.fasta', '--type', feature_type, '--out', output_file], check=True)

folder_path = 'feature'
file_paths = glob.glob(os.path.join(folder_path, '*.tsv'))

for file_path in file_paths:
    try:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        df = pd.read_csv(file_path, sep='\t', header=0)
        df.columns = [f"{file_name}_{col}" for col in df.columns]
        df.to_csv(file_path, sep='\t', index=False)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

file_paths = glob.glob('feature/*.tsv')
fixed_first_column_name = 'Name'
tabular_combined = pd.DataFrame()

for file_path in file_paths:
    try:
        tabular = pd.read_csv(file_path, sep='\t', header=None, low_memory=False)
        if tabular.empty or len(tabular.columns) < 2:
            continue
        tabular.columns = [fixed_first_column_name] + list(tabular.iloc[0, 1:])
        tabular = tabular[1:].reset_index(drop=True)
        if tabular_combined.empty:
            tabular_combined = tabular
        else:
            tabular_combined = pd.merge(tabular_combined, tabular, on=fixed_first_column_name)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

if not tabular_combined.empty:
    tabular_combined.to_csv('all_cal_feature.tsv', sep='\t', index=False)

pep = new_df['sequence'].tolist()
desc = GlobalDescriptor(pep)
desc.calculate_all(amide=True)
global_desc = pd.DataFrame(desc.descriptor, columns=desc.featurenames)
global_desc.insert(0, 'Name', new_df['Name'])

tabular_ifeature = pd.read_csv('all_cal_feature.tsv', sep='\t', low_memory=False).filter(
    regex=r'^(Name|AAC|GAAC|DPC|TPC|CTD|CKSAAGP)', axis=1
)
tabular_modlamp = global_desc
tabular_df = tabular_ifeature.merge(tabular_modlamp, on="Name", how="inner")

target_df = new_df[['Name', 'target']]
features_df = pd.merge(tabular_df, target_df, on='Name').reset_index(drop=True)
model_df = features_df.fillna(0)

def remove_semi_constant_features(df, exclude_columns=None, threshold=0.8):
    if exclude_columns is None:
        exclude_columns = []
    columns_to_drop = [
        col for col in df.columns if col not in exclude_columns
        and df[col].value_counts(normalize=True).iloc[0] >= threshold
    ]
    return df.drop(columns=columns_to_drop, inplace=False)

def remove_highly_correlated(df: pd.DataFrame, threshold: float = 0.9, keep: list[str] = None):
    if keep is None:
        keep = []
    feature_cols = sorted([col for col in df.columns if col not in keep])
    df_features = df[feature_cols]
    corr_matrix = df_features.corr(numeric_only=True).abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = []
    for col in sorted(upper.columns):
        for row in sorted(upper.index):
            if upper.loc[row, col] > threshold:
                if row in keep:
                    if col not in to_drop:
                        to_drop.append(col)
                elif col in keep:
                    if row not in to_drop:
                        to_drop.append(row)
                else:
                    if col not in to_drop:
                        to_drop.append(col)
    reduced_df = df.drop(columns=to_drop, errors="ignore")
    return reduced_df

feature_columns = model_df.columns.difference(['Name', 'target'])

base_groups = {
    "AAC":       [col for col in feature_columns if col.startswith("AAC")],
    "GAAC":      [col for col in feature_columns if col.startswith("GAAC")],
    "DPC":       [col for col in feature_columns if col.startswith("DPC")],
    "TPC":       [col for col in feature_columns if col.startswith("TPC")],
    "CKSAAGP":   [col for col in feature_columns if col.startswith("CKSAAGP")],
    "CTD":       [col for col in feature_columns if col.startswith("CTD")],
    "global":    [col for col in feature_columns if "_" not in col],
}

base_groups["composition"] = (
    base_groups["AAC"] + base_groups["GAAC"] + base_groups["DPC"] +
    base_groups["TPC"] + base_groups["CKSAAGP"]
)

combo_keys = ["composition", "CTD", "global"]
group_combinations = {}
for r in range(1, len(combo_keys) + 1):
    for subset in combinations(combo_keys, r):
        name = "_".join(subset)
        cols = []
        for key in subset:
            cols.extend(base_groups[key])
        group_combinations[name] = cols

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(script_dir, "output")
os.makedirs(parent_dir, exist_ok=True)

for group_name, cols in group_combinations.items():
    df_0 = model_df[cols + ['Name', 'target']].copy()
    df_1 = remove_semi_constant_features(df_0, exclude_columns=['Name', 'target'], threshold=threshold)
    df_2 = remove_highly_correlated(df_1, keep=['Name', 'target'], threshold=threshold)

    X = df_2.drop(columns=['Name', 'target'])
    y = df_2['target']
    names = df_2['Name']

    X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
        X, y, names, test_size=0.2, stratify=y, random_state=100
    )

    group_dir = os.path.join(parent_dir, group_name)
    os.makedirs(group_dir, exist_ok=True)

    X_train.to_csv(f"{group_dir}/x_train_{group_name}.csv", index=False)
    X_test.to_csv(f"{group_dir}/x_test_{group_name}.csv", index=False)
    y_train.to_csv(f"{group_dir}/y_train_{group_name}.csv", index=False)
    y_test.to_csv(f"{group_dir}/y_test_{group_name}.csv", index=False)

    X_train_named = X_train.copy(); X_train_named["Name"] = names_train
    X_test_named = X_test.copy();   X_test_named["Name"] = names_test

    X_train_named.to_csv(f"{group_dir}/train_with_name_{group_name}.csv", index=False)
    X_test_named.to_csv(f"{group_dir}/test_with_name_{group_name}.csv", index=False)
