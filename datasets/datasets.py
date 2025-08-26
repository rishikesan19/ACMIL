import os
import json
import random
from typing import Dict, List

import h5py
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

# ----------------------------
# Dataset objects
# ----------------------------
class HDF5_feat_dataset2(data.Dataset):
    """
    Returns a dict: {'input': FloatTensor, 'label': LongTensor, 'coords': LongTensor, 'name': str}
    """
    def __init__(self, split: Dict[str, dict], keys: List[str] = None):
        self.split = split
        self.keys = list(split.keys()) if keys is None else keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        name = self.keys[index]
        item = self.split[name]
        x = torch.from_numpy(item['input']).float()
        y = torch.tensor(int(item['label'])).long()
        coords = torch.from_numpy(item['coords']).long() if isinstance(item['coords'], np.ndarray) else torch.as_tensor(item['coords'])
        return {'input': x, 'label': y, 'coords': coords, 'name': name}

# ----------------------------
# Dataset-specific split functions (Restored from your original script)
# ----------------------------
def split_dataset_camelyon(file_path, conf):
    h5_data = h5py.File(file_path, 'r')
    split_file_path = f'./splits/{conf.dataset}/split_{conf.seed}.json'
    if os.path.exists(split_file_path):
        with open(split_file_path, 'r') as json_file:
            data = json.load(json_file)
        train_names, val_names, test_names = data['train_names'], data['val_names'], data['test_names']
    else:
        slide_names = list(h5_data.keys())
        train_val_names, test_names = [], []
        for name in slide_names:
            if 'test' in name.lower():
                test_names.append(name)
            else:
                train_val_names.append(name)
        labels = []
        for name in train_val_names:
            slide = h5_data[name]
            if 'label' in slide.attrs:
                labels.append(int(slide.attrs['label']))
            else:
                labels.append(1 if 'tumor' in name.lower() else 0)
        from sklearn.model_selection import train_test_split
        train_names, val_names = train_test_split(train_val_names, test_size=0.1, random_state=conf.seed, stratify=labels)
        os.makedirs(f'./splits/{conf.dataset}', exist_ok=True)
        with open(split_file_path, 'w') as json_file:
            json.dump({'train_names': train_names, 'val_names': val_names, 'test_names': test_names}, json_file)

    train_split, val_split, test_split = {}, {}, {}
    for names, split in [(train_names, train_split), (val_names, val_split), (test_names, test_split)]:
        for name in names:
            slide = h5_data[name]
            label = int(slide.attrs['label']) if 'label' in slide.attrs else (1 if 'tumor' in name.lower() else 0)
            feat = slide['feat'][:]
            coords = slide['coords'][:]
            split[name] = {'input': feat, 'coords': coords, 'label': label}
    h5_data.close()
    return train_split, train_names, val_split, val_names, test_split, test_names

def split_dataset_bracs(file_path, conf):
    csv_path = './dataset_csv/bracs.csv'
    slide_info = pd.read_csv(csv_path).set_index('slide_id')
    class_transfer_dict_3class = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
    class_transfer_dict_2class = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1}
    h5_data = h5py.File(file_path, 'r')
    train_split, val_split, test_split = {}, {}, {}
    train_names, val_names, test_names = [], [], []
    for slide_id in list(h5_data.keys()):
        slide = h5_data[slide_id]
        raw_label = int(slide.attrs['label'])
        if getattr(conf, 'n_class', 3) == 3: label = class_transfer_dict_3class[raw_label]
        elif getattr(conf, 'n_class', 3) == 2: label = class_transfer_dict_2class[raw_label]
        else: label = raw_label
        item = {'input': slide['feat'][:], 'coords': slide['coords'][:], 'label': int(label)}
        split_info = slide_info.loc[slide_id]['split_info']
        if split_info == 'train': train_names.append(slide_id); train_split[slide_id] = item
        elif split_info == 'val': val_names.append(slide_id); val_split[slide_id] = item
        else: test_names.append(slide_id); test_split[slide_id] = item
    h5_data.close()
    return train_split, train_names, val_split, val_names, test_split, test_names

def split_dataset_tcga_brca(file_path, conf):
    master_csv = getattr(conf, 'tcga_master_csv', './dataset_csv/brca_idc_ilc_pure.csv')
    ref = pd.read_csv(master_csv)
    if ref['label'].dtype == object:
        ref['label'] = ref['label'].astype(str).str.strip().str.upper().map({'IDC': 0, 'ILC': 1, '0': 0, '1': 1})
    ref['label'] = ref['label'].astype(int)
    ref['patient'] = ref['patient_id'].astype(str) if 'patient_id' in ref.columns else ref['slide_id'].str.split('-').str[:3].str.join('-')
    from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=conf.seed)
    (train_val_idx, test_idx), = gss.split(ref, groups=ref['patient'])
    train_val, test = ref.iloc[train_val_idx], ref.iloc[test_idx]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=conf.seed)
    (tr_idx, va_idx), = sss.split(train_val, train_val['label'])
    tr, va = train_val.iloc[tr_idx], train_val.iloc[va_idx]
    with h5py.File(file_path, 'r') as h5_data:
        def build(df):
            split, names = {}, []
            for _, row in df.iterrows():
                key = _find_h5_key(row['slide_id'], h5_data)
                if key:
                    split[key] = {'input': h5_data[key]['feat'][:], 'coords': h5_data[key]['coords'][:], 'label': int(row['label'])}
                    names.append(key)
            return split, names
        train_split, train_names = build(tr)
        val_split, val_names = build(va)
        test_split, test_names = build(test)
    return train_split, train_names, val_split, val_names, test_split, test_names

# ----------------------------
# Fold-aware helpers (Restored and Fixed)
# ----------------------------
def _find_fold_or_split_dir(cv_root: str, number: int) -> str:
    fold_dir = os.path.join(cv_root, f'fold_{number}')
    split_dir = os.path.join(cv_root, f'split_{number}')
    if os.path.isdir(fold_dir): return fold_dir
    if os.path.isdir(split_dir): return split_dir
    raise FileNotFoundError(f"Directory for fold/split {number} not found in {cv_root}. Checked for 'fold_{number}' and 'split_{number}'.")

def _find_h5_key(slide_id: str, h5_file: h5py.File) -> str or None:
    """Finds the correct key in an H5 file for a given slide_id using multiple strategies."""
    if slide_id in h5_file: return slide_id
    key_no_ext = slide_id.replace('.svs', '')
    if key_no_ext in h5_file: return key_no_ext
    key_prefix = slide_id.split('.')[0]
    if key_prefix in h5_file: return key_prefix
    key_splitext = os.path.splitext(slide_id)[0]
    if key_splitext in h5_file: return key_splitext
    return None

def _read_fold_csv(csv_path: str) -> pd.DataFrame:
    """Reads a CSV and normalizes the label column if it exists."""
    df = pd.read_csv(csv_path)
    if 'slide_id' not in df.columns:
        raise ValueError(f"{csv_path} must contain a 'slide_id' column")
    if 'label' in df.columns and df['label'].dtype == object:
        st = df['label'].astype(str).str.strip().str.upper()
        mapping = {'IDC': 0, 'ILC': 1, 'TUMOR': 1, 'NORMAL': 0, 'METASTASIS': 1, 'POSITIVE': 1, 'NEGATIVE': 0, '1': 1, '0': 0}
        df['label'] = st.map(mapping)
    return df

def _build_split_from_df(h5_data, df: pd.DataFrame):
    split_dict, names, missing = {}, [], []
    for _, row in df.iterrows():
        sid, lab = row['slide_id'], row['label']
        key = _find_h5_key(sid, h5_data)
        if key is None:
            missing.append(sid)
            continue
        # This is where the error happened: 'lab' must be an integer here.
        split_dict[key] = {'input': h5_data[key]['feat'][:], 'coords': h5_data[key]['coords'][:], 'label': int(lab)}
        names.append(key)
    if missing:
        raise KeyError(f"{len(missing)} slide_ids from CSV not in H5 file (e.g., {missing[:5]})")
    return split_dict, names

def get_splits_from_fold_files(h5_file_path: str, cv_root: str, split_num: int):
    split_dir = _find_fold_or_split_dir(cv_root, split_num)
    # --- FIX: Use the _read_fold_csv helper to normalize labels immediately after reading ---
    df_tr = _read_fold_csv(os.path.join(split_dir, 'train.csv'))
    df_va = _read_fold_csv(os.path.join(split_dir, 'val.csv'))
    df_te = _read_fold_csv(os.path.join(split_dir, 'test.csv'))
    
    with h5py.File(h5_file_path, 'r') as h5_data:
        train_split, train_names = _build_split_from_df(h5_data, df_tr)
        val_split, val_names = _build_split_from_df(h5_data, df_va)
        test_split, test_names = _build_split_from_df(h5_data, df_te)
    if set(train_names) & set(val_names) or set(train_names) & set(test_names) or set(val_names) & set(test_names):
        raise ValueError("Slide IDs overlap between train/val/test splits.")
    return train_split, train_names, val_split, val_names, test_split, test_names

# ----------------------------
# Few-shot subsampling
# ----------------------------
def generate_fewshot_dataset(split_dict, names, num_shots=0):
    if not num_shots or num_shots <= 0: return split_dict, names
    by_cls = {}
    for name in names: by_cls.setdefault(int(split_dict[name]['label']), []).append(name)
    new_names = [s for c, items in by_cls.items() for s in random.sample(items, min(len(items), num_shots))]
    return {k: split_dict[k] for k in new_names}, new_names

# ----------------------------
# Main Entry Point
# ----------------------------
def build_HDF5_feat_dataset(file_path, conf):
    cv_root = getattr(conf, 'cv_root', None)
    split_num = getattr(conf, 'fold', None) or getattr(conf, 'split', None)

    if cv_root and split_num is not None:
        print(f"--> Loading from CV folds/splits: root={cv_root}, number={split_num}")
        train_split, train_names, val_split, val_names, test_split, test_names = \
            get_splits_from_fold_files(file_path, cv_root, int(split_num))
    else:
        print(f"--> No folds/splits specified. Using fallback logic for dataset: '{conf.dataset}'")
        if conf.dataset == 'tcga_brca':
            train_split, train_names, val_split, val_names, test_split, test_names = split_dataset_tcga_brca(file_path, conf)
        elif conf.dataset == 'camelyon':
            train_split, train_names, val_split, val_names, test_split, test_names = split_dataset_camelyon(file_path, conf)
        elif conf.dataset == 'bracs':
            train_split, train_names, val_split, val_names, test_split, test_names = split_dataset_bracs(file_path, conf)
        else:
            raise ValueError(f"Unknown dataset for fallback: '{conf.dataset}'")

    num_shots = getattr(conf, 'n_shot', 0)
    if num_shots > 0:
        print(f"Applying few-shot learning with {num_shots} shots.")
        train_split, train_names = generate_fewshot_dataset(train_split, train_names, num_shots)

    return (
        HDF5_feat_dataset2(train_split, train_names),
        HDF5_feat_dataset2(val_split, val_names),
        HDF5_feat_dataset2(test_split, test_names)
    )