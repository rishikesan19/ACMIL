import random

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import os
import json
import pandas as pd


class HDF5_feat_dataset2(data.Dataset):
    def __init__(self, split, keys=None):
        self.split = split
        self.keys = list(split.keys()) if keys is None else keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        name = self.keys[index]
        item = self.split[name]

        # numpy -> torch
        x = torch.from_numpy(item['input']).float()   # [N, D] or [D, N], whichever you store
        y = torch.tensor(int(item['label'])).long()
        coords = torch.from_numpy(item['coords']).long()

        # return a dict so Step3 can do data['input'] / data['label']
        return {
            'input': x,
            'label': y,
            'coords': coords,
            'name': name,
        }



# =============================
# Existing dataset split helpers
# =============================

def split_dataset_camelyon(file_path, conf):
    h5_data = h5py.File(file_path, 'r')
    split_file_path = './splits/%s/split_%s.json'%(conf.dataset, conf.seed)
    if os.path.exists(split_file_path):
        with open(split_file_path, 'r') as json_file:
            data = json.load(json_file)
        train_names, val_names, test_names = data['train_names'], data['val_names'], data['test_names']
    else:
        slide_names = list(h5_data.keys())
        train_val_names, test_names = [], []
        for name in slide_names:
            if 'test' in name:
                test_names.append(name)
            else:
                train_val_names.append(name)
        from sklearn.model_selection import train_test_split
        labels = []
        for name in train_val_names:
            slide = h5_data[name]
            if 'label' in slide.attrs:
                labels.append(int(slide.attrs['label']))
            else:
                # Fallback: assume tumor in name
                labels.append(1 if 'tumor' in name.lower() else 0)
        train_names, val_names = train_test_split(train_val_names, test_size=0.1, random_state=conf.seed, stratify=labels)
        os.makedirs('./splits/%s'%conf.dataset, exist_ok=True)
        with open(split_file_path, 'w') as json_file:
            json.dump({'train_names':train_names, 'val_names':val_names, 'test_names':test_names}, json_file)

    train_split, val_split, test_split = {}, {}, {}
    for names, split in [(train_names, train_split), (val_names, val_split), (test_names, test_split)]:
        for slide_id in names:
            slide = h5_data[slide_id]
            label = int(slide.attrs['label']) if 'label' in slide.attrs else (1 if 'tumor' in slide_id.lower() else 0)
            feat = slide['feat'][:]
            coords = slide['coords'][:]
            split[slide_id] = {'input': feat, 'coords': coords, 'label': label}
    h5_data.close()
    return train_split, train_names, val_split, val_names, test_split, test_names


def split_dataset_bracs(file_path, conf):
    h5_data = h5py.File(file_path, 'r')
    split_file_path = './splits/%s/split_%s.json'%(conf.dataset, conf.seed)
    if os.path.exists(split_file_path):
        with open(split_file_path, 'r') as json_file:
            data = json.load(json_file)
        train_names, val_names, test_names = data['train_names'], data['val_names'], data['test_names']
    else:
        slide_names = list(h5_data.keys())
        labels = [int(h5_data[name].attrs['label']) for name in slide_names]
        from sklearn.model_selection import train_test_split
        train_names, temp_names, y_train, y_temp = train_test_split(slide_names, labels, test_size=0.28, random_state=conf.seed, stratify=labels)
        val_names, test_names, _, _ = train_test_split(temp_names, y_temp, test_size=87/152, random_state=conf.seed, stratify=y_temp)
        os.makedirs('./splits/%s'%conf.dataset, exist_ok=True)
        with open(split_file_path, 'w') as json_file:
            json.dump({'train_names':train_names, 'val_names':val_names, 'test_names':test_names}, json_file)
    train_split, val_split, test_split = {}, {}, {}
    for names, split in [(train_names, train_split), (val_names, val_split), (test_names, test_split)]:
        for slide_id in names:
            slide = h5_data[slide_id]
            label = int(slide.attrs['label'])
            feat = slide['feat'][:]
            coords = slide['coords'][:]
            split[slide_id] = {'input': feat, 'coords': coords, 'label': label}
    h5_data.close()
    return train_split, train_names, val_split, val_names, test_split, test_names


def split_dataset_lct(file_path, conf):
    h5_data = h5py.File(file_path, 'r')
    slide_names = list(h5_data.keys())
    labels = [int(h5_data[name].attrs['label']) for name in slide_names]
    from sklearn.model_selection import train_test_split
    train_names, temp_names, y_train, y_temp = train_test_split(slide_names, labels, test_size=0.4, random_state=conf.seed, stratify=labels)
    val_names, test_names, _, _ = train_test_split(temp_names, y_temp, test_size=0.5, random_state=conf.seed, stratify=y_temp)
    train_split, val_split, test_split = {}, {}, {}
    for names, split in [(train_names, train_split), (val_names, val_split), (test_names, test_split)]:
        for slide_id in names:
            slide = h5_data[slide_id]
            label = int(slide.attrs['label'])
            feat = slide['feat'][:]
            coords = slide['coords'][:]
            split[slide_id] = {'input': feat, 'coords': coords, 'label': label}
    h5_data.close()
    return train_split, train_names, val_split, val_names, test_split, test_names


def split_dataset_tcga_brca(file_path, conf):
    """Default stratified patient-level split using a master CSV, kept for backward compatibility."""
    # Expect: ./dataset_csv/brca_idc_ilc_pure.csv with slide_id,label
    master_csv = getattr(conf, 'tcga_master_csv', './dataset_csv/brca_idc_ilc_pure.csv')
    ref = pd.read_csv(master_csv)
    if ref['label'].dtype == object:
        st = ref['label'].astype(str).str.strip().str.upper()
        mapping = {'IDC':0,'ILC':1,'1':1,'0':0}
        ref['label'] = st.map(mapping)
    # Build groups by patient
    ref['patient'] = ref['slide_id'].str.split('-').str[:3].str.join('-')
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=conf.seed)
    (train_val_idx, test_idx), = gss.split(ref, groups=ref['patient'])
    train_val = ref.iloc[train_val_idx].reset_index(drop=True)
    test = ref.iloc[test_idx].reset_index(drop=True)
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=conf.seed)
    (tr_idx, va_idx), = sss.split(train_val, train_val['label'])
    tr = train_val.iloc[tr_idx].reset_index(drop=True)
    va = train_val.iloc[va_idx].reset_index(drop=True)

    h5_data = h5py.File(file_path, 'r')
    def build(df):
        split = {}
        names = []
        for sid, lab in zip(df['slide_id'].tolist(), df['label'].astype(int).tolist()):
            key = sid if sid in h5_data else sid.replace('.svs','')
            slide = h5_data[key]
            feat = slide['feat'][:]
            coords = slide['coords'][:]
            split[key] = {'input': feat, 'coords': coords, 'label': lab}
            names.append(key)
        return split, names
    train_split, train_names = build(tr)
    val_split,   val_names   = build(va)
    test_split,  test_names  = build(test)
    h5_data.close()
    return train_split, train_names, val_split, val_names, test_split, test_names


# === Fold-aware helpers for TCGA-BRCA and CAMELYON16 ===

def _read_fold_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if 'slide_id' not in df.columns:
        raise ValueError(f"{csv_path} must contain a 'slide_id' column")
    # If label exists, normalize object labels if needed
    if 'label' in df.columns and df['label'].dtype == object:
        st = df['label'].astype(str).str.strip().str.upper()
        mapping = {'IDC':0,'ILC':1,'TUMOR':1,'NORMAL':0,'METASTASIS':1,'POSITIVE':1,'NEGATIVE':0,'1':1,'0':0}
        df['label'] = st.map(mapping)
    return df


def _attach_labels_from_master(df: pd.DataFrame, master_csv: str) -> pd.DataFrame:
    if 'label' in df.columns:
        if df['label'].dtype == object:
            st = df['label'].astype(str).str.strip().str.upper()
            mapping = {'IDC':0,'ILC':1,'TUMOR':1,'NORMAL':0,'METASTASIS':1,'POSITIVE':1,'NEGATIVE':0,'1':1,'0':0}
            df['label'] = st.map(mapping)
        return df
    ref = pd.read_csv(master_csv)
    if 'slide_id' not in ref.columns or 'label' not in ref.columns:
        raise ValueError(f"Master CSV {master_csv} must contain 'slide_id' and 'label' columns")
    if ref['label'].dtype == object:
        st = ref['label'].astype(str).str.strip().str.upper()
        mapping = {'IDC':0,'ILC':1,'TUMOR':1,'NORMAL':0,'METASTASIS':1,'POSITIVE':1,'NEGATIVE':0,'1':1,'0':0}
        ref['label'] = st.map(mapping)
    merged = df.merge(ref[['slide_id','label']], on='slide_id', how='left', validate='one_to_one')
    if merged['label'].isna().any():
        missing = merged.loc[merged['label'].isna(), 'slide_id'].tolist()
        raise ValueError(f"Missing labels for {len(missing)} slides from master CSV (e.g., {missing[:5]})")
    return merged


def _build_split_from_csv_list(h5_data, df: pd.DataFrame):
    split_dict = {}
    names = []
    missing = []
    for sid, lab in zip(df['slide_id'].tolist(), df['label'].astype(int).tolist()):
        if sid not in h5_data:
            # Try sid without extension fallback
            alt = sid.replace('.svs','')
            if alt in h5_data:
                sid = alt
            else:
                missing.append(sid)
                continue
        slide = h5_data[sid]
        feat = slide['feat'][:]
        coords = slide['coords'][:]
        split_dict[sid] = {'input': feat, 'coords': coords, 'label': lab}
        names.append(sid)
    if missing:
        raise KeyError(f"{len(missing)} slide_id(s) not found in H5 (e.g., {missing[:5]})")
    return split_dict, names


def split_dataset_tcga_brca_from_folds(h5_file_path: str, cv_root: str, fold: int, conf):
    fold_dir = os.path.join(cv_root, f'fold_{fold}')
    train_csv = os.path.join(fold_dir, 'train.csv')
    val_csv   = os.path.join(fold_dir, 'val.csv')
    test_csv  = os.path.join(fold_dir, 'test.csv')
    df_tr = _read_fold_csv(train_csv)
    df_va = _read_fold_csv(val_csv)
    df_te = _read_fold_csv(test_csv)
    master_csv = getattr(conf, 'tcga_master_csv', None)
    if master_csv is not None:
        df_tr = _attach_labels_from_master(df_tr, master_csv)
        df_va = _attach_labels_from_master(df_va, master_csv)
        df_te = _attach_labels_from_master(df_te, master_csv)
    # Ensure label present after attach
    for tag, d in [('train',df_tr),('val',df_va),('test',df_te)]:
        if 'label' not in d.columns:
            raise ValueError(f"{tag}.csv at {fold_dir} must have a 'label' column or provide conf.tcga_master_csv")
    with h5py.File(h5_file_path, 'r') as h5_data:
        train_split, train_names = _build_split_from_csv_list(h5_data, df_tr)
        val_split,   val_names   = _build_split_from_csv_list(h5_data, df_va)
        test_split,  test_names  = _build_split_from_csv_list(h5_data, df_te)
    # sanity no overlaps
    st, sv, se = set(train_names), set(val_names), set(test_names)
    if (st & sv) or (st & se) or (sv & se):
        raise ValueError("Slide IDs overlap across splits within the fold.")
    return train_split, train_names, val_split, val_names, test_split, test_names


def split_dataset_camelyon16_from_folds(h5_file_path: str, cv_root: str, fold: int, conf):
    fold_dir = os.path.join(cv_root, f'fold_{fold}')
    train_csv = os.path.join(fold_dir, 'train.csv')
    val_csv   = os.path.join(fold_dir, 'val.csv')
    test_csv  = os.path.join(fold_dir, 'test.csv')
    df_tr = _read_fold_csv(train_csv)
    df_va = _read_fold_csv(val_csv)
    df_te = _read_fold_csv(test_csv)
    # Camelyon16 fold CSVs already include labels; enforce presence
    for tag, d in [('train',df_tr),('val',df_va),('test',df_te)]:
        if 'label' not in d.columns:
            raise ValueError(f"{tag}.csv at {fold_dir} must include 'label' for CAMELYON16 folds")
    with h5py.File(h5_file_path, 'r') as h5_data:
        train_split, train_names = _build_split_from_csv_list(h5_data, df_tr)
        val_split,   val_names   = _build_split_from_csv_list(h5_data, df_va)
        test_split,  test_names  = _build_split_from_csv_list(h5_data, df_te)
    # sanity
    st, sv, se = set(train_names), set(val_names), set(test_names)
    if (st & sv) or (st & se) or (sv & se):
        raise ValueError("Slide IDs overlap across splits within the fold.")
    return train_split, train_names, val_split, val_names, test_split, test_names


# =============================
# Utility: few-shot sub-sampling
# =============================

def generate_fewshot_dataset(split_dict, names, num_shots=0):
    if num_shots is None or num_shots <= 0:
        return split_dict, names
    # Subsample per class
    by_cls = {0: [], 1: []}
    for name in names:
        by_cls[split_dict[name]['label']].append(name)
    new_names = []
    for c in by_cls:
        if len(by_cls[c]) <= num_shots:
            new_names.extend(by_cls[c])
        else:
            new_names.extend(random.sample(by_cls[c], num_shots))
    new_split = {k: split_dict[k] for k in new_names}
    return new_split, new_names


# =============================
# Dataset builder
# =============================

def build_HDF5_feat_dataset(file_path, conf):
    if conf.dataset == 'camelyon':
        cv_root = getattr(conf, 'cv_root', None)
        fold    = getattr(conf, 'fold', None)
        if cv_root is not None and fold is not None:
            train_split, train_names, val_split, val_names, test_split, test_names = split_dataset_camelyon16_from_folds(file_path, cv_root, int(fold), conf)
            train_split, train_names = generate_fewshot_dataset(train_split, train_names, num_shots=conf.n_shot)
            return HDF5_feat_dataset2(train_split, train_names), HDF5_feat_dataset2(val_split, val_names), HDF5_feat_dataset2(test_split, test_names)
        train_split, train_names, val_split, val_names, test_split, test_names = split_dataset_camelyon(file_path, conf)
        train_split, train_names = generate_fewshot_dataset(train_split, train_names, num_shots=conf.n_shot)
        return HDF5_feat_dataset2(train_split, train_names), HDF5_feat_dataset2(val_split, val_names), HDF5_feat_dataset2(test_split, test_names)
    elif conf.dataset == 'bracs':
        train_split, train_names, val_split, val_names, test_split, test_names = split_dataset_bracs(file_path, conf)
        train_split, train_names = generate_fewshot_dataset(train_split, train_names, num_shots=conf.n_shot)
        return HDF5_feat_dataset2(train_split, train_names), HDF5_feat_dataset2(val_split, val_names), HDF5_feat_dataset2(test_split, test_names)
    elif conf.dataset == 'lct':
        train_split, train_names, val_split, val_names, test_split, test_names = split_dataset_lct(file_path, conf)
        train_split, train_names = generate_fewshot_dataset(train_split, train_names, num_shots=conf.n_shot)
        return HDF5_feat_dataset2(train_split, train_names), HDF5_feat_dataset2(val_split, val_names), HDF5_feat_dataset2(test_split, test_names)
    elif conf.dataset == 'tcga_brca':
        cv_root = getattr(conf, 'cv_root', None)
        fold    = getattr(conf, 'fold', None)
        if cv_root is not None and fold is not None:
            train_split, train_names, val_split, val_names, test_split, test_names = split_dataset_tcga_brca_from_folds(file_path, cv_root, int(fold), conf)
            train_split, train_names = generate_fewshot_dataset(train_split, train_names, num_shots=conf.n_shot)
            return HDF5_feat_dataset2(train_split, train_names), HDF5_feat_dataset2(val_split, val_names), HDF5_feat_dataset2(test_split, test_names)
        train_split, train_names, val_split, val_names, test_split, test_names = split_dataset_tcga_brca(file_path, conf)
        train_split, train_names = generate_fewshot_dataset(train_split, train_names, num_shots=conf.n_shot)
        return HDF5_feat_dataset2(train_split, train_names), HDF5_feat_dataset2(val_split, val_names), HDF5_feat_dataset2(test_split, test_names)


if __name__ == '__main__':
    # Example: split LCT if needed
    pass
