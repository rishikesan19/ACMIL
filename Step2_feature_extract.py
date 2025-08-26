import torch
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import numpy as np
import pandas as pd # Use pandas for easier CSV handling
from utils.utils import collate_features
import time
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
import argparse
from models import build_model
import h5py
import openslide
import yaml
from utils.utils import Struct
import glob

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Extract Features of Patches')
parser.add_argument('--data_h5_dir', type=str, help='Directory where patch coordinate h5 files are stored')
parser.add_argument('--csv_path', type=str, help='Path to the CSV file mapping slides')
parser.add_argument('--slide_ext', type=str, default='.svs')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=224)
parser.add_argument('--config', dest='config', help='settings in yaml format')
args = parser.parse_args()


def extract_feature(file_path, wsi, model, batch_size=256, custom_downsample=1, target_patch_size=-1):
    """
    Extracts features from a single slide.
    """
    dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=True,
                                 custom_downsample=custom_downsample, target_patch_size=target_patch_size)

    loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=8, collate_fn=collate_features)
    
    if len(loader) == 0:
        return None, None

    feature_list = []
    coord_list = []
    for count, (batch, coords) in enumerate(loader):
        with torch.no_grad():
            batch = batch.to(device, dtype=torch.float32)
            _, feature = model(batch, return_feature=True)
            feature_list.append(feature.cpu())
            coord_list.append(coords)
            
    features = torch.cat(feature_list, dim=0)
    coords = np.concatenate(coord_list, axis=0)
    return features.numpy(), coords


if __name__ == '__main__':
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    cfg = Struct(**cfg)
    print("\nRunning configs.")
    print(cfg, "\n")
    
    # Define output directory and file path
    output_dir = args.data_h5_dir # Save features in the same dir as patch coords
    os.makedirs(output_dir, exist_ok=True)

    print('initializing dataset from:', args.csv_path)
    try:
        slide_data = pd.read_csv(args.csv_path)
    except FileNotFoundError:
        print(f"Error: The file {args.csv_path} was not found.")
        exit()

    print('loading model checkpoint')
    model = build_model(cfg)
    model = model.to(device)
    model.eval()
    
    safe_pretrain = str(cfg.pretrain).replace('/', '_').replace('\\', '_')
    safe_backbone = str(cfg.backbone).replace('/', '_').replace('\\', '_')
    feature_filename = f"patch_feats_pretrain_{safe_pretrain}_{safe_backbone}.h5"
    feature_h5_path = os.path.join(output_dir, feature_filename)
    print(f"Will save features to: {feature_h5_path}")

    h5file = h5py.File(feature_h5_path, "w")

    total = len(slide_data)
    for index, row in slide_data.iterrows():
        # Read IDs and paths directly from the new CSV columns
        slide_id = row['slide_id']      # The long ID with UUID, e.g., TCGA-XX.YYYY
        # slide_id1 = row['slide_id1']    # The short ID, e.g., TCGA-XX
        label = row['label']
        slide_file_path = row['full_path'] # The full path to the .svs file
        
        # Construct the path to the patch coordinate file using the long slide_id
        # This assumes patch files are named like 'LONG_ID.h5'
        patch_file_path = os.path.join(args.data_h5_dir, 'patches', f"{slide_id}.h5")

        print(f'\nprogress: {index+1}/{total}')
        print(f"Processing Slide ID: {slide_id}")

        if not os.path.exists(patch_file_path):
            print(f"--> SKIPPING: Patch coordinate file not found at {patch_file_path}")
            continue

        if not os.path.exists(slide_file_path):
            print(f"--> SKIPPING: SVS image file not found at {slide_file_path}")
            continue
            
        time_start = time.time()
        try:
            wsi = openslide.open_slide(slide_file_path)
            features, coords = extract_feature(patch_file_path, wsi, model,
                                               batch_size=args.batch_size,
                                               custom_downsample=args.custom_downsample,
                                               target_patch_size=args.target_patch_size)
        except Exception as e:
            print(f"--> ERROR processing slide {slide_id}: {e}")
            continue

        if features is None:
            print(f"--> SKIPPING: No processable patches found in {patch_file_path}")
            continue

        # Save features to the HDF5 file using the long slide_id as the group name
        slide_grp = h5file.create_group(slide_id)
        slide_grp.create_dataset('feat', data=features.astype(np.float32))
        slide_grp.create_dataset('coords', data=coords)
        slide_grp.attrs['label'] = label
        # slide_grp.attrs['short_id'] = slide_id1

        time_elapsed = time.time() - time_start
        print(f'Computing features for {slide_id} took {time_elapsed:.2f} s')

    h5file.close()
    print("\nStored features successfully!")