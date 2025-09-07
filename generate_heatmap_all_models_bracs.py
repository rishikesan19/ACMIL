#!/usr/bin/env python3
"""
BRACS Heatmap Generator with Consolidated Visualization and WandB Logging
Complete version with all CAMELYON functionality
"""

import torch
import torch.nn as nn
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.ndimage import gaussian_filter, label, zoom
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score, average_precision_score
import seaborn as sns
import openslide
import os
import sys
import json
import logging
from PIL import Image, ImageDraw
import cv2
from datetime import datetime
import yaml
import gc
import wandb

# Add project paths
sys.path.append('/vol/research/scratch1/NOBACKUP/rk01337/ACMIL') 

from architecture.transformer import ACMIL_GA, ABMIL
from architecture.transMIL import TransMIL
from architecture.Attention import Attention_Gated as Attention, Attention_with_Classifier
from architecture.network import Classifier_1fc, DimReduction
import modules.mean_max as mean_max
from oodml.models import OODML
from utils.utils import Struct


class QuantitativeHeatmapAnalyzer:
    """Comprehensive quantitative analysis for MIL model attention heatmaps."""
    
    def __init__(self, output_dir='quantitative_results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def create_ground_truth_masks(self, annotations_by_class, image_shape):
        """Create binary masks for each annotation class."""
        masks = {}
        
        for class_name, regions in annotations_by_class.items():
            mask = np.zeros(image_shape[:2], dtype=np.uint8)
            for coords in regions:
                if len(coords) >= 3:
                    cv2.fillPoly(mask, [np.array(coords, dtype=np.int32)], 1)
            masks[class_name] = mask
        
        # Combined mask for any abnormal region
        masks['abnormal'] = np.logical_or(
            masks.get('malignant', np.zeros_like(mask)),
            masks.get('atypical', np.zeros_like(mask))
        ).astype(np.uint8)
        
        return masks
    
    def calculate_dice_coefficient(self, pred_mask, true_mask):
        """Calculate Dice similarity coefficient."""
        intersection = np.sum(pred_mask * true_mask)
        if np.sum(pred_mask) + np.sum(true_mask) == 0:
            return 1.0
        dice = 2.0 * intersection / (np.sum(pred_mask) + np.sum(true_mask))
        return dice
    
    def calculate_iou(self, pred_mask, true_mask):
        """Calculate Intersection over Union."""
        intersection = np.sum(pred_mask * true_mask)
        union = np.sum(np.logical_or(pred_mask, true_mask))
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        return intersection / union
    
    def calculate_attention_entropy(self, attention_map):
        """Calculate entropy of attention distribution."""
        attention_flat = attention_map.flatten()
        attention_flat = attention_flat[attention_flat > 0]
        
        if len(attention_flat) == 0:
            return 0.0
        
        attention_prob = attention_flat / np.sum(attention_flat)
        entropy = -np.sum(attention_prob * np.log(attention_prob + 1e-10))
        max_entropy = np.log(len(attention_prob))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        return normalized_entropy
    
    def calculate_roc_metrics(self, attention_map, ground_truth_mask):
        """Calculate ROC curve and optimal threshold."""
        y_true = ground_truth_mask.flatten()
        y_scores = attention_map.flatten()
        
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        return {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'optimal_threshold': optimal_threshold,
            'fpr': fpr,
            'tpr': tpr
        }


class BRACSHeatmapGenerator:
    """Main class for generating attention heatmaps for BRACS dataset."""
    
    def __init__(self, h5_path, slide_path, checkpoint_dir, annotation_path=None, 
                 output_dir='bracs_heatmap_results', device='cuda',
                 patch_size=256, extraction_level=1,
                 use_wandb=True, wandb_project='bracs-heatmap', wandb_run_name=None):
        self.h5_path = h5_path
        self.slide_path = slide_path
        self.checkpoint_dir = checkpoint_dir
        self.annotation_path = annotation_path
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.patch_size = patch_size
        self.extraction_level = extraction_level
        
        # Output directory setup
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(output_dir, f"run_{self.timestamp}")
        
        # Create directory structure
        self.dirs = {
            'overlays': os.path.join(self.run_dir, "wsi_overlays"),
            'heatmaps': os.path.join(self.run_dir, "heatmaps_only"),
            'comparisons': os.path.join(self.run_dir, "comparisons"),
            'annotations': os.path.join(self.run_dir, "ground_truth"),
            'data': os.path.join(self.run_dir, "attention_data"),
            'logs': os.path.join(self.run_dir, "logs"),
            'metrics': os.path.join(self.run_dir, "metrics"),
            'reports': os.path.join(self.run_dir, "reports"),
            'roc_curves': os.path.join(self.run_dir, "roc_curves"),
            'debug': os.path.join(self.run_dir, "debug")
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        self.setup_logging()
        
        # Initialize WandB
        self.use_wandb = use_wandb
        if self.use_wandb:
            run_name = wandb_run_name or f"bracs_heatmap_{self.timestamp}"
            wandb.init(project=wandb_project, name=run_name, config={
                'patch_size': patch_size,
                'extraction_level': extraction_level,
                'device': str(self.device),
                'timestamp': self.timestamp,
                'slide_path': slide_path
            })
        
        self.logger.info("="*80)
        self.logger.info("BRACS Heatmap Generator with Complete Functionality")
        self.logger.info("="*80)
        self.logger.info(f"Output directory: {self.run_dir}")
        self.logger.info(f"WandB enabled: {self.use_wandb}")
        self.logger.info(f"Patch size: {patch_size}, Extraction level: {extraction_level}")
    
    def setup_logging(self):
        """Setup logging configuration."""
        self.logger = logging.getLogger('BRACSHeatmap')
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []
        
        fh = logging.FileHandler(os.path.join(self.dirs['logs'], f'generation_{self.timestamp}.log'))
        fh.setLevel(logging.DEBUG)
        
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def load_config(self):
        """Load model configuration."""
        conf = Struct()
        conf.D_feat = 512
        conf.D_inner = 256
        conf.n_class = 3
        conf.n_token = 5
        conf.dropout = 0.25
        return conf
    
    def parse_geojson_annotation(self, geojson_path):
        """Parse GeoJSON annotation file."""
        try:
            with open(geojson_path, 'r') as f:
                geo_data = json.load(f)
            
            annotations = []
            annotations_by_class = {}
            
            for feature in geo_data.get('features', []):
                if feature['geometry']['type'] == 'Polygon':
                    coords = np.array(feature['geometry']['coordinates'][0])
                    class_name = feature.get('properties', {}).get('classification', {}).get('name', 'unknown')
                    
                    annotations.append({
                        'class': class_name,
                        'coordinates': coords
                    })
                    
                    # Group by class
                    class_key = class_name.lower()
                    if 'malignant' in class_key:
                        class_group = 'malignant'
                    elif 'dcis' in class_key or 'udh' in class_key:
                        class_group = 'atypical'
                    elif 'benign' in class_key or 'normal' in class_key:
                        class_group = 'benign'
                    else:
                        class_group = 'other'
                    
                    if class_group not in annotations_by_class:
                        annotations_by_class[class_group] = []
                    annotations_by_class[class_group].append(coords)
                    
                    self.logger.info(f"      Found annotation: {class_name} ({class_group}) with {len(coords)} points")
            
            self.logger.info(f"    Successfully parsed {len(annotations)} annotation regions")
            self.logger.info(f"    Classes found: {list(annotations_by_class.keys())}")
            
            return annotations, annotations_by_class
            
        except Exception as e:
            self.logger.warning(f"    Could not parse GeoJSON file: {e}")
            return [], {}
    
    def load_wsi_region(self, slide_path, level=3):
        """Load WSI region at specified level."""
        try:
            wsi = openslide.OpenSlide(slide_path)
            if level >= len(wsi.level_dimensions):
                level = len(wsi.level_dimensions) - 1
            
            width, height = wsi.level_dimensions[level]
            downsample = wsi.level_downsamples[level]
            region = wsi.read_region((0, 0), level, (width, height)).convert('RGB')
            
            self.logger.info(f"  Loaded WSI at level {level}: {width}x{height} (Downsample: {downsample:.2f})")
            
            # For BRACS at level 1
            if level == 1:
                self.logger.info(f"  Magnification: ~10x (from 40x base)")
            
            return np.array(region), (width, height), downsample, wsi
            
        except Exception as e:
            self.logger.error(f"  Error loading WSI: {e}")
            return None, None, None, None
    
    def load_features(self, slide_id):
        """Load features from HDF5 file."""
        with h5py.File(self.h5_path, 'r') as h5_file:
            if slide_id not in h5_file:
                slide_id = slide_id.split('.')[0]
            
            features = torch.from_numpy(h5_file[slide_id]['feat'][:]).to(self.device, torch.float32)
            coords = h5_file[slide_id]['coords'][:]
            
            self.logger.info(f"  Loaded features: {features.shape}, coords: {coords.shape}")
        
        return features.unsqueeze(0), coords
    
    def convert_to_grayscale(self, image):
        """Convert RGB image to grayscale while preserving dimensions."""
        gray = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
        gray_rgb = np.stack([gray, gray, gray], axis=-1).astype(np.uint8)
        return gray_rgb
    
    def create_ground_truth_overlay(self, wsi_image, annotations, downsample_factor):
        """Create ground truth visualization with colored regions."""
        self.logger.info("  Creating ground truth visualization...")
        
        try:
            # Convert WSI to grayscale for background
            gray_wsi = cv2.cvtColor(wsi_image, cv2.COLOR_RGB2GRAY)
            base_image = cv2.cvtColor(gray_wsi, cv2.COLOR_GRAY2RGB)
            
            color_overlay = np.zeros_like(base_image, dtype=np.uint8)
            
            def get_color_for_class(class_name):
                class_lower = class_name.lower()
                if 'benign' in class_lower or 'normal' in class_lower:
                    return [0, 255, 0]  # Green
                elif 'dcis' in class_lower or 'udh' in class_lower or 'atypical' in class_lower:
                    return [255, 255, 0]  # Yellow
                elif 'malignant' in class_lower or 'cancer' in class_lower or 'carcinoma' in class_lower:
                    return [255, 0, 0]  # Red
                else:
                    return [128, 128, 128]  # Gray
            
            annotation_mask = np.zeros(base_image.shape[:2], dtype=bool)
            
            for ann in annotations:
                class_name = ann.get('class', 'unknown')
                color = get_color_for_class(class_name)
                coords = ann['coordinates']
                
                scaled_coords = np.array([(int(x / downsample_factor), int(y / downsample_factor)) 
                                         for x, y in coords], dtype=np.int32)
                
                if len(scaled_coords) >= 3:
                    poly_mask = np.zeros(base_image.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(poly_mask, [scaled_coords], 255)
                    color_overlay[poly_mask > 0] = color
                    annotation_mask[poly_mask > 0] = True
                    cv2.polylines(color_overlay, [scaled_coords], True, (255, 255, 255), thickness=3)
                    
                    color_name = ('Yellow (DCIS/UDH)' if color == [255, 255, 0] else 
                                 'Red (Malignant)' if color == [255, 0, 0] else 
                                 'Green (Benign)' if color == [0, 255, 0] else 'Gray')
                    self.logger.info(f"      Drew {class_name} region with {color_name}")
            
            result = base_image.copy()
            alpha = 0.7
            result[annotation_mask] = (
                alpha * color_overlay[annotation_mask] + 
                (1 - alpha) * base_image[annotation_mask]
            ).astype(np.uint8)
            
            output_path = os.path.join(self.dirs['annotations'], 'ground_truth.png')
            Image.fromarray(result).save(output_path)
            self.logger.info(f"  ✓ Saved ground truth image to: {output_path}")
            
            return result, annotation_mask
            
        except Exception as e:
            self.logger.error(f"  ✗ Failed to create ground truth image: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None, None
    
    def generate_heatmap_adaptive(self, attention_scores, coords, slide_dims, downsample, model_name=""):
        """Generate heatmap with adaptive processing."""
        width, height = slide_dims
        heatmap = np.zeros((height, width), dtype=np.float32)
        patch_size_at_level = max(1, int(round(self.patch_size / downsample)))
        
        # Ensure attention is 1D numpy
        if isinstance(attention_scores, torch.Tensor):
            attention_scores = attention_scores.cpu().numpy()
        attention_scores = np.squeeze(attention_scores)
        
        # Validate dimensions
        if len(coords) != len(attention_scores):
            self.logger.error(f"  Mismatch: {len(coords)} coords vs {len(attention_scores)} attention scores")
            return heatmap
        
        # Analyze attention distribution
        non_zero_mask = attention_scores > np.percentile(attention_scores, 50)
        non_zero_count = non_zero_mask.sum()
        total_patches = len(attention_scores)
        sparsity_ratio = non_zero_count / total_patches if total_patches > 0 else 0
        
        self.logger.info(f"    {model_name} - Patches above median: {non_zero_count}/{total_patches} "
                        f"({sparsity_ratio*100:.1f}%)")
        
        # Check for uniform attention
        is_uniform = np.allclose(attention_scores, attention_scores.mean(), rtol=1e-5)
        
        # Process attention based on distribution
        if model_name == "MEANMIL" or is_uniform:
            attention_processed = attention_scores
            self.logger.info(f"    Detected uniform attention")
        elif sparsity_ratio < 0.05:  # Very sparse
            threshold = np.percentile(attention_scores, 95)
            mask = attention_scores > threshold
            attention_processed = np.zeros_like(attention_scores)
            if mask.any():
                attention_processed[mask] = attention_scores[mask]
                min_val = attention_processed[mask].min()
                max_val = attention_processed[mask].max()
                if max_val > min_val:
                    attention_processed[mask] = 0.2 + 0.8 * (attention_processed[mask] - min_val) / (max_val - min_val)
            self.logger.info(f"    Applied sparse attention processing")
        elif sparsity_ratio < 0.2:  # Sparse
            threshold = np.percentile(attention_scores, 80)
            mask = attention_scores > threshold
            attention_processed = np.zeros_like(attention_scores)
            if mask.any():
                attention_processed[mask] = attention_scores[mask]
                min_val = attention_processed[mask].min()
                max_val = attention_processed[mask].max()
                if max_val > min_val:
                    attention_processed[mask] = 0.2 + 0.8 * (attention_processed[mask] - min_val) / (max_val - min_val)
            self.logger.info(f"    Applied moderate sparse processing")
        else:
            # Standard normalization
            if attention_scores.max() > attention_scores.min():
                attention_processed = (attention_scores - attention_scores.min()) / (attention_scores.max() - attention_scores.min())
            else:
                attention_processed = attention_scores
            self.logger.info(f"    Applied standard normalization")
        
        # Fill patches
        for (x, y), prob in zip(coords, attention_processed):
            if prob > 1e-5:
                x_level = int(round(x / downsample))
                y_level = int(round(y / downsample))
                x_end = min(x_level + patch_size_at_level, width)
                y_end = min(y_level + patch_size_at_level, height)
                
                if x_level < width and y_level < height and x_end > x_level and y_end > y_level:
                    heatmap[y_level:y_end, x_level:x_end] = np.maximum(
                        heatmap[y_level:y_end, x_level:x_end], prob
                    )
        
        # Adaptive smoothing
        sigma_pixels = max(1.0, patch_size_at_level / 8.0)
        heatmap = gaussian_filter(heatmap, sigma=sigma_pixels)
        
        self.logger.info(f"    Final heatmap - Non-zero pixels: {(heatmap > 0.01).sum()}, "
                        f"Max value: {heatmap.max():.3f}, Mean: {heatmap.mean():.4f}")
        
        return heatmap
    
    def save_individual_heatmap(self, heatmap, slide_id, model_name):
        """Save individual heatmap as both npy and image."""
        # Save raw heatmap data
        np_path = os.path.join(self.dirs['heatmaps'], f'{slide_id}_{model_name}_heatmap.npy')
        np.save(np_path, heatmap)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Normalize for display
        if heatmap.max() > heatmap.min():
            hm_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        else:
            hm_norm = heatmap
        
        im = ax.imshow(hm_norm, cmap='jet', interpolation='bilinear')
        ax.set_title(f'{model_name} Heatmap - {slide_id}', fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Attention Score')
        
        img_path = os.path.join(self.dirs['heatmaps'], f'{slide_id}_{model_name}_heatmap.png')
        plt.savefig(img_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"    Saved individual heatmap: {img_path}")
        
        return img_path
    
    def create_consolidated_heatmaps_only(self, slide_id, wsi_image, heatmaps, annotations_by_class):
        """Create consolidated figure with only heatmaps (no overlays on WSI)."""
        try:
            self.logger.info(f"  Creating consolidated heatmap figure for {slide_id}...")
            
            n_cols = 3
            n_rows = 3
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 22))
            fig.subplots_adjust(wspace=0.4, hspace=0.5)
            axes = axes.flatten()
            
            # 1. Original WSI
            ax = axes[0]
            ax.imshow(wsi_image)
            ax.set_title('Original WSI', fontsize=14, fontweight='bold')
            ax.axis('off')
            
            # 2. Ground truth annotation mask - FIXED
            ax = axes[1]
            if annotations_by_class:
                # Create composite annotation mask at the same resolution as wsi_image
                mask_composite = np.zeros((*wsi_image.shape[:2], 3), dtype=np.uint8)
                
                # Color map for different classes
                color_map = {
                    'benign': [0, 255, 0],     # Green
                    'atypical': [255, 255, 0],  # Yellow
                    'malignant': [255, 0, 0],   # Red
                    'other': [128, 128, 128]    # Gray
                }
                
                # Get the actual downsample factor from the WSI
                # When loading at level 1, need to check the actual downsample
                wsi = openslide.OpenSlide(self.slide_path)
                actual_downsample = wsi.level_downsamples[3]  # Get level 3 downsample
                wsi.close()
                
                self.logger.info(f"    Creating ground truth with downsample factor: {actual_downsample}")
                
                for class_name, regions in annotations_by_class.items():
                    color = color_map.get(class_name, [128, 128, 128])
                    for coords in regions:
                        if len(coords) >= 3:
                            # Scale coordinates based on actual downsample factor
                            scaled_coords = coords / actual_downsample
                            scaled_coords = scaled_coords.astype(np.int32)
                            
                            # Ensure coordinates are within bounds
                            scaled_coords[:, 0] = np.clip(scaled_coords[:, 0], 0, mask_composite.shape[1]-1)
                            scaled_coords[:, 1] = np.clip(scaled_coords[:, 1], 0, mask_composite.shape[0]-1)
                            
                            cv2.fillPoly(mask_composite, [scaled_coords], color)
                
                # Check if any annotations were drawn
                if mask_composite.max() > 0:
                    ax.imshow(mask_composite)
                    ax.set_title('Ground Truth Annotations', fontsize=14, fontweight='bold', color='green')
                    self.logger.info(f"    Ground truth mask created with {(mask_composite.max(axis=2) > 0).sum()} annotated pixels")
                else:
                    # If still black, show the issue
                    ax.text(0.5, 0.5, 'Annotation scaling issue\nCheck coordinates', 
                           ha='center', va='center', fontsize=14, color='red')
                    ax.set_title('Ground Truth Failed', fontsize=14, fontweight='bold', color='red')
                    self.logger.warning("    No pixels drawn in ground truth mask - check coordinate scaling")
            else:
                ax.text(0.5, 0.5, 'No Annotations', ha='center', va='center', fontsize=14)
                ax.set_title('No Ground Truth Available', fontsize=14, fontweight='bold')
            ax.axis('off')
            
            # 3-9. Model heatmaps
            model_names = sorted(heatmaps.keys())
            self.logger.debug(f"    Adding {len(model_names)} model heatmaps")
            
            for i, model_name in enumerate(model_names):
                if i + 2 >= 9:
                    break
                
                ax = axes[i + 2]
                heatmap = heatmaps[model_name]
                
                # Normalize heatmap for display
                if heatmap.max() > heatmap.min():
                    hm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
                else:
                    hm = np.zeros_like(heatmap)
                
                # Display heatmap
                im = ax.imshow(hm, cmap='jet', interpolation='bilinear', aspect='equal', vmin=0, vmax=1)
                
                # Calculate statistics
                coverage = (heatmap > 0.01).sum() / heatmap.size * 100
                non_zero = heatmap[heatmap > 0]
                mean_val = non_zero.mean() if non_zero.size > 0 else 0
                
                # Title with statistics
                title = f'{model_name}\n'
                title += f'Max: {heatmap.max():.3f}, Mean: {mean_val:.3f}\n'
                title += f'Coverage: {coverage:.1f}%'
                
                ax.set_title(title, fontsize=11, fontweight='bold')
                ax.axis('off')
                
                # Add colorbar below each heatmap
                pos = ax.get_position()
                cbar_ax = fig.add_axes([pos.x0, pos.y0 - 0.03, pos.width, 0.01])
                cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
                cbar.ax.tick_params(labelsize=8)
            
            # Hide unused axes
            for i in range(len(model_names) + 2, 9):
                axes[i].axis('off')
            
            # Add title
            fig.suptitle(f'Heatmap Comparison - {slide_id}\n(Pure Attention Heatmaps)', 
                        fontsize=16, fontweight='bold', y=0.95)
            
            # Save figure
            save_path = os.path.join(self.dirs['heatmaps'], f'{slide_id}_consolidated_heatmaps.png')
            self.logger.info(f"  Saving consolidated heatmap to: {save_path}")
            
            plt.savefig(save_path, dpi=150, facecolor='white', edgecolor='none', format='png')
            plt.close(fig)
            
            # Verify file was created
            if os.path.exists(save_path):
                file_size = os.path.getsize(save_path)
                self.logger.info(f"  Successfully saved consolidated heatmap ({file_size} bytes)")
            else:
                self.logger.error(f"  ERROR: Consolidated heatmap file was not created!")
            
            if self.use_wandb and os.path.exists(save_path):
                wandb.log({f"{slide_id}_consolidated_heatmaps": wandb.Image(save_path)})
            
            return save_path
            
        except Exception as e:
            self.logger.error(f"  ERROR creating consolidated heatmap: {e}")
            import traceback
            traceback.print_exc()
            plt.close('all')
            return None
    
    def create_heatmap_overlay_bright(self, wsi_image, heatmap):
        """Create bright, visible heatmap overlay with smoother appearance."""
        # Convert WSI to grayscale
        wsi_gray = self.convert_to_grayscale(wsi_image)
        
        # Ensure heatmap matches image dimensions
        if heatmap.shape != wsi_image.shape[:2]:
            heatmap = zoom(heatmap,
                          (wsi_image.shape[0] / heatmap.shape[0],
                           wsi_image.shape[1] / heatmap.shape[1]),
                          order=1)
        
        # Create overlay starting with grayscale
        overlay = wsi_gray.copy()
        
        # Check if heatmap has any values
        if heatmap.max() <= 0:
            self.logger.warning("    Heatmap has no positive values!")
            return overlay
        
        # Strong normalization for visibility
        heatmap_norm = heatmap.copy()
        
        # Apply percentile-based normalization
        non_zero = heatmap_norm[heatmap_norm > 0]
        if non_zero.size > 0:
            vmin = np.percentile(non_zero, 5)
            vmax = np.percentile(non_zero, 95)
            heatmap_norm = np.clip((heatmap_norm - vmin) / (vmax - vmin), 0, 1)
            
            # Apply gamma correction for brightness
            gamma = 0.25
            heatmap_norm = np.power(heatmap_norm, gamma)
        
        # Use JET colormap for vibrant colors
        cmap = cm.get_cmap('jet')
        heatmap_colored = cmap(heatmap_norm)[:, :, :3]
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # Create smooth blend
        mask = heatmap_norm > 0.001
        
        if mask.any():
            for c in range(3):
                alpha = np.clip(heatmap_norm[mask] * 1.5, 0.3, 1.0)
                overlay[mask, c] = (
                    (1 - alpha) * wsi_gray[mask, c] * 0.3 +
                    alpha * heatmap_colored[mask, c]
                ).astype(np.uint8)
            
            self.logger.info(f"    Bright overlay created with {mask.sum()} colored pixels "
                           f"({mask.sum()*100/mask.size:.2f}% of image)")
        else:
            self.logger.warning("    No pixels above threshold")
        
        return overlay
    
    def create_consolidated_heatmap_figure(self, slide_id, wsi_image, heatmaps, annotations, 
                                          downsample, methods_used):
        """Create consolidated figure with all models (with overlays)."""
        n_models = len(heatmaps) + 1  # +1 for ground truth
        n_cols = 4
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(24, 6*n_rows))
        idx = 1
        
        # Ground Truth
        ax = plt.subplot(n_rows, n_cols, idx)
        if annotations:
            gt_overlay, _ = self.create_ground_truth_overlay(wsi_image, annotations, downsample)
            if gt_overlay is not None:
                ax.imshow(gt_overlay)
                ax.set_title('Ground Truth Annotations', fontsize=14, fontweight='bold', color='darkgreen')
            else:
                ax.imshow(wsi_image)
                ax.set_title('Original WSI', fontsize=14, fontweight='bold')
        else:
            ax.imshow(wsi_image)
            ax.set_title('Original WSI', fontsize=14, fontweight='bold')
        ax.axis('off')
        idx += 1
        
        # Model heatmaps with overlays
        for model_name in sorted(heatmaps.keys()):
            heatmap = heatmaps[model_name]
            ax = plt.subplot(n_rows, n_cols, idx)
            
            # Create bright overlay
            overlay = self.create_heatmap_overlay_bright(wsi_image, heatmap)
            ax.imshow(overlay)
            
            # Add statistics
            method_label = methods_used.get(model_name, "Attention")
            non_zero = heatmap[heatmap > 0]
            if non_zero.size > 0:
                max_val = heatmap.max()
                mean_val = non_zero.mean()
                coverage = (heatmap > 0.01).sum() / heatmap.size * 100
            else:
                max_val = 0
                mean_val = 0
                coverage = 0
            
            title = f'{model_name} ({method_label})\n'
            title += f'Max: {max_val:.3f}, Mean: {mean_val:.3f}\n'
            title += f'Coverage: {coverage:.1f}%'
            
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.axis('off')
            idx += 1
        
        # Fill remaining subplots
        while idx <= n_rows * n_cols:
            ax = plt.subplot(n_rows, n_cols, idx)
            ax.axis('off')
            idx += 1
        
        # Legend
        legend_elements = [
            mpatches.Patch(color='red', label='High Attention'),
            mpatches.Patch(color='yellow', label='Medium Attention'),
            mpatches.Patch(color='blue', label='Low Attention'),
            mpatches.Patch(color='gray', label='No Attention')
        ]
        
        if annotations:
            legend_elements.extend([
                mpatches.Patch(color='green', label='Benign (GT)', alpha=0.5),
                mpatches.Patch(color='yellow', label='Atypical/DCIS/UDH (GT)', alpha=0.5),
                mpatches.Patch(color='red', label='Malignant (GT)', alpha=0.5)
            ])
        
        plt.figlegend(handles=legend_elements, 
                     loc='lower center', ncol=4, fontsize=10,
                     bbox_to_anchor=(0.5, -0.02))
        
        plt.suptitle(f'Consolidated Model Attention Comparison - {slide_id}', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        save_path = os.path.join(self.dirs['comparisons'], f'{slide_id}_consolidated_all_models.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"  Saved consolidated comparison to: {save_path}")
        
        if self.use_wandb:
            wandb.log({f"{slide_id}_comparison": wandb.Image(save_path)})
        
        return save_path
    
    def calculate_quantitative_metrics(self, heatmaps, annotations_by_class, wsi_shape, slide_id):
        """Calculate quantitative metrics for all models."""
        if not annotations_by_class:
            self.logger.info("  No annotations available for quantitative analysis")
            return None
        
        analyzer = QuantitativeHeatmapAnalyzer(self.dirs['metrics'])
        
        # Create ground truth masks
        gt_masks = analyzer.create_ground_truth_masks(annotations_by_class, wsi_shape)
        
        metrics_all = {}
        
        for model_name, heatmap in heatmaps.items():
            metrics = {}
            
            # Resize heatmap if needed
            if heatmap.shape != wsi_shape[:2]:
                heatmap_resized = zoom(heatmap,
                                      (wsi_shape[0] / heatmap.shape[0],
                                       wsi_shape[1] / heatmap.shape[1]),
                                      order=1)
            else:
                heatmap_resized = heatmap
            
            # Calculate metrics for each class
            for class_name, gt_mask in gt_masks.items():
                if gt_mask.sum() == 0:
                    continue
                
                # Threshold heatmap
                threshold = np.percentile(heatmap_resized[heatmap_resized > 0], 75) if (heatmap_resized > 0).any() else 0
                heatmap_binary = heatmap_resized > threshold
                
                # Calculate metrics
                dice = analyzer.calculate_dice_coefficient(heatmap_binary, gt_mask)
                iou = analyzer.calculate_iou(heatmap_binary, gt_mask)
                
                # ROC metrics
                roc_metrics = analyzer.calculate_roc_metrics(heatmap_resized, gt_mask)
                
                metrics[f'dice_{class_name}'] = dice
                metrics[f'iou_{class_name}'] = iou
                metrics[f'auc_{class_name}'] = roc_metrics['roc_auc']
                metrics[f'pr_auc_{class_name}'] = roc_metrics['pr_auc']
            
            # Calculate entropy
            metrics['entropy'] = analyzer.calculate_attention_entropy(heatmap)
            metrics['attention_coverage'] = (heatmap > 0.01).sum() / heatmap.size
            
            metrics_all[model_name] = metrics
            
            # Log to WandB
            if self.use_wandb:
                for metric_name, value in metrics.items():
                    wandb.log({f"{slide_id}_{model_name}_{metric_name}": value})
        
        # Save metrics
        metrics_df = pd.DataFrame(metrics_all).T
        metrics_path = os.path.join(self.dirs['reports'], f'{slide_id}_quantitative_metrics.csv')
        metrics_df.to_csv(metrics_path)
        self.logger.info(f"  Saved quantitative metrics to: {metrics_path}")
        
        return metrics_df
    
    # Model loading methods
    def load_model(self, checkpoint_path, model_class, **kwargs):
        """Generic model loading function."""
        model = model_class(self.load_config(), **kwargs)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get('model', checkpoint.get('model_state_dict', checkpoint))
        model.load_state_dict(state_dict, strict=False)
        return model.to(self.device).eval()
    
    def load_dtfd(self, checkpoint_paths):
        """Load DTFD model components."""
        conf = self.load_config()
        modules = {
            'classifier': Classifier_1fc(conf.D_inner, conf.n_class, 0).to(self.device),
            'attention': Attention(conf.D_inner).to(self.device),
            'dimReduction': DimReduction(conf.D_feat, conf.D_inner).to(self.device),
            'attCls': Attention_with_Classifier(L=conf.D_inner, num_cls=conf.n_class, droprate=0).to(self.device)
        }
        
        for name, module in modules.items():
            if name in checkpoint_paths and os.path.exists(checkpoint_paths[name]):
                ckpt = torch.load(checkpoint_paths[name], map_location=self.device, weights_only=False)
                module.load_state_dict(ckpt.get('model', ckpt), strict=False)
        
        return {name: module.eval() for name, module in modules.items()}
    
    def load_oodml(self, checkpoint_path):
        """Load OODML model."""
        model = OODML(input_dim=512, n_classes=3, K=5, embed_dim=512,
                      pseudo_bag_size=512, tau=2.0, dropout=0.1, heads=8, use_ddm=True)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
        model.load_state_dict(state_dict, strict=False)
        return model.to(self.device).eval()
    
    @torch.no_grad()
    def get_acmil_attention(self, model, features):
        """Extract ACMIL attention with logging."""
        x = features[0] if len(features.shape) == 3 else features
        
        # Get attention scores (assuming model returns _, _, attn_scores)
        _, _, attn_scores = model(features)  # Adjust if needed based on forward
        
        # attn_scores likely (1, N) or similar; log raw shape
        self.logger.info(f"ACMIL raw attn_scores shape: {attn_scores.shape}")
        
        # Handle multi-head or token averaging
        if len(attn_scores.shape) == 3:
            attn_raw = attn_scores.mean(dim=1).squeeze()  # Average over heads if 3D
        else:
            attn_raw = attn_scores.squeeze()
        
        # Dynamically choose softmax dim: the larger one (N patches)
        if len(attn_raw.shape) >= 2:
            dim = 0 if attn_raw.shape[0] > attn_raw.shape[1] else 1
            self.logger.info(f"ACMIL selected softmax dim: {dim}")
        else:
            dim = 0  # Fallback
        
        # Apply softmax over patches
        attention = torch.softmax(attn_raw, dim=dim).cpu().numpy()
        
        # If still 2D after softmax, take appropriate dimension or mean
        if len(attention.shape) > 1:
            attention = attention[0, :] if attention.shape[0] == 1 else attention[:, 0]
        
        # Analyze distribution for logging
        n_patches = len(attention)
        top_k = min(100, max(10, int(n_patches * 0.01)))  # Top 1% or at least 10 patches
        top_k_indices = np.argpartition(attention, -top_k)[-top_k:]
        top_k_mean = attention[top_k_indices].mean()
        
        self.logger.info(f" ACMIL attention - Min: {attention.min():.6f}, "
                         f"Max: {attention.max():.6f}, Mean: {attention.mean():.6f}")
        self.logger.info(f" ACMIL - Top {top_k} patches mean: {top_k_mean:.6f} "
                         f"({top_k_mean/attention.mean():.1f}x average)")
        self.logger.info(f" ACMIL - Attention sum: {attention.sum():.4f}")
        
        return attention
    
    @torch.no_grad()
    def get_abmil_attention(self, model, features):
        """Extract ABMIL attention - corrected for proper gated attention."""
        x = features[0] if len(features.shape) == 3 else features
       
        # Dimension reduction
        med_feat = model.dimreduction(x)
       
        # Get attention scores - ABMIL uses gated attention
        A = model.attention(med_feat)
        self.logger.info(f"ABMIL raw A shape: {A.shape}")  # Added for shape debugging
       
        # Dynamically choose softmax dim: the one with the larger size (N patches)
        if len(A.shape) >= 2:
            dim = 0 if A.shape[0] > A.shape[1] else 1
            self.logger.info(f"ABMIL selected softmax dim: {dim}")  # Log the selected dim
        else:
            dim = 0  # Fallback for unexpected shapes
       
        # Apply softmax normalization over patches
        A = torch.softmax(A, dim=dim)
       
        attention = A.squeeze().cpu().numpy()
       
        # If still 2D after squeeze, take the appropriate dimension
        if len(attention.shape) > 1:
            attention = attention[0, :] if attention.shape[0] == 1 else attention[:, 0]
       
        # Analyze distribution for logging
        n_patches = len(attention)
        top_k = min(100, max(10, int(n_patches * 0.01))) # Top 1% or at least 10 patches
        top_k_indices = np.argpartition(attention, -top_k)[-top_k:]
        top_k_mean = attention[top_k_indices].mean()
       
        self.logger.info(f" ABMIL attention - Min: {attention.min():.6f}, "
                        f"Max: {attention.max():.6f}, Mean: {attention.mean():.6f}")
        self.logger.info(f" ABMIL - Top {top_k} patches mean: {top_k_mean:.6f} "
                        f"({top_k_mean/attention.mean():.1f}x average)")
        self.logger.info(f" ABMIL - Attention sum: {attention.sum():.4f}")
       
        return attention
    
    def get_meanmil_attention(self, features):
        """Extract MeanMIL attention (uniform)."""
        n_instances = features.shape[1] if len(features.shape) == 3 else features.shape[0]
        return np.ones(n_instances) / n_instances
    
    @torch.no_grad()
    def get_maxmil_attention(self, model, features):
        """Extract MaxMIL attention."""
        x = features[0] if len(features.shape) == 3 else features
        patch_scores = model.head(x).max(dim=1)[0]
        
        # Select top patches
        k = min(100, max(5, int(len(patch_scores) * 0.05)))
        top_k_values, top_k_indices = torch.topk(patch_scores, k)
        
        # Create sparse attention
        attention = torch.zeros_like(patch_scores)
        attention[top_k_indices] = torch.nn.functional.softmax(top_k_values, dim=0)
        
        result = attention.cpu().numpy()
        self.logger.info(f"    MaxMIL attention - Selected {k} patches, Max: {result.max():.6f}")
        return result
    
    @torch.no_grad()
    def get_dtfd_attention(self, models, features):
        """Extract DTFD attention."""
        x = features[0] if len(features.shape) == 3 else features
        med_feat = models['dimReduction'](x)
        AA = models['attention'](med_feat, isNorm=False).squeeze(0)
        attention = torch.nn.functional.softmax(AA, dim=0).cpu().numpy()
        self.logger.info(f"    DTFD attention - Min: {attention.min():.6f}, "
                        f"Max: {attention.max():.6f}, Mean: {attention.mean():.6f}")
        return attention
    
    def get_transmil_attention_gradcam(self, model, features):
        """Extract TransMIL attention using GradCAM."""
        self.logger.info("    Using GradCAM for TransMIL")
        model.eval()
        features = features.clone().detach().requires_grad_(True)
        logits = model(features)
        
        # Get target class score
        if logits.shape[-1] > 1:
            target_score = logits[0, 1]  # Assuming abnormal class
        else:
            target_score = logits[0] if len(logits.shape) == 1 else logits[0, 0]
        
        model.zero_grad()
        target_score.backward()
        
        gradients = features.grad[0] if len(features.shape) == 3 else features.grad
        gradients_detached = gradients.detach()
        gradients_detached = torch.nn.functional.relu(gradients_detached)
        
        weights = gradients_detached.mean(dim=1)
        if weights.max() > weights.min():
            weights = (weights - weights.min()) / (weights.max() - weights.min())
        
        result = weights.cpu().numpy()
        self.logger.info(f"    TransMIL GradCAM - Min: {result.min():.6f}, "
                        f"Max: {result.max():.6f}, Mean: {result.mean():.6f}")
        return result
    
    def get_oodml_attention_gradcam(self, model, features):
        """Extract OODML attention using GradCAM."""
        self.logger.info("    Using GradCAM for OODML")
        model.eval()
        x = features[0] if len(features.shape) == 3 else features
        x = x.clone().detach().requires_grad_(True)
        
        output_dict = model(x)
        logits = output_dict.get('Y_hat_DM', output_dict.get('Y_hat', None))
        
        if logits is None:
            self.logger.error("    OODML output not found")
            return np.ones(x.shape[0]) / x.shape[0]
        
        # Get target score
        if len(logits.shape) > 1 and logits.shape[1] > 1:
            target_score = logits[0, 1]
        elif len(logits.shape) == 1 and logits.shape[0] > 1:
            target_score = logits[1]
        else:
            target_score = logits[0] if len(logits.shape) == 1 else logits[0, 0]
        
        model.zero_grad()
        target_score.backward()
        
        gradients = x.grad
        gradients = torch.nn.functional.relu(gradients)
        weights = gradients.mean(dim=1)
        
        if weights.max() > weights.min():
            weights = (weights - weights.min()) / (weights.max() - weights.min())
        
        result = weights.cpu().numpy()
        self.logger.info(f"    OODML GradCAM - Min: {result.min():.6f}, "
                        f"Max: {result.max():.6f}, Mean: {result.mean():.6f}")
        return result
    
    def process_slide(self, slide_id, model_paths, dtfd_paths=None):
        """Process a single slide with all models."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Processing slide: {slide_id}")
        self.logger.info('='*60)
        
        # Load WSI
        wsi_image, wsi_dims, downsample, wsi = self.load_wsi_region(self.slide_path, level=3)
        if wsi_image is None:
            return None
        
        # Parse annotations
        annotations = None
        annotations_by_class = {}
        if self.annotation_path and os.path.exists(self.annotation_path):
            annotations, annotations_by_class = self.parse_geojson_annotation(self.annotation_path)
        
        # Load features
        features, coords = self.load_features(slide_id)
        if features is None:
            return None
        
        # Process each model
        heatmaps = {}
        methods_used = {}
        
        # Process regular models
        for model_name, checkpoint_path in model_paths.items():
            if model_name == 'dtfd':
                continue  # Handle separately
            
            if not os.path.exists(checkpoint_path):
                self.logger.warning(f"  Skipping {model_name}: checkpoint not found")
                continue
            
            try:
                self.logger.info(f"\n  Processing {model_name.upper()}...")
                
                # Create fresh copies
                features_copy = features.clone()
                coords_copy = coords.copy()
                
                if model_name == 'acmil':
                    model = self.load_model(checkpoint_path, ACMIL_GA, D=256)
                    attention = self.get_acmil_attention(model, features_copy)
                    methods_used[model_name.upper()] = "Attention"
                
                elif model_name == 'abmil':
                    model = self.load_model(checkpoint_path, ABMIL, D=128)
                    attention = self.get_abmil_attention(model, features_copy)
                    methods_used[model_name.upper()] = "Attention"
                
                elif model_name == 'transmil':
                    model = self.load_model(checkpoint_path, TransMIL)
                    attention = self.get_transmil_attention_gradcam(model, features_copy)
                    methods_used[model_name.upper()] = "GradCAM"
                
                elif model_name == 'meanmil':
                    model = self.load_model(checkpoint_path, mean_max.MeanMIL)
                    attention = self.get_meanmil_attention(features_copy)
                    methods_used[model_name.upper()] = "Uniform"
                
                elif model_name == 'maxmil':
                    model = self.load_model(checkpoint_path, mean_max.MaxMIL)
                    attention = self.get_maxmil_attention(model, features_copy)
                    methods_used[model_name.upper()] = "Attention"
                
                elif model_name == 'oodml':
                    model = self.load_oodml(checkpoint_path)
                    attention = self.get_oodml_attention_gradcam(model, features_copy)
                    methods_used[model_name.upper()] = "GradCAM"
                else:
                    continue
                
                # Generate heatmap
                heatmap = self.generate_heatmap_adaptive(
                    attention.copy(), 
                    coords_copy, 
                    wsi_dims, 
                    downsample, 
                    model_name.upper()
                )
                heatmaps[model_name.upper()] = heatmap
                
                # Save individual heatmap
                self.save_individual_heatmap(heatmap, slide_id, model_name.upper())
                
                # Log to WandB
                if self.use_wandb:
                    wandb.log({
                        f"{slide_id}_{model_name}_max_attention": heatmap.max(),
                        f"{slide_id}_{model_name}_coverage": (heatmap > 0.01).sum() / heatmap.size
                    })
                
                # Clean up
                del model
                torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                self.logger.error(f"    Error with {model_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Handle DTFD
        if dtfd_paths and all(os.path.exists(p) for p in dtfd_paths.values()):
            try:
                self.logger.info(f"\n  Processing DTFD...")
                
                features_copy = features.clone()
                coords_copy = coords.copy()
                
                dtfd_models = self.load_dtfd(dtfd_paths)
                attention = self.get_dtfd_attention(dtfd_models, features_copy)
                methods_used['DTFD'] = "Attention"
                
                heatmap = self.generate_heatmap_adaptive(
                    attention.copy(), 
                    coords_copy, 
                    wsi_dims, 
                    downsample, 
                    "DTFD"
                )
                heatmaps['DTFD'] = heatmap
                
                # Save individual heatmap
                self.save_individual_heatmap(heatmap, slide_id, 'DTFD')
                
                # Log to WandB
                if self.use_wandb:
                    wandb.log({
                        f"{slide_id}_dtfd_max_attention": heatmap.max(),
                        f"{slide_id}_dtfd_coverage": (heatmap > 0.01).sum() / heatmap.size
                    })
                
                # Clean up
                for model in dtfd_models.values():
                    del model
                torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                self.logger.error(f"    Error with DTFD: {e}")
        
        # Clean up WSI handle
        if wsi is not None:
            wsi.close()
        
        # Create visualizations
        if heatmaps:
            # Create consolidated comparison with overlays
            self.create_consolidated_heatmap_figure(
                slide_id, wsi_image, heatmaps, 
                annotations, downsample, methods_used
            )
            
            # Create consolidated heatmaps-only figure
            self.create_consolidated_heatmaps_only(
                slide_id, wsi_image, heatmaps, annotations_by_class
            )
            
            # Calculate quantitative metrics if annotations available
            if annotations_by_class:
                metrics_df = self.calculate_quantitative_metrics(
                    heatmaps, annotations_by_class, wsi_image.shape, slide_id
                )
        
        # Save slide metadata
        metadata = {
            'slide_id': slide_id,
            'timestamp': self.timestamp,
            'models_processed': list(heatmaps.keys()),
            'has_annotations': bool(annotations),
            'annotation_classes': list(annotations_by_class.keys()),
            'wsi_dimensions': wsi_dims,
            'downsample_factor': downsample,
            'extraction_level': 1
        }
        
        metadata_path = os.path.join(self.dirs['data'], f'{slide_id}_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        gc.collect()
        
        return heatmaps


def main():
    """Main execution function."""
    
    # Configuration
    h5_path = '/vol/research/scratch1/NOBACKUP/rk01337/ACMIL/processed_bracs_10_filtered/patch_feats_pretrain_natural_supervised_Resnet18.h5'
    slide_path = '/vol/research/scratch1/NOBACKUP/rk01337/BRACS/wsi_slide/BRACS_1284.svs'
    annotation_path = '/vol/research/scratch1/NOBACKUP/rk01337/BRACS/wsi_slide/BRACS_1284.geojson'
    checkpoint_dir = '/vol/research/scratch1/NOBACKUP/rk01337/ACMIL/checkpoint_analysis/BRACS/resnet18'
    output_dir = '/vol/research/scratch1/NOBACKUP/rk01337/ACMIL/bracs_heatmap_results_resnet18'
    
    # Model paths
    model_paths = {
        'acmil': os.path.join(checkpoint_dir, 'bracs_acmil_resnet18_checkpoint-best.pth'),
        'abmil': os.path.join(checkpoint_dir, 'bracs_abmil_resnet18_checkpoint-best.pth'),
        'transmil': os.path.join(checkpoint_dir, 'bracs_transmil_resnet18_checkpoint-best.pth'),
        'meanmil': os.path.join(checkpoint_dir, 'bracs_meanmil_resnet18_checkpoint-best.pth'),
        'maxmil': os.path.join(checkpoint_dir, 'bracs_maxmil_resnet18_checkpoint-best.pth'),
        'oodml': os.path.join(checkpoint_dir, 'bracs_oodml_resnet18_checkpoint-best.pt'),
    }
    
    # DTFD paths
    dtfd_paths = {
        'classifier': os.path.join(checkpoint_dir, 'bracs_dtfd_resnet18_checkpoint-best_classifier.pth'),
        'attention': os.path.join(checkpoint_dir, 'bracs_dtfd_resnet18_checkpoint-best_attention.pth'),
        'dimReduction': os.path.join(checkpoint_dir, 'bracs_dtfd_resnet18_checkpoint-best_dimReduction.pth'),
        'attCls': os.path.join(checkpoint_dir, 'bracs_dtfd_resnet18_checkpoint-best_attCls.pth'),
    }
    
    # Initialize generator with WandB
    generator = BRACSHeatmapGenerator(
        h5_path=h5_path,
        slide_path=slide_path,
        checkpoint_dir=checkpoint_dir,
        annotation_path=annotation_path,
        output_dir=output_dir,
        use_wandb=True,
        wandb_project='bracs-heatmap',
        wandb_run_name=f'bracs_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    
    # Check checkpoints
    generator.logger.info("\nChecking model checkpoints:")
    for name, path in model_paths.items():
        if os.path.exists(path):
            generator.logger.info(f"  ✓ {name}: Found")
        else:
            generator.logger.error(f"  ✗ {name}: Not found")
    
    generator.logger.info("\nChecking DTFD components:")
    for name, path in dtfd_paths.items():
        if os.path.exists(path):
            generator.logger.info(f"  ✓ DTFD {name}: Found")
        else:
            generator.logger.error(f"  ✗ DTFD {name}: Not found")
    
    # Process slide
    slide_id = 'BRACS_1284'
    heatmaps = generator.process_slide(slide_id, model_paths, dtfd_paths)
    
    if heatmaps:
        generator.logger.info(f"\n{'='*60}")
        generator.logger.info("Analysis Complete!")
        generator.logger.info('='*60)
        generator.logger.info(f"Models processed: {list(heatmaps.keys())}")
        generator.logger.info(f"All results saved in: {generator.run_dir}")
        generator.logger.info("Outputs include:")
        generator.logger.info("  - Individual heatmaps for each model")
        generator.logger.info("  - Consolidated heatmaps-only figure")
        generator.logger.info("  - Consolidated comparison figure with overlays")
        generator.logger.info("  - Ground truth annotations")
        generator.logger.info("  - Quantitative metrics (if annotations available)")
    else:
        generator.logger.error("No heatmaps generated. Please check logs for errors.")
    
    if generator.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
