import torch
import torch.nn as nn
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.ndimage import gaussian_filter, zoom
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import openslide
import os
import sys
import yaml
from datetime import datetime
import json
import logging
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
from skimage.filters import threshold_otsu
import cv2
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

class CAMELYONHeatmapGenerator:
    def __init__(self, h5_path, slide_dir, checkpoint_dir, annotation_dir, 
                 output_dir='camelyon_heatmap_results', device='cuda',
                 patch_size=256, extraction_level=0,
                 use_wandb=True, wandb_project='camelyon-heatmap', wandb_run_name=None):
        """
        Multi-threshold version for better performance
        """
        self.h5_path = h5_path
        self.slide_dir = slide_dir
        self.checkpoint_dir = checkpoint_dir
        self.annotation_dir = annotation_dir
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.patch_size = patch_size
        self.extraction_level = extraction_level
        
        # Output directory setup
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(output_dir, f"run_{self.timestamp}")
        
        self.dirs = {
            'overlays': os.path.join(self.run_dir, "wsi_overlays"),
            'heatmaps': os.path.join(self.run_dir, "heatmaps_only"),
            'comparisons': os.path.join(self.run_dir, "comparisons"),
            'annotations': os.path.join(self.run_dir, "ground_truth"),
            'data': os.path.join(self.run_dir, "attention_data"),
            'logs': os.path.join(self.run_dir, "logs"),
            'metrics': os.path.join(self.run_dir, "metrics"),
            'roc_curves': os.path.join(self.run_dir, "roc_curves")
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        self.setup_logging()
        
        # Initialize WandB
        self.use_wandb = use_wandb
        if self.use_wandb:
            run_name = wandb_run_name or f"heatmap_multithresh_{self.timestamp}"
            wandb.init(project=wandb_project, name=run_name, config={
                'patch_size': patch_size,
                'extraction_level': extraction_level,
                'device': str(self.device),
                'timestamp': self.timestamp,
                'thresholding': 'multi-threshold'
            })
        
        self.logger.info("="*80)
        self.logger.info("CAMELYON16 Heatmap Generator - Multi-threshold Version")
        self.logger.info("="*80)
        self.logger.info(f"Output directory: {self.run_dir}")
        self.logger.info(f"WandB enabled: {self.use_wandb}")
    
    def setup_logging(self):
        """Setup logging"""
        self.logger = logging.getLogger('CAMELYONHeatmap')
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
        """Load configuration"""
        config_path = '/vol/research/scratch1/NOBACKUP/rk01337/ACMIL/config/camelyon_medical_ssl_config.yml'
        with open(config_path, 'r') as f:
            conf = Struct(**yaml.load(f, Loader=yaml.FullLoader))
        
        conf.D_feat = 384
        conf.D_inner = 128
        conf.n_class = 2
        conf.n_token = 2
        conf.dropout = 0.25
        
        return conf
    
    def parse_camelyon_xml(self, xml_path, wsi_dims, downsample):
        """Parse CAMELYON16 XML annotations"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            annotation_groups = []
            
            for annotation in root.findall('.//Annotation'):
                annotation_type = annotation.get('Type', 'None')
                coordinates = []
                
                for coord in annotation.findall('.//Coordinate'):
                    x = float(coord.get('X'))
                    y = float(coord.get('Y'))
                    coordinates.append([x, y])
                
                if coordinates:
                    annotation_groups.append({
                        'type': annotation_type,
                        'coordinates': np.array(coordinates)
                    })
            
            self.logger.info(f"    Found {len(annotation_groups)} annotation regions")
            return annotation_groups
            
        except Exception as e:
            self.logger.error(f"    Error parsing XML: {e}")
            return []
    
    def create_annotation_mask(self, annotation_groups, wsi_dims, downsample):
        """Create annotation mask at the specified level"""
        width, height = wsi_dims
        
        mask_img = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask_img)
        
        for group in annotation_groups:
            coords_scaled = (group['coordinates'] / downsample).astype(np.int32)
            coord_list = [(int(x), int(y)) for x, y in coords_scaled]
            
            if len(coord_list) >= 3:
                draw.polygon(coord_list, fill=255)
        
        mask = np.array(mask_img) > 0
        
        cancer_pixels = mask.sum()
        total_pixels = mask.size
        cancer_ratio = cancer_pixels / total_pixels
        
        self.logger.info(f"    Mask created: {width}x{height}")
        self.logger.info(f"    Cancer pixels: {cancer_pixels} ({cancer_ratio*100:.2f}%)")
        
        return mask
    
    def load_wsi_region(self, slide_path, level=5):
        """Load WSI at specified level"""
        try:
            wsi = openslide.OpenSlide(slide_path)
            
            if level >= len(wsi.level_dimensions):
                self.logger.warning(f"Level {level} not available, using level {len(wsi.level_dimensions)-1}")
                level = len(wsi.level_dimensions) - 1
            
            width, height = wsi.level_dimensions[level]
            downsample = float(wsi.level_downsamples[level])
            
            region = wsi.read_region((0, 0), level, (width, height))
            region = region.convert('RGB')
            
            self.logger.info(f"  Loaded WSI at level {level}: {width}x{height}")
            self.logger.info(f"  Downsample factor: {downsample:.4f}")
            
            return np.array(region), (width, height), downsample, wsi
            
        except Exception as e:
            self.logger.error(f"  Error loading WSI: {e}")
            return None, None, None, None
    
    def load_features(self, slide_id):
        """Load features and coordinates from HDF5"""
        with h5py.File(self.h5_path, 'r') as h5_file:
            if slide_id not in h5_file:
                self.logger.error(f"  Slide {slide_id} not found in HDF5")
                return None, None
            
            features = torch.from_numpy(h5_file[slide_id]['feat'][:]).to(self.device, torch.float32)
            coords = h5_file[slide_id]['coords'][:]
            
            self.logger.info(f"  Loaded features: {features.shape}, coords: {coords.shape}")
            
        return features.unsqueeze(0), coords
    
    def convert_to_grayscale(self, image):
        """Convert RGB image to grayscale RGB for overlay"""
        gray = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
        gray_rgb = np.stack([gray, gray, gray], axis=-1).astype(np.uint8)
        return gray_rgb
    
    def generate_heatmap_refined(self, attention_scores, coords, slide_dims, downsample, 
                                wsi_gray, model_name="", wsi_handle=None):
        """Generate heatmap with adaptive enhancement"""
        width, height = slide_dims
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        patch_size_at_level = max(1, int(round(self.patch_size / downsample)))
        
        # Ensure attention is 1D numpy
        if isinstance(attention_scores, torch.Tensor):
            attention_scores = attention_scores.cpu().numpy()
        if len(attention_scores.shape) > 1:
            attention_scores = attention_scores.squeeze()
        
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
        
        # Adaptive enhancement based on sparsity
        if model_name == "MEANMIL" or is_uniform:
            attention_enhanced = attention_scores
            self.logger.info(f"    Detected uniform attention, no enhancement applied")
        elif sparsity_ratio < 0.05:  # Very sparse
            threshold = np.percentile(attention_scores, 95)
            mask = attention_scores > threshold
            attention_enhanced = np.zeros_like(attention_scores)
            if mask.any():
                attention_enhanced[mask] = attention_scores[mask]
                min_val = attention_enhanced[mask].min()
                max_val = attention_enhanced[mask].max()
                if max_val > min_val:
                    attention_enhanced[mask] = 0.2 + 0.8 * (attention_enhanced[mask] - min_val) / (max_val - min_val)
            self.logger.info(f"    Applied strong enhancement for very sparse attention")
        elif sparsity_ratio < 0.2:  # Sparse
            threshold = np.percentile(attention_scores, 80)
            mask = attention_scores > threshold
            attention_enhanced = np.zeros_like(attention_scores)
            if mask.any():
                attention_enhanced[mask] = attention_scores[mask]
                min_val = attention_enhanced[mask].min()
                max_val = attention_enhanced[mask].max()
                if max_val > min_val:
                    attention_enhanced[mask] = 0.2 + 0.8 * (attention_enhanced[mask] - min_val) / (max_val - min_val)
            self.logger.info(f"    Applied moderate enhancement for sparse attention")
        else:  # Dense
            if attention_scores.max() > attention_scores.min():
                attention_enhanced = (attention_scores - attention_scores.min()) / (attention_scores.max() - attention_scores.min())
            else:
                attention_enhanced = attention_scores
            self.logger.info(f"    Applied standard normalization for dense attention")
        
        # Fill patches
        for (x, y), prob in zip(coords, attention_enhanced):
            if prob > 1e-5:
                x_level = int(round(x / downsample))
                y_level = int(round(y / downsample))
                
                x_end = min(x_level + patch_size_at_level, width)
                y_end = min(y_level + patch_size_at_level, height)
                
                if x_level < width and y_level < height and x_end > x_level and y_end > y_level:
                    heatmap[y_level:y_end, x_level:x_end] = np.maximum(
                        heatmap[y_level:y_end, x_level:x_end], prob
                    )
        
        # Apply smoothing
        sigma_pixels = max(1.0, patch_size_at_level / 8.0)
        heatmap = gaussian_filter(heatmap, sigma=sigma_pixels)
        
        self.logger.debug(f"    Applied smoothing with sigma={sigma_pixels:.1f}")
        self.logger.info(f"    Final heatmap - Non-zero pixels: {(heatmap > 0.01).sum()}, "
                        f"Max: {heatmap.max():.3f}, Mean: {heatmap.mean():.4f}")
        
        return heatmap
    
    def save_individual_heatmap(self, heatmap, slide_id, model_name):
        """Save individual heatmap as image"""
        # Create colormap
        colors = [(0.0, 0.0, 0.2),  # Dark blue
                  (0.2, 0.2, 1.0),  # Bright blue
                  (0.0, 1.0, 1.0),  # Cyan
                  (1.0, 1.0, 0.0),  # Yellow
                  (1.0, 0.5, 0.0),  # Orange
                  (1.0, 0.0, 0.0)]  # Red
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
        
        # Normalize heatmap
        hm = heatmap.copy()
        if hm.max() > hm.min():
            hm = (hm - hm.min()) / (hm.max() - hm.min())
        
        # Save raw heatmap data
        np_path = os.path.join(self.dirs['heatmaps'], f'{slide_id}_{model_name}_heatmap.npy')
        np.save(np_path, heatmap)
        
        # Save as colormap image
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(hm, cmap=cmap, interpolation='bilinear')
        ax.set_title(f'{model_name} Heatmap - {slide_id}', fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        img_path = os.path.join(self.dirs['heatmaps'], f'{slide_id}_{model_name}_heatmap.png')
        plt.savefig(img_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"    Saved individual heatmap: {img_path}")
        
        return img_path
    
    def create_consolidated_heatmaps_only(self, slide_id, wsi_image, heatmaps, annotation_mask, 
                                          metrics_dict, methods_used):
        """Create consolidated figure with only heatmaps (no overlays), plus WSI and ground truth"""
        try:
            self.logger.info(f"  Creating consolidated heatmap figure for {slide_id}...")
            
            n_cols = 3
            n_rows = 3
            
            # Create figure without tight_layout complications
            plt.ioff()  # Turn off interactive mode
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 22))
            fig.subplots_adjust(wspace=0.4, hspace=0.5)
            
            # Flatten axes for easier indexing
            axes = axes.flatten()
            
            # Create colormap for heatmaps
            colors = [(0.0, 0.0, 0.2),  # Dark blue
                      (0.2, 0.2, 1.0),  # Bright blue
                      (0.0, 1.0, 1.0),  # Cyan
                      (1.0, 1.0, 0.0),  # Yellow
                      (1.0, 0.5, 0.0),  # Orange
                      (1.0, 0.0, 0.0)]  # Red
            n_bins = 256
            cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
            
            # 1. Original WSI
            ax = axes[0]
            ax.imshow(wsi_image)
            ax.set_title('Original WSI', fontsize=14, fontweight='bold')
            ax.axis('off')
            
            # 2. Ground truth annotation mask  
            ax = axes[1]
            gt_cmap = LinearSegmentedColormap.from_list('gt', [(0, 0, 0), (0, 1, 0)], N=2)
            ax.imshow(annotation_mask, cmap=gt_cmap, interpolation='nearest')
            ax.set_title('Ground Truth Cancer Regions', fontsize=14, fontweight='bold', color='green')
            ax.axis('off')
            
            # 3-9. Model heatmaps (sorted alphabetically)
            model_names = sorted(heatmaps.keys())
            self.logger.debug(f"    Adding {len(model_names)} model heatmaps")
            
            for i, model_name in enumerate(model_names):
                if i + 2 >= 9:  # We already used positions 0 and 1
                    break
                    
                ax = axes[i + 2]
                heatmap = heatmaps[model_name]
                
                # Normalize heatmap for display
                hm = heatmap.copy()
                if hm.max() > hm.min():
                    hm = (hm - hm.min()) / (hm.max() - hm.min())
                else:
                    hm = np.zeros_like(hm)
                
                # Display heatmap with colorbar
                im = ax.imshow(hm, cmap=cmap, interpolation='bilinear', aspect='equal', 
                              vmin=0, vmax=1)
                
                # Add subtle ground truth contour on heatmap
                try:
                    mask_uint8 = (annotation_mask * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        contour = contour.squeeze()
                        if len(contour.shape) == 2 and contour.shape[0] > 2:
                            ax.plot(contour[:, 0], contour[:, 1], 'g-', linewidth=1, alpha=0.5)
                except Exception as e:
                    self.logger.debug(f"    Could not add contours: {e}")
                
                # Set title with metrics
                method_label = methods_used.get(model_name, "Attention")
                
                if model_name in metrics_dict:
                    m = metrics_dict[model_name]
                    title = f'{model_name} ({method_label})\n'
                    title += f'IoU: {m["iou"]:.3f}, F1: {m["f1_score"]:.3f}\n'
                    title += f'AUC: {m.get("auc_roc", 0):.3f}, AP: {m.get("avg_precision", 0):.3f}'
                    
                    # Color code based on IoU performance
                    if m["iou"] > 0.3:
                        color = 'green'
                    elif m["iou"] > 0.1:
                        color = 'orange'
                    else:
                        color = 'red'
                else:
                    title = f'{model_name} ({method_label})'
                    color = 'black'
                
                ax.set_title(title, fontsize=11, fontweight='bold', color=color)
                ax.axis('off')
                
                # Add a simple colorbar below each heatmap
                # Create a new axis for colorbar
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
            
            # Save without tight bbox to avoid the error
            plt.savefig(save_path, dpi=150, facecolor='white', edgecolor='none', format='png')
            plt.close(fig)
            
            # Verify file was created
            if os.path.exists(save_path):
                file_size = os.path.getsize(save_path)
                self.logger.info(f"  Successfully saved consolidated heatmap ({file_size} bytes)")
            else:
                self.logger.error(f"  ERROR: Consolidated heatmap file was not created!")
                
            if self.use_wandb and os.path.exists(save_path):
                wandb.log({f"{slide_id}_heatmaps": wandb.Image(save_path)})
            
            return save_path
            
        except Exception as e:
            self.logger.error(f"  ERROR creating consolidated heatmap: {e}")
            import traceback
            traceback.print_exc()
            plt.close('all')  # Ensure all figures are closed on error
            return None
    
    def create_heatmap_overlay_refined(self, wsi_image, heatmap, alpha=0.7, min_visible_value=0.01):
        """Create refined overlay with dimension validation"""
        wsi_gray = self.convert_to_grayscale(wsi_image)
        
        if heatmap.shape != wsi_image.shape[:2]:
            self.logger.warning(f"  Resizing heatmap from {heatmap.shape} to {wsi_image.shape[:2]}")
            heatmap = zoom(heatmap,
                          (wsi_image.shape[0] / heatmap.shape[0],
                           wsi_image.shape[1] / heatmap.shape[1]),
                          order=1)
        
        overlay = wsi_gray.copy()
        
        if heatmap.max() <= 0:
            self.logger.warning("  Heatmap has no positive values!")
            return overlay
        
        heatmap_norm = heatmap.copy()
        if heatmap_norm.max() > heatmap_norm.min():
            heatmap_norm = (heatmap_norm - heatmap_norm.min()) / (heatmap_norm.max() - heatmap_norm.min())
        
        mask = heatmap_norm > min_visible_value
        
        if mask.any():
            colors = [(0.2, 0.2, 1.0),  # Bright blue
                     (0.0, 1.0, 1.0),  # Cyan
                     (1.0, 1.0, 0.0),  # Yellow
                     (1.0, 0.5, 0.0),  # Orange
                     (1.0, 0.0, 0.0)]  # Red
            n_bins = 256
            cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
            
            heatmap_colored = cmap(heatmap_norm)[:, :, :3]
            heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
            
            for c in range(3):
                alpha_map = np.clip(heatmap_norm[mask] * alpha * 2.5, 0, 1)
                overlay[mask, c] = (
                    (1 - alpha_map) * wsi_gray[mask, c] +
                    alpha_map * heatmap_colored[mask, c]
                ).astype(np.uint8)
            
            self.logger.info(f"  Overlay created with {mask.sum()} colored pixels "
                           f"({mask.sum()*100/mask.size:.2f}% of image)")
        else:
            self.logger.warning("  No pixels above visibility threshold")
        
        return overlay
    
    def draw_contours_on_image(self, image, mask, color=(0, 255, 0), thickness=3):
        """Draw contours on image"""
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = image.copy()
        cv2.drawContours(result, contours, -1, color, thickness)
        return result
    
    def create_ground_truth_overlay(self, wsi_image, annotation_mask):
        """Create ground truth overlay"""
        overlay = wsi_image.copy()
        
        mask_colored = np.zeros_like(wsi_image)
        mask_colored[:, :, 1] = annotation_mask * 200
        mask_colored[:, :, 0] = annotation_mask * 50
        
        alpha = 0.3
        for c in range(3):
            overlay[annotation_mask, c] = (
                (1 - alpha) * wsi_image[annotation_mask, c] +
                alpha * mask_colored[annotation_mask, c]
            ).astype(np.uint8)
        
        overlay = self.draw_contours_on_image(overlay, annotation_mask, 
                                             color=(0, 255, 0), thickness=4)
        
        return overlay
    
    def create_consolidated_comparison(self, slide_id, wsi_image, heatmaps, annotation_mask, 
                                      metrics_dict, methods_used):
        """Create consolidated 9-diagram comparison figure with overlays"""
        n_cols = 3
        n_rows = 3
        
        fig = plt.figure(figsize=(20, 20))
        
        idx = 1
        
        # 1. Original WSI
        ax = plt.subplot(n_rows, n_cols, idx)
        ax.imshow(wsi_image)
        ax.set_title('Original WSI', fontsize=14, fontweight='bold')
        ax.axis('off')
        idx += 1
        
        # 2. Ground truth
        ax = plt.subplot(n_rows, n_cols, idx)
        gt_overlay = self.create_ground_truth_overlay(wsi_image, annotation_mask)
        ax.imshow(gt_overlay)
        ax.set_title('Ground Truth Cancer Regions', fontsize=14, fontweight='bold', color='green')
        ax.axis('off')
        idx += 1
        
        # 3-9. Model predictions with overlays (sorted alphabetically)
        model_names = sorted(heatmaps.keys())
        for model_name in model_names:
            if idx > 9:
                break
            
            heatmap = heatmaps[model_name]
            ax = plt.subplot(n_rows, n_cols, idx)
            
            overlay = self.create_heatmap_overlay_refined(wsi_image, heatmap, alpha=0.7)
            overlay = self.draw_contours_on_image(overlay, annotation_mask,
                                                 color=(0, 255, 0), thickness=3)
            
            ax.imshow(overlay)
            
            method_label = methods_used.get(model_name, "Attention")
            
            if model_name in metrics_dict:
                m = metrics_dict[model_name]
                title = f'{model_name} ({method_label})\n'
                title += f'IoU: {m["iou"]:.3f}, F1: {m["f1_score"]:.3f}\n'
                title += f'Thresh: {m.get("threshold_type", "NA")} ({m.get("threshold_value", 0):.3f})'
                
                # Color code based on IoU performance
                if m["iou"] > 0.3:
                    color = 'green'
                elif m["iou"] > 0.1:
                    color = 'orange'
                else:
                    color = 'red'
            else:
                title = f'{model_name} ({method_label})'
                color = 'black'
            
            ax.set_title(title, fontsize=11, fontweight='bold', color=color)
            ax.axis('off')
            idx += 1
        
        # Fill remaining slots if needed
        while idx <= 9:
            ax = plt.subplot(n_rows, n_cols, idx)
            ax.axis('off')
            idx += 1
        
        plt.suptitle(f'Model Attention Comparison - {slide_id}\n(Multi-threshold Analysis)', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        save_path = os.path.join(self.dirs['comparisons'], f'{slide_id}_consolidated_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"  Saved consolidated comparison to: {save_path}")
        
        if self.use_wandb:
            wandb.log({f"{slide_id}_comparison": wandb.Image(save_path)})
        
        return save_path
    
    def calculate_metrics_multi_threshold(self, heatmap, annotation_mask):
        """Calculate metrics using multiple thresholds and select best"""
        if heatmap.shape != annotation_mask.shape:
            self.logger.warning(f"  Dimension mismatch: heatmap {heatmap.shape} vs mask {annotation_mask.shape}")
            heatmap = zoom(heatmap,
                         (annotation_mask.shape[0] / heatmap.shape[0],
                          annotation_mask.shape[1] / heatmap.shape[1]),
                         order=1)
        
        # Calculate tumor burden
        tumor_burden = annotation_mask.sum() / annotation_mask.size
        
        # Normalize heatmap to [0, 1]
        hm = heatmap.copy()
        if hm.max() > hm.min():
            hm = (hm - hm.min()) / (hm.max() - hm.min())
        
        # Convert to boolean for logical operations
        mask_bool = annotation_mask.astype(bool)
        
        results = {}
        
        # Adaptive threshold selection based on heatmap characteristics
        non_zero_ratio = (hm > 0.01).sum() / hm.size
        
        if non_zero_ratio < 0.05:
            percentiles = [85, 90, 92, 95, 97, 99]
        elif non_zero_ratio < 0.2:
            percentiles = [75, 80, 85, 90, 95]
        else:
            percentiles = [60, 70, 75, 80, 85, 90]
        
        # Test percentile thresholds
        for percentile in percentiles:
            threshold = np.percentile(hm, percentile)
            heatmap_binary = hm >= threshold
            
            tp = np.float64(np.logical_and(heatmap_binary, mask_bool).sum())
            fp = np.float64(np.logical_and(heatmap_binary, ~mask_bool).sum())
            fn = np.float64(np.logical_and(~heatmap_binary, mask_bool).sum())
            tn = np.float64(np.logical_and(~heatmap_binary, ~mask_bool).sum())
            
            metrics = self._compute_metrics(tp, fp, fn, tn)
            metrics['threshold_type'] = f'percentile_{percentile}'
            metrics['threshold_value'] = float(threshold)
            results[f'p{percentile}'] = metrics
        
        # Otsu threshold
        try:
            if hm.std() > 0:
                otsu_thresh = threshold_otsu(hm)
                heatmap_binary = hm >= otsu_thresh
                
                tp = np.float64(np.logical_and(heatmap_binary, mask_bool).sum())
                fp = np.float64(np.logical_and(heatmap_binary, ~mask_bool).sum())
                fn = np.float64(np.logical_and(~heatmap_binary, mask_bool).sum())
                tn = np.float64(np.logical_and(~heatmap_binary, ~mask_bool).sum())
                
                metrics = self._compute_metrics(tp, fp, fn, tn)
                metrics['threshold_type'] = 'otsu'
                metrics['threshold_value'] = float(otsu_thresh)
                results['otsu'] = metrics
        except:
            pass
        
        # Find best threshold by F1 score
        if results:
            best_method = max(results.keys(), key=lambda k: results[k]['f1_score'])
            best_metrics = results[best_method]
            best_metrics['tumor_burden'] = float(tumor_burden)
            
            # Also calculate threshold-free metrics
            hm_flat = hm.flatten()
            mask_flat = annotation_mask.flatten().astype(float)
            
            try:
                best_metrics['auc_roc'] = float(roc_auc_score(mask_flat, hm_flat))
                best_metrics['avg_precision'] = float(average_precision_score(mask_flat, hm_flat))
            except:
                best_metrics['auc_roc'] = 0.0
                best_metrics['avg_precision'] = 0.0
            
            self.logger.debug(f"    Best threshold: {best_method} (F1={best_metrics['f1_score']:.3f})")
            
            return best_metrics, results
        else:
            return {}, {}
    
    def _compute_metrics(self, tp, fp, fn, tn):
        """Compute standard metrics from confusion matrix"""
        tp = np.float64(tp)
        fp = np.float64(fp)
        fn = np.float64(fn)
        tn = np.float64(tn)
        
        metrics = {
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
            'sensitivity': float(tp / (tp + fn) if (tp + fn) > 0 else 0),
            'specificity': float(tn / (tn + fp) if (tn + fp) > 0 else 0),
            'precision': float(tp / (tp + fp) if (tp + fp) > 0 else 0),
            'f1_score': float(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0),
            'iou': float(tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0),
            'dice': float(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0),
            'accuracy': float((tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0),
            'balanced_accuracy': float(((tp / (tp + fn) if (tp + fn) > 0 else 0) + 
                                      (tn / (tn + fp) if (tn + fp) > 0 else 0)) / 2)
        }
        
        return metrics
    
    # Model loading functions
    def load_acmil(self, checkpoint_path):
        conf = self.load_config()
        model = ACMIL_GA(conf, D=128, droprate=0, n_token=2, n_masked_patch=10, mask_drop=0.6)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        return model.to(self.device).eval()
    
    def load_abmil(self, checkpoint_path):
        conf = self.load_config()
        model = ABMIL(conf, D=128, droprate=0)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        return model.to(self.device).eval()
    
    def load_transmil(self, checkpoint_path):
        conf = self.load_config()
        model = TransMIL(conf)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        return model.to(self.device).eval()
    
    def load_meanmil(self, checkpoint_path):
        conf = self.load_config()
        model = mean_max.MeanMIL(conf)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        return model.to(self.device).eval()
    
    def load_maxmil(self, checkpoint_path):
        conf = self.load_config()
        model = mean_max.MaxMIL(conf)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        return model.to(self.device).eval()
    
    def load_dtfd(self, checkpoint_paths):
        conf = self.load_config()
        classifier = Classifier_1fc(conf.D_inner, conf.n_class, 0).to(self.device)
        attention = Attention(conf.D_inner).to(self.device)
        dimReduction = DimReduction(conf.D_feat, conf.D_inner).to(self.device)
        attCls = Attention_with_Classifier(L=conf.D_inner, num_cls=conf.n_class, droprate=0).to(self.device)
        
        if 'classifier' in checkpoint_paths and os.path.exists(checkpoint_paths['classifier']):
            ckpt = torch.load(checkpoint_paths['classifier'], map_location=self.device, weights_only=False)
            classifier.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt, strict=False)
        if 'attention' in checkpoint_paths and os.path.exists(checkpoint_paths['attention']):
            ckpt = torch.load(checkpoint_paths['attention'], map_location=self.device, weights_only=False)
            attention.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt, strict=False)
        if 'dimReduction' in checkpoint_paths and os.path.exists(checkpoint_paths['dimReduction']):
            ckpt = torch.load(checkpoint_paths['dimReduction'], map_location=self.device, weights_only=False)
            dimReduction.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt, strict=False)
        if 'attCls' in checkpoint_paths and os.path.exists(checkpoint_paths['attCls']):
            ckpt = torch.load(checkpoint_paths['attCls'], map_location=self.device, weights_only=False)
            attCls.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt, strict=False)
        
        return {
            'classifier': classifier.eval(),
            'attention': attention.eval(),
            'dimReduction': dimReduction.eval(),
            'attCls': attCls.eval()
        }
    
    def load_oodml(self, checkpoint_path):
        model = OODML(
            input_dim=384, n_classes=2, K=5, embed_dim=512,
            pseudo_bag_size=512, tau=1.0, dropout=0.1, heads=8, use_ddm=True
        )
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        return model.to(self.device).eval()
    
    # Attention extraction functions
    def get_acmil_attention(self, model, features):
        with torch.no_grad():
            _, _, attn_scores = model(features)
            if len(attn_scores.shape) == 3:
                attn_raw = attn_scores[0].mean(0)
            else:
                attn_raw = attn_scores[0]
            attn_weights = torch.nn.functional.softmax(attn_raw, dim=-1)
            attention = attn_weights.cpu().numpy()
            self.logger.info(f"    ACMIL attention - Min: {attention.min():.6f}, "
                           f"Max: {attention.max():.6f}, Mean: {attention.mean():.6f}")
            return attention
    
    def get_abmil_attention(self, model, features):
        with torch.no_grad():
            x = features[0] if len(features.shape) == 3 else features
            med_feat = model.dimreduction(x)
            A = model.attention(med_feat)
            self.logger.info(f"ABMIL raw A shape: {A.shape}")  # Added for shape debugging
            
            # Dynamically choose softmax dim: the one with the larger size (N patches)
            if len(A.shape) >= 2:
                dim = 0 if A.shape[0] > A.shape[1] else 1
                self.logger.info(f"ABMIL selected softmax dim: {dim}")  # Log the selected dim
            else:
                dim = 0  # Fallback for unexpected shapes
            
            # Apply softmax normalization over patches
            A = torch.nn.functional.softmax(A, dim=dim)
            
            attention = A.squeeze().cpu().numpy()
            
            # If still 2D after squeeze, take the appropriate dimension
            if len(attention.shape) > 1:
                attention = attention[0, :] if attention.shape[0] == 1 else attention[:, 0]
            
            # Analyze distribution for logging
            n_patches = len(attention)
            top_k = min(100, max(10, int(n_patches * 0.01)))
            top_k_indices = np.argpartition(attention, -top_k)[-top_k:]
            top_k_mean = attention[top_k_indices].mean()
            
            self.logger.info(f" ABMIL attention - Min: {attention.min():.6f}, "
                            f"Max: {attention.max():.6f}, Mean: {attention.mean():.6f}")
            self.logger.info(f" ABMIL - Top {top_k} patches mean: {top_k_mean:.6f} "
                            f"({top_k_mean/attention.mean():.1f}x average)")
            self.logger.info(f" ABMIL - Attention sum: {attention.sum():.4f}")
            
            return attention
    
    def get_meanmil_attention(self, features):
        n_instances = features.shape[1] if len(features.shape) == 3 else features.shape[0]
        return np.ones(n_instances) / n_instances
    
    def get_maxmil_attention(self, model, features):
        with torch.no_grad():
            x = features[0] if len(features.shape) == 3 else features
            activations = model.head(x)
            patch_scores = activations.max(dim=1)[0]
            k = min(50, max(5, int(len(patch_scores) * 0.05)))
            top_k_values, top_k_indices = torch.topk(patch_scores, k)
            attention = torch.zeros_like(patch_scores)
            attention[top_k_indices] = torch.nn.functional.softmax(top_k_values, dim=0)
            result = attention.cpu().numpy()
            self.logger.info(f"    MaxMIL attention - Selected {k} patches, Max: {result.max():.6f}")
            return result
    
    def get_dtfd_attention(self, models, features):
        with torch.no_grad():
            x = features[0] if len(features.shape) == 3 else features
            med_feat = models['dimReduction'](x)
            AA = models['attention'](med_feat, isNorm=False).squeeze(0)
            attention = torch.nn.functional.softmax(AA, dim=0)
            result = attention.cpu().numpy()
            self.logger.info(f"    DTFD attention - Min: {result.min():.6f}, "
                           f"Max: {result.max():.6f}, Mean: {result.mean():.6f}")
            return result
    
    def get_transmil_attention_gradcam(self, model, features):
        self.logger.info("    Using GradCAM for TransMIL")
        model.eval()
        features = features.clone().detach().requires_grad_(True)
        logits = model(features)
        if logits.shape[-1] > 1:
            target_score = logits[0, 1]
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
        self.logger.info("    Using GradCAM for OODML")
        model.eval()
        x = features[0] if len(features.shape) == 3 else features
        x = x.clone().detach().requires_grad_(True)
        output_dict = model(x)
        logits = output_dict.get('Y_hat_DM', output_dict.get('Y_hat', None))
        if logits is None:
            self.logger.error("    OODML output not found")
            return np.ones(x.shape[0]) / x.shape[0]
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
        """Process a single slide with all models"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Processing slide: {slide_id}")
        self.logger.info('='*60)
        
        # Check annotation
        annotation_path = os.path.join(self.annotation_dir, f'{slide_id}.xml')
        if not os.path.exists(annotation_path):
            self.logger.warning(f"  No annotation found for {slide_id}")
            return None
        
        # Load WSI
        slide_path = os.path.join(self.slide_dir, f'{slide_id}.tif')
        if not os.path.exists(slide_path):
            self.logger.error(f"  Slide not found: {slide_path}")
            return None
        
        wsi_image, wsi_dims, downsample, wsi = self.load_wsi_region(slide_path, level=5)
        if wsi_image is None:
            return None
        
        # Get grayscale for tissue masking
        wsi_gray = self.convert_to_grayscale(wsi_image)
        
        # Parse annotation
        annotation_groups = self.parse_camelyon_xml(annotation_path, wsi_dims, downsample)
        annotation_mask = self.create_annotation_mask(annotation_groups, wsi_dims, downsample)
        
        # Load features
        features, coords = self.load_features(slide_id)
        if features is None:
            return None
        
        # Process each model
        heatmaps = {}
        metrics_dict = {}
        methods_used = {}
        
        # Process models
        for model_name, checkpoint_path in model_paths.items():
            if model_name == 'dtfd':
                continue
            
            if not os.path.exists(checkpoint_path):
                self.logger.warning(f"  Skipping {model_name}: checkpoint not found")
                continue
            
            try:
                self.logger.info(f"\n  Processing {model_name.upper()}...")
                
                # Create fresh copies
                features_copy = features.clone()
                coords_copy = coords.copy()
                wsi_gray_copy = wsi_gray.copy()
                annotation_mask_copy = annotation_mask.copy()
                
                # Load model and get attention
                if model_name == 'acmil':
                    model = self.load_acmil(checkpoint_path)
                    attention = self.get_acmil_attention(model, features_copy)
                    methods_used[model_name.upper()] = "Attention"
                elif model_name == 'abmil':
                    model = self.load_abmil(checkpoint_path)
                    attention = self.get_abmil_attention(model, features_copy)
                    methods_used[model_name.upper()] = "Attention"
                elif model_name == 'transmil':
                    model = self.load_transmil(checkpoint_path)
                    attention = self.get_transmil_attention_gradcam(model, features_copy)
                    methods_used[model_name.upper()] = "GradCAM"
                elif model_name == 'meanmil':
                    model = self.load_meanmil(checkpoint_path)
                    attention = self.get_meanmil_attention(features_copy)
                    methods_used[model_name.upper()] = "Uniform"
                elif model_name == 'maxmil':
                    model = self.load_maxmil(checkpoint_path)
                    attention = self.get_maxmil_attention(model, features_copy)
                    methods_used[model_name.upper()] = "Attention"
                elif model_name == 'oodml':
                    model = self.load_oodml(checkpoint_path)
                    attention = self.get_oodml_attention_gradcam(model, features_copy)
                    methods_used[model_name.upper()] = "GradCAM"
                else:
                    continue
                
                # Generate heatmap
                heatmap = self.generate_heatmap_refined(
                    attention.copy(),
                    coords_copy,
                    wsi_dims,
                    downsample,
                    wsi_gray_copy,
                    model_name.upper(),
                    wsi
                )
                heatmaps[model_name.upper()] = heatmap
                
                # Save individual heatmap
                self.save_individual_heatmap(heatmap, slide_id, model_name.upper())
                
                # Calculate metrics with multi-threshold
                best_metrics, all_thresholds = self.calculate_metrics_multi_threshold(
                    heatmap.copy(),
                    annotation_mask_copy
                )
                metrics_dict[model_name.upper()] = best_metrics
                
                self.logger.info(f"    IoU: {best_metrics.get('iou', 0):.4f}, F1: {best_metrics.get('f1_score', 0):.4f}")
                self.logger.info(f"    Best threshold: {best_metrics.get('threshold_type', 'NA')} = {best_metrics.get('threshold_value', 0):.3f}")
                self.logger.info(f"    AUC-ROC: {best_metrics.get('auc_roc', 0):.4f}, AP: {best_metrics.get('avg_precision', 0):.4f}")
                
                # Log to WandB
                if self.use_wandb:
                    wandb.log({
                        f"{slide_id}_{model_name}_iou": best_metrics.get('iou', 0),
                        f"{slide_id}_{model_name}_f1": best_metrics.get('f1_score', 0),
                        f"{slide_id}_{model_name}_auc": best_metrics.get('auc_roc', 0),
                        f"{slide_id}_{model_name}_ap": best_metrics.get('avg_precision', 0)
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
                wsi_gray_copy = wsi_gray.copy()
                annotation_mask_copy = annotation_mask.copy()
                
                dtfd_models = self.load_dtfd(dtfd_paths)
                attention = self.get_dtfd_attention(dtfd_models, features_copy)
                methods_used['DTFD'] = "Attention"
                
                heatmap = self.generate_heatmap_refined(
                    attention.copy(),
                    coords_copy,
                    wsi_dims,
                    downsample,
                    wsi_gray_copy,
                    "DTFD",
                    wsi
                )
                heatmaps['DTFD'] = heatmap
                
                # Save individual heatmap
                self.save_individual_heatmap(heatmap, slide_id, 'DTFD')
                
                best_metrics, all_thresholds = self.calculate_metrics_multi_threshold(
                    heatmap.copy(),
                    annotation_mask_copy
                )
                metrics_dict['DTFD'] = best_metrics
                
                self.logger.info(f"    IoU: {best_metrics.get('iou', 0):.4f}, F1: {best_metrics.get('f1_score', 0):.4f}")
                self.logger.info(f"    Best threshold: {best_metrics.get('threshold_type', 'NA')} = {best_metrics.get('threshold_value', 0):.3f}")
                self.logger.info(f"    AUC-ROC: {best_metrics.get('auc_roc', 0):.4f}, AP: {best_metrics.get('avg_precision', 0):.4f}")
                
                if self.use_wandb:
                    wandb.log({
                        f"{slide_id}_dtfd_iou": best_metrics.get('iou', 0),
                        f"{slide_id}_dtfd_f1": best_metrics.get('f1_score', 0),
                        f"{slide_id}_dtfd_auc": best_metrics.get('auc_roc', 0),
                        f"{slide_id}_dtfd_ap": best_metrics.get('avg_precision', 0)
                    })
                
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
            # Create consolidated 9-diagram comparison with overlays
            self.create_consolidated_comparison(slide_id, wsi_image, heatmaps, annotation_mask,
                                               metrics_dict, methods_used)
            
            # Create consolidated heatmaps-only figure
            self.create_consolidated_heatmaps_only(slide_id, wsi_image, heatmaps, annotation_mask,
                                                  metrics_dict, methods_used)
        
        # Save metrics
        metrics_path = os.path.join(self.dirs['metrics'], f'{slide_id}_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        gc.collect()
        
        return metrics_dict

def main():
    # Configuration
    h5_path = '/vol/research/scratch1/NOBACKUP/rk01337/ACMIL/processed_camelyon_20/patch_feats_pretrain_medical_ssl_ViT-S_16.h5'
    slide_dir = '/vol/research/datasets/pathology/Camelyon/Camelyon16/testing/images'
    annotation_dir = '/vol/research/datasets/pathology/Camelyon/Camelyon16/testing/annotations'
    checkpoint_dir = '/vol/research/scratch1/NOBACKUP/rk01337/ACMIL/checkpoint_analysis/CAMELYON16/ssl'
    output_dir = '/vol/research/scratch1/NOBACKUP/rk01337/ACMIL/camelyon_heatmap_multithresh'
    
    # Model paths
    model_paths = {
        'acmil': os.path.join(checkpoint_dir, 'camelyon_acmil_ssl_checkpoint-best.pth'),
        'abmil': os.path.join(checkpoint_dir, 'camelyon_abmil_ssl_checkpoint-best.pth'),
        'transmil': os.path.join(checkpoint_dir, 'camelyon_transmil_ssl_checkpoint-best.pth'),
        'meanmil': os.path.join(checkpoint_dir, 'camelyon_meanmil_ssl_checkpoint-best.pth'),
        'maxmil': os.path.join(checkpoint_dir, 'camelyon_maxmil_ssl_checkpoint-best.pth'),
        'oodml': os.path.join(checkpoint_dir, 'camelyon_oodml_ssl_checkpoint-best.pt'),
    }
    
    # DTFD paths
    dtfd_paths = {
        'classifier': os.path.join(checkpoint_dir, 'camelyon_dtfd_ssl_checkpoint-best_classifier.pth'),
        'attention': os.path.join(checkpoint_dir, 'camelyon_dtfd_ssl_checkpoint-best_attention.pth'),
        'dimReduction': os.path.join(checkpoint_dir, 'camelyon_dtfd_ssl_checkpoint-best_dimReduction.pth'),
        'attCls': os.path.join(checkpoint_dir, 'camelyon_dtfd_ssl_checkpoint-best_attCls.pth'),
    }
    
    # Initialize generator with multi-threshold
    generator = CAMELYONHeatmapGenerator(
        h5_path=h5_path,
        slide_dir=slide_dir,
        checkpoint_dir=checkpoint_dir,
        annotation_dir=annotation_dir,
        output_dir=output_dir,
        patch_size=256,
        extraction_level=1,
        use_wandb=True,
        wandb_project='camelyon-heatmap',
        wandb_run_name='multithresh_analysis'
    )
    
    # Process test slides
    test_slides = ['test_001', 'test_002', 'test_004', 'test_008', 'test_010']
    
    all_metrics = {}
    for slide_id in test_slides:
        try:
            metrics = generator.process_slide(slide_id, model_paths, dtfd_paths)
            if metrics:
                all_metrics[slide_id] = metrics
                generator.logger.info(f"\n{slide_id} results summary:")
                for model, m in metrics.items():
                    generator.logger.info(f"  {model}: IoU={m.get('iou', 0):.3f}, F1={m.get('f1_score', 0):.3f}, "
                                        f"Thresh={m.get('threshold_type', 'NA')}")
        except Exception as e:
            generator.logger.error(f"Failed to process {slide_id}: {e}")
    
    generator.logger.info(f"\nAll results saved in: {generator.run_dir}")
    
    if generator.use_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()
