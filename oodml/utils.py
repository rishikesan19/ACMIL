import argparse
import torch
import os
import numpy as np
from typing import Optional
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

def make_parse():
    """Argument parser for OODML training - aligned with paper specifications."""
    parser = argparse.ArgumentParser(description='OODML: WSI Classification')

    # === Experiment settings ===
    parser.add_argument('--exp_name', type=str, default='oodml_experiment', 
                        help='Name of the experiment for logging and checkpoints')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    
    # === Data settings ===
    parser.add_argument('--csv', type=str, default='./data/camelyon16.csv', 
                        help='Path to CSV file with data splits')
    parser.add_argument('--h5_path', type=str, default='./data/patch_feats.h5', 
                        help='Path to H5 file with extracted features')
    parser.add_argument('--dataset_name', type=str, default='camelyon', 
                        choices=['camelyon', 'bracs', 'tcga_brca', 'tcga_lung', 'tcga_esca'],
                        help='Dataset name')
    parser.add_argument('--cv_root', type=str, default=None, 
                        help='Root for cross-validation fold CSVs')
    parser.add_argument('--fold', type=int, default=None, 
                        help='Fold index for cross-validation (1-based)')
    parser.add_argument('--split', type=int, default=None, 
                        help='Split index (alias for fold)')
    parser.add_argument('--tcga_master_csv', type=str, default=None, 
                        help='Master CSV with labels for TCGA dataset')
    
    # === Model architecture (as per paper) ===
    parser.add_argument('--input_dim', type=int, default=1024, 
                        help='Dimension of input instance features')
    parser.add_argument('--n_classes', type=int, default=2, 
                        help='Number of classes')
    parser.add_argument('--K', type=int, default=5, 
                        help='Size of Adaptive Memory Bank (AMB) - default 5 as per paper')
    parser.add_argument('--embed_dim', type=int, default=512, 
                        help='Internal embedding dimension')
    parser.add_argument('--pseudo_bag_size', type=int, default=512, 
                        help='Number of instances per pseudo-bag M - default 512 as per paper')
    parser.add_argument('--tau', type=float, default=1.0, 
                        help='Temperature for DRPL - τ in Equation 12')
    parser.add_argument('--dropout', type=float, default=0.1, 
                        help='Dropout rate - default 0.1 as per paper')
    parser.add_argument('--use_ddm', action='store_true', default=True,
                        help='Use DDM module (OODML♠) or average (OODML⋆)')
    parser.add_argument('--heads', type=int, default=8,
                        help='Number of attention heads')

    # === Training settings (aligned with paper) ===
    parser.add_argument('--epochs', type=int, default=200, 
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, 
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, 
                        help='Weight decay for optimizer')
    parser.add_argument('--optimizer', type=str, default='adamax', 
                        choices=['adamw', 'adam', 'adamax'],
                        help='Optimizer type - paper uses AdaMax')
    parser.add_argument('--lambda_plce', type=float, default=0.5, 
                        help='Maximum weight for PLCE loss λPLCE')
    parser.add_argument('--warmup_epochs', type=int, default=5, 
                        help='Number of warmup epochs for cosine scheduler')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers for data loading')
    
    # === EMA settings ===
    parser.add_argument('--use_ema', action='store_true', 
                        help='Enable Exponential Moving Average')
    parser.add_argument('--ema_decay', type=float, default=0.995, 
                        help='EMA decay rate')

    # === Checkpoint and logging ===
    parser.add_argument('--best_val_auc_ckpt_path', type=str, default='checkpoints/', 
                        help='Directory for best validation AUC checkpoint')
    parser.add_argument('--use_wandb', action='store_true', 
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_mode', type=str, default='online', 
                        choices=['offline', 'online', 'disabled'],
                        help='W&B logging mode')
    parser.add_argument('--wandb_project', type=str, default='oodml-wsi', 
                        help='W&B project name')
    parser.add_argument('--results_txt', type=str, default=None, 
                        help='File to append results')
    parser.add_argument('--feat_ext', type=str, default='resnet50', 
                        help='Feature extractor name for results logging')

    return parser.parse_args()

def calculate_metrics(labels, probs):
    """Calculate comprehensive evaluation metrics."""
    if len(probs.shape) == 2:
        preds = np.argmax(probs, axis=1)
    else:
        preds = (probs > 0.5).astype(int)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    
    try:
        n_classes = len(np.unique(labels))
        if n_classes == 2:
            # Binary classification
            if len(probs.shape) == 2 and probs.shape[1] == 2:
                auc = roc_auc_score(labels, probs[:, 1])
            else:
                auc = roc_auc_score(labels, probs)
        else:
            # Multi-class classification
            if len(probs.shape) == 2:
                auc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
            else:
                # Convert to one-hot for multi-class
                from sklearn.preprocessing import label_binarize
                labels_onehot = label_binarize(labels, classes=list(range(n_classes)))
                auc = roc_auc_score(labels_onehot, probs, average='macro')
    except (ValueError, IndexError) as e:
        print(f"Warning: Could not compute AUC: {e}")
        auc = 0.5  # Default AUC

    return acc, auc, f1, precision, recall

class ModelCheckpoint:
    """Early stopping and model checkpointing."""
    def __init__(self, patience=25, verbose=True, delta=0, path='checkpoint.pt', metric_name='metric'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_max = -np.Inf
        self.delta = delta
        self.path = path
        self.metric_name = metric_name

    def __call__(self, val_score, model, ema=None):
        """Check if validation score improved and save if it did."""
        score = val_score
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model, ema)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_score, model, ema)
            self.counter = 0

    def save_checkpoint(self, val_score, model, ema):
        """Save model checkpoint."""
        if self.verbose:
            print(f'{self.metric_name} improved ({self.val_score_max:.6f} --> {val_score:.6f}). Saving model...')
        
        state = {'model_state_dict': model.state_dict()}
        if ema:
            state['ema_state_dict'] = ema.state_dict()
            
        torch.save(state, self.path)
        self.val_score_max = val_score

class EMA:
    """Exponential Moving Average for model parameters."""
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        """Register model parameters for EMA."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply EMA parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """Restore original model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self):
        """Get EMA state dict."""
        return self.shadow
        
    def load_state_dict(self, state_dict):
        """Load EMA state dict."""
        self.shadow = state_dict

# ----------------------------
# Results file helper
# ----------------------------
def _ensure_dir_for_file(path: Optional[str]):
    """Ensure directory exists for file."""
    if not path:
        return
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        try:
            os.makedirs(d, exist_ok=True)
        except OSError as e:
            print(f"[WARNING] Failed to create directory for {path}: {e}")


def append_results_line(
    results_txt: Optional[str],
    *,
    dataset: str,
    fold: Optional[int],
    seed: int,
    test_auc: float,
    test_acc: float,
    test_f1: float,
    feat_ext: str
):
    """
    Append one line to results file.
    """
    if not results_txt:
        return
    _ensure_dir_for_file(results_txt)
    fval = 'NA' if fold is None else str(int(fold))
    line = (
        f"dataset={dataset} feat_ext={feat_ext} fold={fval} seed={seed} "
        f"test_auc={float(test_auc):.6f} test_acc={float(test_acc):.4f} test_f1={float(test_f1):.4f}\n"
    )
    try:
        with open(results_txt, 'a') as f:
            f.write(line)
    except OSError as e:
        print(f"[ERROR] Failed to write to {results_txt}: {e}")
