import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import torch.nn.functional as F
import math
import wandb
from tqdm import tqdm
import random

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets.load_datasets import build_dataset
from oodml.models import OODML
from oodml.utils import make_parse, ModelCheckpoint, calculate_metrics, EMA, append_results_line

def setup_reproducibility(seed=42):
    """Setup reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    """Cosine scheduler for learning rate."""
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    
    schedule = np.concatenate((warmup_schedule, schedule))
    return schedule

def cosine_weight_scheduler(epoch, warmup_epochs, total_epochs, max_lambda):
    """Cosine scheduler for lambda_PLCE as shown in paper."""
    if epoch < warmup_epochs:
        return 0.0
    
    # After warmup, use cosine schedule from 0 to max_lambda
    progress = (epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
    return max_lambda * 0.5 * (1 - np.cos(np.pi * progress))

def train_one_epoch(model, train_loader, optimizer, device, args, epoch, ema=None, lr_schedule=None, iter_count_start=0):
    """Train model for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_loss_er = 0.0
    total_loss_ir = 0.0
    total_loss_dm = 0.0
    total_loss_plce = 0.0
    all_labels = []
    all_probs = []
    
    # Calculate lambda_PLCE using cosine schedule (as per paper)
    lambda_plce = cosine_weight_scheduler(epoch, args.warmup_epochs, args.epochs, args.lambda_plce)
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=True)
    iter_count = iter_count_start
    
    for _, data, label in progress_bar:
        if data.squeeze().numel() == 0:
            continue
            
        data = data.squeeze(0).to(device, dtype=torch.float)
        label = label.to(device, dtype=torch.long)
        
        # Forward pass
        results_dict = model(data, bag_label=label)
        
        Y_hat_ER = results_dict['Y_hat_ER']
        Y_hat_IR = results_dict['Y_hat_IR']
        Y_hat_DM = results_dict['Y_hat_DM']
        
        # Bag-level losses (Equation 16)
        loss_er = F.cross_entropy(Y_hat_ER, label)
        loss_ir = F.cross_entropy(Y_hat_IR, label)
        loss_dm = F.cross_entropy(Y_hat_DM, label)
        
        # Pseudo-label loss (Equation 17)
        pseudo_label_preds = results_dict['pseudo_label_preds'].squeeze()
        drpl_pseudo_labels = results_dict['drpl_pseudo_labels'].squeeze()
        
        loss_plce = torch.tensor(0.0, device=device)
        if pseudo_label_preds.numel() > 0 and drpl_pseudo_labels.numel() > 0:
            # Binary cross-entropy for pseudo-labels
            loss_plce = F.binary_cross_entropy(
                pseudo_label_preds,
                drpl_pseudo_labels.detach()  # Detach to prevent gradient flow
            )
        
        # Total loss (Equation 18 - note paper uses star notation for OODML without DDM)
        if args.use_ddm:
            # OODML with DDM (diamond notation in paper)
            total_batch_loss = loss_dm + lambda_plce * loss_plce
        else:
            # OODML without DDM (star notation in paper)
            total_batch_loss = loss_er + loss_ir + lambda_plce * loss_plce
        
        # Backward pass
        if torch.isnan(total_batch_loss):
            print(f"Warning: NaN loss detected. Skipping batch.")
            continue
            
        optimizer.zero_grad()
        total_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update learning rate
        if lr_schedule is not None and iter_count < len(lr_schedule):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_schedule[iter_count]
            iter_count += 1
        
        # Update EMA
        if ema is not None:
            ema.update()
        
        # Accumulate losses
        total_loss += total_batch_loss.item()
        total_loss_er += loss_er.item()
        total_loss_ir += loss_ir.item()
        total_loss_dm += loss_dm.item()
        total_loss_plce += (lambda_plce * loss_plce).item()
        
        # Store predictions
        all_labels.append(label.cpu().numpy())
        all_probs.append(F.softmax(Y_hat_DM, dim=1).detach().cpu().numpy())
        
        progress_bar.set_postfix(
            loss=f'{total_batch_loss.item():.4f}',
            lr=f'{optimizer.param_groups[0]["lr"]:.6f}',
            λ_plce=f'{lambda_plce:.3f}'
        )
    
    # Calculate metrics
    avg_loss = total_loss / len(train_loader)
    avg_loss_er = total_loss_er / len(train_loader)
    avg_loss_ir = total_loss_ir / len(train_loader)
    avg_loss_dm = total_loss_dm / len(train_loader)
    avg_loss_plce = total_loss_plce / len(train_loader)
    
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    acc, auc, f1, _, _ = calculate_metrics(all_labels, all_probs)
    
    return {
        "loss": avg_loss,
        "loss_er": avg_loss_er,
        "loss_ir": avg_loss_ir,
        "loss_dm": avg_loss_dm,
        "loss_plce": avg_loss_plce,
        "lambda_plce": lambda_plce,
        "acc": acc,
        "auc": auc,
        "f1": f1
    }, iter_count


def validate(model, val_loader, device, args, epoch, ema=None):
    """Validate model."""
    model.eval()
    if ema is not None:
        ema.apply_shadow()
    
    total_loss = 0
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for _, data, label in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation", leave=True):
            if data.squeeze().numel() == 0:
                continue
            
            data = data.squeeze(0).to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.long)
            
            results_dict = model(data, bag_label=label)
            Y_hat_DM = results_dict['Y_hat_DM']
            
            loss = F.cross_entropy(Y_hat_DM, label)
            total_loss += loss.item()

            all_labels.append(label.cpu().numpy())
            all_probs.append(F.softmax(Y_hat_DM, dim=1).cpu().numpy())
            
    if ema is not None:
        ema.restore()
        
    avg_loss = total_loss / len(val_loader)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    acc, auc, f1, _, _ = calculate_metrics(all_labels, all_probs)
    
    return {"loss": avg_loss, "acc": acc, "auc": auc, "f1": f1}


def test(model, test_loader, device, args, epoch, ema=None):
    """Test model."""
    model.eval()
    if ema is not None:
        ema.apply_shadow()
    
    total_loss = 0
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for _, data, label in tqdm(test_loader, desc=f"Epoch {epoch+1} Test", leave=True):
            if data.squeeze().numel() == 0:
                continue
            
            data = data.squeeze(0).to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.long)
            
            results_dict = model(data, bag_label=label)
            Y_hat_DM = results_dict['Y_hat_DM']
            
            loss = F.cross_entropy(Y_hat_DM, label)
            total_loss += loss.item()

            all_labels.append(label.cpu().numpy())
            all_probs.append(F.softmax(Y_hat_DM, dim=1).cpu().numpy())
            
    if ema is not None:
        ema.restore()
        
    avg_loss = total_loss / len(test_loader)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    acc, auc, f1, _, _ = calculate_metrics(all_labels, all_probs)
    
    return {"loss": avg_loss, "acc": acc, "auc": auc, "f1": f1}


def main():
    args = make_parse()
    setup_reproducibility(args.seed)
    
    # Handle split/fold aliasing
    if hasattr(args, 'split') and args.split is not None and args.fold is None:
        args.fold = args.split
    
    # Build experiment name
    split_tag = 'NA' if args.fold is None else str(args.fold)
    auto_exp_name = (
        f"{args.dataset_name}-{args.feat_ext}-"
        f"{'oodml-ddm' if args.use_ddm else 'oodml-star'}-"
        f"seed{args.seed}-split{split_tag}"
    )
    
    if not args.exp_name or args.exp_name == 'oodml_experiment':
        args.exp_name = auto_exp_name

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.exp_name,
            config=args,
            mode='online'
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build datasets
    train_dset, val_dset, test_dset = build_dataset(args)

    # Create data loaders
    train_loader = DataLoader(train_dset, batch_size=1, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # Initialize model
    model_params = {
        'input_dim': args.input_dim,
        'n_classes': args.n_classes,
        'K': args.K,
        'embed_dim': args.embed_dim,
        'pseudo_bag_size': args.pseudo_bag_size,
        'tau': args.tau,
        'dropout': args.dropout,
        'use_ddm': args.use_ddm
    }
    model = OODML(**model_params).to(device)

    # Initialize optimizer (paper uses AdaMax)
    if args.optimizer == 'adamax':
        optimizer = optim.Adamax(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate schedule
    lr_schedule = cosine_scheduler(args.lr, 1e-6, args.epochs, len(train_loader), warmup_epochs=args.warmup_epochs)
    
    # Initialize EMA if enabled
    ema = EMA(model, decay=args.ema_decay) if args.use_ema else None
    
    # Setup checkpoint directory
    ckpt_dir = os.path.join(args.best_val_auc_ckpt_path, args.exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'checkpoint-best.pt')
    
    print(f"Experiment name: {args.exp_name}")
    print(f"Model version: {'OODML♠ (with DDM)' if args.use_ddm else 'OODML⋆ (without DDM)'}")
    print(f"Checkpoints will be saved in: {ckpt_dir}")
    
    # Initialize checkpoint and metrics tracking
    best_val_combined_score = -1.0  # AUC + F1 combined
    best_val_auc = 0.0
    best_val_f1 = 0.0
    best_val_acc = 0.0
    best_test_auc = 0.0
    best_test_f1 = 0.0
    best_test_acc = 0.0
    best_epoch = -1
    
    # Modified checkpointer for strict improvement (no replacement on ties)
    checkpointer = ModelCheckpoint(
        patience=args.patience, 
        verbose=True, 
        path=ckpt_path, 
        metric_name='Val AUC+F1',
        delta=1e-6  # Small delta to avoid replacing on essentially equal scores
    )
    
    test_metrics_at_best = {}

    # Training loop
    iter_count = 0
    for epoch in range(args.epochs):
        # Train
        train_metrics, iter_count = train_one_epoch(
            model, train_loader, optimizer, device, args, epoch, ema, lr_schedule, iter_count
        )
        
        # Validate
        val_metrics = validate(model, val_loader, device, args, epoch, ema)
        
        # Test
        test_metrics = test(model, test_loader, device, args, epoch, ema)
        
        # Calculate combined score (AUC + F1)
        val_combined_score = val_metrics['auc'] + val_metrics['f1']
        
        # Print metrics
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train -> Loss: {train_metrics['loss']:.4f}, "
              f"L_ER: {train_metrics['loss_er']:.4f}, "
              f"L_IR: {train_metrics['loss_ir']:.4f}, "
              f"L_DM: {train_metrics['loss_dm']:.4f}, "
              f"L_PLCE: {train_metrics['loss_plce']:.4f}")
        print(f"  Train Metrics -> Acc: {train_metrics['acc']:.4f}, "
              f"AUC: {train_metrics['auc']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}")
        print(f"  Lambda PLCE: {train_metrics['lambda_plce']:.4f}")
        print(f"  Val   -> Loss: {val_metrics['loss']:.4f}, "
              f"Acc: {val_metrics['acc']:.4f}, "
              f"AUC: {val_metrics['auc']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, "
              f"Combined: {val_combined_score:.4f}")
        print(f"  Test  -> Loss: {test_metrics['loss']:.4f}, "
              f"Acc: {test_metrics['acc']:.4f}, "
              f"AUC: {test_metrics['auc']:.4f}, "
              f"F1: {test_metrics['f1']:.4f}")
        
        # Check if this is  better than previous or equal
        if val_combined_score >= best_val_combined_score:
            best_val_combined_score = val_combined_score
            best_val_auc = val_metrics['auc']
            best_val_f1 = val_metrics['f1']
            best_val_acc = val_metrics['acc']
            best_test_auc = test_metrics['auc']
            best_test_f1 = test_metrics['f1']
            best_test_acc = test_metrics['acc']
            best_epoch = epoch + 1
            test_metrics_at_best = test_metrics
            
            print(f"  🎯 New best validation AUC+F1: {best_val_combined_score:.4f}")
            print(f"     Best Val  -> AUC: {best_val_auc:.4f}, F1: {best_val_f1:.4f}, Acc: {best_val_acc:.4f}")
            print(f"     Best Test -> AUC: {best_test_auc:.4f}, F1: {best_test_f1:.4f}, Acc: {best_test_acc:.4f}")
            
            # Save checkpoint only on strict improvement
            print(f"     Saving best model checkpoint...")
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_combined': best_val_combined_score,
                'best_val_auc': best_val_auc,
                'best_val_f1': best_val_f1,
                'best_val_acc': best_val_acc,
                'best_test_auc': best_test_auc,
                'best_test_f1': best_test_f1,
                'best_test_acc': best_test_acc,
                'args': args
            }
            if ema:
                state['ema_state_dict'] = ema.state_dict()
            torch.save(state, ckpt_path)
            
            # Reset patience counter on improvement
            checkpointer.counter = 0
        else:
            # Increment patience counter when no improvement
            checkpointer.counter += 1
            if checkpointer.counter >= checkpointer.patience:
                checkpointer.early_stop = True

        # Log to wandb
        if args.use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_metrics['loss'],
                "train/loss_er": train_metrics['loss_er'],
                "train/loss_ir": train_metrics['loss_ir'],
                "train/loss_dm": train_metrics['loss_dm'],
                "train/loss_plce": train_metrics['loss_plce'],
                "train/auc": train_metrics['auc'],
                "train/acc": train_metrics['acc'],
                "train/f1": train_metrics['f1'],
                "lambdas/plce": train_metrics['lambda_plce'],
                "val/loss": val_metrics['loss'],
                "val/auc": val_metrics['auc'],
                "val/acc": val_metrics['acc'],
                "val/f1": val_metrics['f1'],
                "val/combined_score": val_combined_score,
                "test/loss": test_metrics['loss'],
                "test/auc": test_metrics['auc'],
                "test/acc": test_metrics['acc'],
                "test/f1": test_metrics['f1'],
                "best/val_combined": best_val_combined_score,
                "best/val_auc": best_val_auc,
                "best/val_f1": best_val_f1,
                "best/test_auc": best_test_auc,
                "best/test_f1": best_test_f1,
                "best/test_acc": best_test_acc
            })

        if checkpointer.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch+1}.")
            print(f"No improvement for {checkpointer.patience} epochs.")
            break
    
    # Save final checkpoint
    final_ckpt_path = os.path.join(ckpt_dir, 'checkpoint-last.pt')
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'final_val_metrics': val_metrics,
        'final_test_metrics': test_metrics
    }
    if ema:
        state['ema_state_dict'] = ema.state_dict()
    torch.save(state, final_ckpt_path)
    
    print("\n" + "="*60)
    print("Training finished!")
    print("="*60)
    print(f"Best epoch: {best_epoch}")
    print(f"Best Validation Combined Score (AUC+F1): {best_val_combined_score:.4f}")
    print(f"Best Validation -> AUC: {best_val_auc:.4f}, F1: {best_val_f1:.4f}, Acc: {best_val_acc:.4f}")
    print(f"Best Test Results (at best validation):")
    print(f"  -> AUC: {best_test_auc:.4f}, F1: {best_test_f1:.4f}, Acc: {best_test_acc:.4f}")
    print("="*60)
    
    # Append results to file
    if args.results_txt and test_metrics_at_best:
        append_results_line(
            args.results_txt,
            dataset=args.dataset_name,
            fold=args.fold,
            seed=args.seed,
            test_auc=best_test_auc,
            test_acc=best_test_acc,
            test_f1=best_test_f1,
            feat_ext=args.feat_ext
        )
        print(f"Results appended to {args.results_txt}")
    
    if args.use_wandb:
        wandb.summary['best_epoch'] = best_epoch
        wandb.summary['best_val_combined'] = best_val_combined_score
        wandb.summary['best_val_auc'] = best_val_auc
        wandb.summary['best_val_f1'] = best_val_f1
        wandb.summary['best_test_auc'] = best_test_auc
        wandb.summary['best_test_f1'] = best_test_f1
        wandb.summary['best_test_acc'] = best_test_acc
        wandb.finish()


if __name__ == '__main__':
    main()
