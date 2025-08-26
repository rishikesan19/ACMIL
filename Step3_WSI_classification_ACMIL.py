#!/usr/bin/env python
import sys
import os

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import yaml
from pprint import pprint
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchmetrics
from timm.utils import accuracy
import wandb
from utils.utils import save_model, Struct, set_seed, MetricLogger
from utils.utils import SmoothedValue, adjust_learning_rate
from datasets.datasets import build_HDF5_feat_dataset
from architecture.transformer import ACMIL_GA, ACMIL_MHA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_arguments():
    parser = argparse.ArgumentParser(
        'ACMIL WSI classification training', add_help=False
    )
    parser.add_argument(
        '--config',
        dest='config',
        default='config/camelyon_config.yml',
        help='settings of dataset in yaml format'
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="set the random seed to ensure reproducibility"
    )
    parser.add_argument(
        '--wandb_mode',
        default='disabled',
        choices=['offline', 'online', 'disabled'],
        help='the model of wandb'
    )
    # Model specific arguments
    parser.add_argument(
        "--n_token",
        type=int,
        default=1,
        help="Number of attention branches in (MBA)."
    )
    parser.add_argument(
        "--n_masked_patch",
        type=int,
        default=0,
        help="Top-K instances to be randomly masked in STKIM."
    )
    parser.add_argument(
        "--mask_drop",
        type=float,
        default=0.6,
        help="Masking ratio in STKIM"
    )
    parser.add_argument(
        "--arch",
        type=str,
        default='ga',
        choices=['ga', 'mha'],
        help="Choice of architecture type"
    )
    parser.add_argument(
        '--pretrain',
        default='medical_ssl',
        choices=[
            'natural_supervised',
            'natural_supervised_Resnet50',
            'medical_ssl',
            'plip',
            'path-clip-B-AAAI',
            'openai-clip-B',
            'openai-clip-L-336',
            'quilt-net',
            'path-clip-B',
            'path-clip-L-336',
            'biomedclip',
            'path-clip-L-768',
            'UNI',
            'GigaPath',
            'resnet50_1024'
        ],
        help='pretrained backbone'
    )
    # Training & Data arguments
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="learning rate"
    )
    parser.add_argument(
        "--cv_root",
        type=str,
        default=None,
        help="Root folder containing fold/split CSVs"
    )
    parser.add_argument(
        "--fold", type=int, default=None, help="Fold index (e.g., 1-5)"
    )
    parser.add_argument(
        "--split",
        type=int,
        default=None,
        help="Split index (e.g., 1-5), used interchangeably with fold"
    )
    # Logging arguments
    parser.add_argument(
        "--results_txt",
        type=str,
        default=None,
        help="Path to a .txt file to append best test results"
    )
    parser.add_argument(
        '--exp_name', type=str, default=None, help='W&B run name (experiment name)'
    )
    parser.add_argument(
        '--wandb_project',
        type=str,
        default='wsi_classification',
        help='W&B project name'
    )
    args = parser.parse_args()
    return args


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, conf):
    model.train()
    metric_logger = MetricLogger(delimiter=" ")
    metric_logger.add_meter(
        'lr', SmoothedValue(window_size=1, fmt='{value:.6f}')
    )
    header = f'Epoch: [{epoch}]'
    print_freq = 100
    for data_it, data in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        image_patches = data['input'].to(device, dtype=torch.float32)
        labels = data['label'].to(device)
        adjust_learning_rate(
            optimizer, epoch + data_it / len(data_loader), conf
        )
        sub_preds, slide_preds, attn = model(image_patches)
        loss1 = criterion(slide_preds, labels)  # Slide-level loss
        # Branch-level loss (if using multiple attention branches)
        loss0 = torch.tensor(0., device=device)
        if conf.n_token > 1:
            loss0 = criterion(
                sub_preds, labels.repeat_interleave(conf.n_token)
            )
        # Diversity loss between attention branches
        diff_loss = torch.tensor(0., device=device)
        if conf.n_token > 1:
            attn = torch.softmax(attn, dim=-1)
            for i in range(conf.n_token):
                for j in range(i + 1, conf.n_token):
                    diff_loss += torch.cosine_similarity(
                        attn[:, i], attn[:, j], dim=-1
                    ).mean()
            diff_loss /= (conf.n_token * (conf.n_token - 1) / 2)
        total_loss = loss1 + loss0 + diff_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        metric_logger.update(
            lr=optimizer.param_groups[0]['lr'],
            slide_loss=loss1.item(),
            sub_loss=loss0.item(),
            diff_loss=diff_loss.item()
        )
        if conf.wandb_mode != 'disabled':
            wandb.log({
                'train/slide_loss': loss1.item(),
                'train/sub_loss': loss0.item(),
                'train/diff_loss': diff_loss.item(),
                'train/lr': optimizer.param_groups[0]['lr']
            })


@torch.no_grad()
def evaluate(net, criterion, data_loader, device, conf, header):
    net.eval()
    y_pred, y_true = [], []
    metric_logger = MetricLogger(delimiter=" ")
    for data in metric_logger.log_every(data_loader, 100, header):
        image_patches = data['input'].to(device, dtype=torch.float32)
        labels = data['label'].to(device)
        sub_preds, slide_preds, attn = net(image_patches)
        loss = criterion(slide_preds, labels)
        pred_probs = torch.softmax(slide_preds, dim=-1)
        acc1 = accuracy(slide_preds, labels, topk=(1,))[0]  # Use logits for accuracy
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=labels.shape[0])
        y_pred.append(pred_probs)
        y_true.append(labels)
    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)
    task = 'multiclass'  # Changed to handle (N, C) preds correctly
    auroc_metric = torchmetrics.AUROC(
        num_classes=conf.n_class, task=task
    ).to(device)
    auroc = auroc_metric(y_pred, y_true).item()
    f1_metric = torchmetrics.F1Score(
        num_classes=conf.n_class, task=task, average='macro'
    ).to(device)
    f1_score = f1_metric(y_pred, y_true).item()
    print(
        f'* Acc@1 {metric_logger.acc1.global_avg:.3f} '
        f'loss {metric_logger.loss.global_avg:.3f} '
        f'auroc {auroc:.3f} f1_score {f1_score:.3f}'
    )
    return auroc, metric_logger.acc1.global_avg, f1_score, metric_logger.loss.global_avg


def main():
    # === 1. Configuration Setup ===
    args = get_arguments()
    with open(args.config, "r") as ymlfile:
        c = yaml.load(ymlfile, Loader=yaml.FullLoader)
        c.update(vars(args))
        conf = Struct(**c)
    # Dictionary for cleaner feature dimension management
    PRETRAIN_DIMS = {
        'medical_ssl': (384, 128),
        'natural_supervised': (512, 256),
        'path-clip-B': (512, 256),
        'openai-clip-B': (512, 256),
        'plip': (512, 256),
        'quilt-net': (512, 256),
        'path-clip-B-AAAI': (512, 256),
        'biomedclip': (512, 256),
        'path-clip-L-336': (768, 384),
        'openai-clip-L-336': (768, 384),
        'UNI': (1024, 512),
        'resnet50_1024': (1024, 512),
        'GigaPath': (1536, 768),
    }
    conf.D_feat, conf.D_inner = PRETRAIN_DIMS.get(
        conf.pretrain, (1024, 512)
    )  # Default fallback
    # === 2. W&B and Checkpoint Setup ===
    split_num = getattr(conf, 'fold', None) or getattr(conf, 'split', None)
    split_tag = 'NA' if split_num is None else str(split_num)
    auto_name = (
        f"{conf.dataset}-{conf.pretrain}-{conf.arch}-"
        f"seed{conf.seed}-split{split_tag}"
    )
    conf.exp_name = args.exp_name or auto_name
    wandb.init(
        project=getattr(args, 'wandb_project', 'wsi_classification'),
        name=conf.exp_name,
        config=vars(conf),  # Log the entire config
        mode=args.wandb_mode
    )
    # Create a unique checkpoint directory
    ckpt_dir = os.path.join('./checkpoints', conf.exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Checkpoints will be saved in: {ckpt_dir}")
    print("Used config:")
    pprint(vars(conf))
    # === 3. Data Loading ===
    set_seed(args.seed)
    safe_pretrain = str(conf.pretrain).replace('/', '_').replace('\\', '_')
    safe_backbone = str(conf.backbone).replace('/', '_').replace('\\', '_')
    h5_path = os.path.join(
        conf.data_dir,
        f'patch_feats_pretrain_{safe_pretrain}_{safe_backbone}.h5'
    )
    train_data, val_data, test_data = build_HDF5_feat_dataset(h5_path, conf)
    train_loader = DataLoader(
        train_data,
        batch_size=conf.B,
        shuffle=True,
        num_workers=conf.n_worker,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_data,
        batch_size=conf.B,
        shuffle=False,
        num_workers=conf.n_worker,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_data,
        batch_size=conf.B,
        shuffle=False,
        num_workers=conf.n_worker,
        pin_memory=True
    )
    # === 4. Model, Loss, and Optimizer Setup ===
    if conf.arch == 'ga':
        model = ACMIL_GA(
            conf,
            n_token=conf.n_token,
            n_masked_patch=conf.n_masked_patch,
            mask_drop=conf.mask_drop
        )
    elif conf.arch == 'mha':
        model = ACMIL_MHA(
            conf,
            n_token=conf.n_token,
            n_masked_patch=conf.n_masked_patch,
            mask_drop=conf.mask_drop
        )
    else:
        raise ValueError(f"Unknown architecture: {conf.arch}")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=conf.lr,
        weight_decay=conf.wd
    )
    # === 5. Training Loop ===
    best_state = {
        'epoch': -1,
        'val_acc': 0,
        'val_auc': 0,
        'val_f1': 0,
        'test_acc': 0,
        'test_auc': 0,
        'test_f1': 0
    }
    for epoch in range(conf.train_epoch):
        train_one_epoch(
            model, criterion, train_loader, optimizer, device, epoch, conf
        )
        val_auc, val_acc, val_f1, val_loss = evaluate(
            model, criterion, val_loader, device, conf, 'Val'
        )
        test_auc, test_acc, test_f1, test_loss = evaluate(
            model, criterion, test_loader, device, conf, 'Test'
        )
        if args.wandb_mode != 'disabled':
            wandb.log({
                'perf/val_acc': val_acc,
                'perf/val_auc': val_auc,
                'perf/val_f1': val_f1,
                'perf/val_loss': val_loss,
                'perf/test_acc': test_acc,
                'perf/test_auc': test_auc,
                'perf/test_f1': test_f1,
                'perf/test_loss': test_loss
            })
        if val_f1 + val_auc > best_state['val_f1'] + best_state['val_auc']:
            best_state.update({
                'epoch': epoch,
                'val_auc': val_auc,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'test_auc': test_auc,
                'test_acc': test_acc,
                'test_f1': test_f1
            })
            save_model(
                conf=conf,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                save_path=os.path.join(ckpt_dir, 'checkpoint-best.pth')
            )
        print('\n')
    save_model(
        conf=conf,
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        save_path=os.path.join(ckpt_dir, 'checkpoint-last.pth')
    )
    print("Results on best epoch:")
    print(best_state)
    if args.results_txt:
        d = os.path.dirname(args.results_txt)
        if d:
            os.makedirs(d, exist_ok=True)
        out_line = (
            f"dataset={conf.dataset} "
            f"fold={split_tag} "
            f"seed={conf.seed} "
            f"test_auc={best_state['test_auc']:.6f} "
            f"test_acc={best_state['test_acc']:.4f} "
            f"test_f1={best_state['test_f1']:.4f}\n"
        )
        with open(args.results_txt, "a") as f:
            f.write(out_line)
    wandb.finish()


if __name__ == '__main__':
    main()