"""
Training script for Asymmetric Matching in Abdominal Lymph Nodes of Follow-up CT Scans.

This script implements the training pipeline for the asymmetric matching framework,
which consists of three main modules:
- SE (Specificity Extraction): Multi-view representation learning with 2D-3D contrastive loss
- TE (Temporal Enhanced): Temporal contrastive learning for cross-CT consistency
- SEAG (Spatial Enhanced Adaptive Graph): Graph neural network for spatial correlations
- SCB (Specificity-Correlation Balancing): Multi-view fusion and thresholded similarity evaluation

The total loss combines: L_total = L_se + L_te + L_seag + L_scb
"""

import argparse
import datetime
import os
import time
from pathlib import Path

import torch
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from torch_geometric.data import DataLoader
from tqdm import tqdm

from src.loss_func import (
    PermutationLoss,
    OffsetLoss,
    CrossEntropyLoss,
    FocalLoss,
    PermutationLossHung,
    HammingLoss,
    ILP_attention_loss,
)
from src.lap_solvers.hungarian import hungarian
from src.utils.model_sl import save_model
from eval import eval_model
from Dataloader import MLNMGraph, collate_fn
from config import cfg
from model import Net
from utils.log_util import get_logger


# ==================== Configuration ====================
# Training hyperparameters
LOSS_FUNC = 'perm'
BATCH_SIZE = 1
STATISTIC_STEP = 1
NUM_EPOCHS = 50
LEARNING_RATE = 0.0001
OPTIMIZER_TYPE = 'adam'
SCHEDULER_MILESTONES = [10, 20]
SCHEDULER_GAMMA = 0.1

# Paths (should be configured via config file or command line arguments)
DEFAULT_CHECKPOINT_PATH = Path('./checkpoints')
DEFAULT_FEAT_SAVE_PATH = Path('./data/feat_gnn')
DEFAULT_OUTPUT_PATH = Path('./COMMON/log')
DEFAULT_CONFIG_FILE = Path('./mlnm_cfgs/mlnm.yaml')


# ==================== Utility Functions ====================
def compute_losses(outputs, criterion):
    """
    Compute all loss components according to the paper.
    
    Args:
        outputs: Model output dictionary
        criterion: Loss function
        
    Returns:
        tuple: (loss_se, loss_te, loss_seag, loss_scb, total_loss)
    """
    # L_seag: SEAG matching loss (perm_mat)
    loss_seag = criterion(
        outputs['perm_mat'],
        outputs['gt_perm_mat'],
        outputs['ns'][0],
        outputs['ns'][1]
    )
    
    # L_te: Temporal Enhanced loss (encoder matching loss)
    loss_te = criterion(
        outputs['ds_mat'],
        outputs['gt_encoder_mat'],
        outputs['ns_encoder'][0],
        outputs['ns_encoder'][1]
    )
    
    # L_se: Specificity Extraction loss (from model output)
    loss_se = outputs['loss']
    
    # L_scb: SCB loss (mix_mat matching loss)
    loss_scb = criterion(
        outputs['mix_mat'],
        outputs['gt_perm_mat'],
        outputs['ns'][0],
        outputs['ns'][1]
    )
    
    # Total loss: L_total = L_se + L_te + L_seag + L_scb
    total_loss = loss_se + loss_te + loss_seag + loss_scb
    
    return loss_se, loss_te, loss_seag, loss_scb, total_loss


def compute_accuracy_metrics(outputs, x_seag, x_encoder, x_scb):
    """
    Compute accuracy metrics for SEAG, TE, and SCB modules.
    
    Args:
        outputs: Model output dictionary
        x_seag: SEAG matching results
        x_encoder: TE matching results
        x_scb: SCB matching results
        
    Returns:
        dict: Dictionary containing accuracy metrics
    """
    batch_size = len(outputs['ns'][1])
    metrics = {
        'seag': {'tp': 0, 'tp_y': 0, 'nodes': 0, 'nodes_y': 0},
        'encoder': {'tp': 0, 'tp_y': 0, 'nodes': 0, 'nodes_y': 0},
        'scb': {'tp': 0, 'nodes': 0}
    }
    
    for i in range(batch_size):
        n0, n1 = outputs['ns'][0][i], outputs['ns'][1][i]
        gt_slice = outputs['gt_perm_mat'][i][:n0, :n1]
        
        # SEAG accuracy
        tp, n_node = cal_acc(x_seag[i][:n0, :n1], gt_slice)
        metrics['seag']['tp'] += tp
        metrics['seag']['nodes'] += n_node
        if outputs['id_list'][i] == 2:
            metrics['seag']['tp_y'] += tp
            metrics['seag']['nodes_y'] += n_node
        
        # Encoder accuracy
        tp1, n_node1 = cal_acc(x_encoder[i][:n0, :n1], gt_slice)
        metrics['encoder']['tp'] += tp1
        metrics['encoder']['nodes'] += n_node1
        if outputs['id_list'][i] == 2:
            metrics['encoder']['tp_y'] += tp1
            metrics['encoder']['nodes_y'] += n_node1
        
        # SCB accuracy
        tp2, n_node2 = cal_acc(x_scb[i][:n0, :n1], gt_slice)
        metrics['scb']['tp'] += tp2
        metrics['scb']['nodes'] += n_node2
    
    return metrics


def cal_acc(x, gt):
    """
    Calculate matching accuracy between predicted and ground truth permutation matrices.
    
    Args:
        x: Predicted permutation matrix
        gt: Ground truth permutation matrix
        
    Returns:
        correct_count: Number of correct matches
        total_count: Total number of nodes
    """
    x = x.squeeze(0)
    gt = gt.squeeze(0)
    
    if x.shape[0] > x.shape[1]:
        x = x.T
        gt = gt.T
    
    # Check one-to-one matching
    pred_indices = x.max(1).indices
    gt_indices = gt.max(1).indices
    correct_matches = (pred_indices == gt_indices)
    
    return int(correct_matches.sum()), correct_matches.shape[0]


def parse_yaml(file_path):
    """Parse YAML configuration file."""
    with open(file_path) as f:
        return yaml.safe_load(f)


def safe_divide(numerator, denominator):
    """Safely divide, returning 0 if denominator is 0."""
    return numerator / denominator if denominator > 0 else 0


# ==================== Training Function ====================
def train_model(
    model,
    criterion,
    optimizer,
    dataloader,
    eval_dataloader,
    tfboardwriter,
    logger,
    num_epochs=50,
    start_epoch=0,
    checkpoint_path=None,
    resume_path=None,
):
    """
    Main training function.
    
    According to the paper, the total loss is:
    L_total = L_se + L_te + L_seag + L_scb
    
    where:
    - L_se: Specificity Extraction loss (2D-3D contrastive loss)
    - L_te: Temporal Enhanced loss (temporal contrastive loss)
    - L_seag: SEAG loss (graph matching loss)
    - L_scb: SCB loss (binary cross-entropy loss)
    """
    logger.info("=" * 60)
    logger.info("Starting training...")
    since = time.time()
    device = next(model.parameters()).device
    logger.info(f'Model on device: {device}')
    
    # Load pretrained model if specified
    if resume_path and os.path.exists(resume_path):
        logger.info(f"Loading model parameters from {resume_path}")
        model.load_state_dict(torch.load(resume_path, map_location=device), strict=False)
    
    # Initialize training state
    best_acc = 0
    best_loss = float('inf')
    best_acc_epoch = 0
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=SCHEDULER_MILESTONES,
        gamma=SCHEDULER_GAMMA,
        last_epoch=start_epoch - 1
    )
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        torch.manual_seed(cfg.RANDOM_SEED + epoch + 1)
        model.train()
        
        # Get learning rate
        lr_str = ", ".join([f"{x['lr']:.2e}" for x in optimizer.param_groups])
        logger.info(f'Epoch {epoch}/{num_epochs - 1} - Learning rate: {lr_str}')
        
        # Initialize epoch statistics
        epoch_losses = {
            'total': 0.0,
            'se': 0.0,
            'te': 0.0,
            'seag': 0.0,
            'scb': 0.0
        }
        
        # Accuracy accumulators
        accuracy_stats = {
            'seag': {'tp': 0.0, 'tp_y': 0.0, 'nodes': 0.0, 'nodes_y': 0.0},
            'encoder': {'tp': 0.0, 'tp_y': 0.0, 'nodes': 0.0, 'nodes_y': 0.0},
            'scb': {'tp': 0.0, 'nodes': 0.0}
        }
        
        running_loss = 0.0
        iter_num = 0
        epoch_iter = len(dataloader)
        
        # Training iteration
        with tqdm(total=len(dataloader)) as pbar:
            for inputs in dataloader:
                iter_num += 1
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(True):
                    # Forward pass
                    outputs = model(data_dict=inputs, training=True, iter_num=iter_num, epoch=epoch)
                    
                    # Verify required outputs (only in debug mode)
                    if not all(key in outputs for key in ['ds_mat', 'perm_mat', 'gt_perm_mat', 'mix_mat', 'loss']):
                        missing = [key for key in ['ds_mat', 'perm_mat', 'gt_perm_mat', 'mix_mat', 'loss'] if key not in outputs]
                        raise KeyError(f"Missing required outputs: {missing}")
                    
                    # Compute all loss components
                    loss_se, loss_te, loss_seag, loss_scb, total_loss = compute_losses(
                        outputs, criterion
                    )
                    
                    # Backward pass
                    total_loss.backward()
                    optimizer.step()
                
                # Compute accuracy using Hungarian algorithm
                x_seag = hungarian(outputs['perm_mat'], outputs['ns'][0], outputs['ns'][1])
                x_encoder = hungarian(outputs['ds_mat'], outputs['ns'][0], outputs['ns'][1])
                x_scb = outputs['mix_mat']
                
                # Calculate accuracy metrics
                batch_metrics = compute_accuracy_metrics(outputs, x_seag, x_encoder, x_scb)
                
                # Accumulate statistics
                for module in ['seag', 'encoder', 'scb']:
                    for key in batch_metrics[module]:
                        accuracy_stats[module][key] += batch_metrics[module][key]
                
                # Calculate running averages
                seag_stats = accuracy_stats['seag']
                encoder_stats = accuracy_stats['encoder']
                scb_stats = accuracy_stats['scb']
                
                acc_avg = safe_divide(seag_stats['tp'], seag_stats['nodes'])
                acc_avg_y = safe_divide(seag_stats['tp_y'], seag_stats['nodes_y'])
                acc_avg_encoder = safe_divide(encoder_stats['tp'], encoder_stats['nodes'])
                acc_avg_encoder_y = safe_divide(encoder_stats['tp_y'], encoder_stats['nodes_y'])
                acc_avg_mix = safe_divide(scb_stats['tp'], scb_stats['nodes'])
                
                # Update epoch losses
                running_loss = total_loss.item()
                epoch_losses['total'] += total_loss.item() * BATCH_SIZE
                epoch_losses['se'] += loss_se.item()
                epoch_losses['te'] += loss_te.item()
                epoch_losses['seag'] += loss_seag.item()
                epoch_losses['scb'] += loss_scb.item()
                
                # TensorBoard logging
                global_step = epoch * epoch_iter + iter_num
                loss_dict = {
                    'total': total_loss.item(),
                    'se': loss_se.item(),
                    'te': loss_te.item(),
                    'seag': loss_seag.item(),
                    'scb': loss_scb.item(),
                }
                tfboardwriter.add_scalars('loss', loss_dict, global_step)
                tfboardwriter.add_scalars('accuracy', {'matching_accuracy': acc_avg}, global_step)
                
                # Update progress bar
                if iter_num % STATISTIC_STEP == 0:
                    pbar.update(1)
                    pbar.set_description(
                        f"[Train] Epoch:{epoch}, Iter:{iter_num}, "
                        f"Loss:{running_loss:.4f}, "
                        f"Acc:{acc_avg:.4f}, Acc_y:{acc_avg_y:.4f}, "
                        f"Acc_encoder:{acc_avg_encoder:.4f}, Acc_encodery:{acc_avg_encoder_y:.4f}, "
                        f"Acc_mix:{acc_avg_mix:.4f}, "
                        f"tp:{int(accuracy_stats['seag']['tp'])}, nodes:{int(accuracy_stats['seag']['nodes'])}"
                    )
                    running_loss = 0.0
                
        
        # Calculate epoch averages
        epoch_loss_avg = epoch_losses['total'] / epoch_iter / BATCH_SIZE
        epoch_loss_se_avg = epoch_losses['se'] / epoch_iter
        epoch_loss_te_avg = epoch_losses['te'] / epoch_iter
        epoch_loss_seag_avg = epoch_losses['seag'] / epoch_iter
        epoch_loss_scb_avg = epoch_losses['scb'] / epoch_iter
        
        # Log epoch summary
        logger.info(
            f"[Train] Epoch:{epoch}, Iter:{iter_num}, "
            f"Loss:{epoch_loss_avg:.4f}, "
            f"L_se:{epoch_loss_se_avg:.4f}, "
            f"L_te:{epoch_loss_te_avg:.4f}, "
            f"L_seag:{epoch_loss_seag_avg:.4f}, "
            f"L_scb:{epoch_loss_scb_avg:.4f}, "
            f"Acc:{acc_avg:.4f}, Acc_y:{acc_avg_y:.4f}, "
            f"Acc_encoder:{acc_avg_encoder:.4f}, Acc_encodery:{acc_avg_encoder_y:.4f}, "
            f"ACC_mix:{acc_avg_mix:.4f}, "
            f"tp:{int(accuracy_stats['seag']['tp'])}, nodes:{int(accuracy_stats['seag']['nodes'])}"
        )
        
        # Evaluation
        is_last_epoch = (epoch == num_epochs - 1)
        eval_results = eval_model(epoch, model, eval_dataloader, is_last_epoch)
        (
            acc_avg, acc_avg_y, acc_avg_encoder, acc_avg_encoder_y, acc_avg_mix,
            rec, pre, nlr, nlr1, nlr11, dlr, dlr1, dlr11, lnma, aa, bb
        ) = eval_results
        
        tfboardwriter.add_scalar('eval/accuracy', acc_avg, epoch)
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        if checkpoint_path and not checkpoint_path.exists():
            checkpoint_path.mkdir(parents=True)
        
        # Save best model
        if acc_avg > best_acc or (acc_avg == best_acc and epoch_loss_avg < best_loss):
            best_acc = acc_avg
            best_acc_epoch = epoch
            best_loss = epoch_loss_avg
            
            if checkpoint_path:
                save_model(model, str(checkpoint_path / 'params.pt'))
                torch.save(optimizer.state_dict(), str(checkpoint_path / 'optim.pt'))
            
            logger.info(
                f'[Eval] Best ACC: {best_acc:.4f} at epoch {best_acc_epoch} with loss {best_loss:.4f}, '
                f'Acc_y:{acc_avg_y:.4f}, Acc_encoder:{acc_avg_encoder:.4f}, '
                f'Acc_encodery:{acc_avg_encoder_y:.4f}, ACC_mix:{acc_avg_mix:.4f}, '
                f'tp_sum:{aa}, node:{bb}, Rec:{rec:.4f}, Pre:{pre:.4f}, '
                f'Nlr:{nlr:.4f}, Nlr_mix:{nlr1:.4f} {nlr11:.4f}, '
                f'Dlr:{dlr:.4f}, Dlr_mix:{dlr1:.4f} {dlr11:.4f}, Lnma:{lnma:.4f}'
            )
    
    time_elapsed = time.time() - since
    hours = int(time_elapsed // 3600)
    minutes = int((time_elapsed // 60) % 60)
    seconds = int(time_elapsed % 60)
    logger.info(f'Training complete in {hours}h {minutes}m {seconds}s')
    logger.info("=" * 60)
    
    return model


# ==================== Main Function ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Lymph Node Matching Training')
    parser.add_argument('--dim', type=int, default=128,
                        help='Dimension of LN feature. default=128')
    parser.add_argument('--gpu', default='0', type=str,
                       help='GPU device ID. default: 0')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config YAML file')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                       help='Path to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to resume checkpoint')
    parser.add_argument('--num_epochs', type=int, default=NUM_EPOCHS,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                       help='Batch size')
    
    args = parser.parse_args()
    
    # Set device
    gpu_id = args.gpu
    torch.cuda.set_device(f'cuda:{gpu_id}')
    device = torch.device(f'cuda:{gpu_id}')
    torch.manual_seed(cfg.RANDOM_SEED)
    
    # Parse configuration
    config_file = args.config if args.config else DEFAULT_CONFIG_FILE
    config = parse_yaml(config_file)
    data_params = config['data']
    
    # Setup paths
    checkpoint_path = (
        Path(args.checkpoint_path) if args.checkpoint_path else DEFAULT_CHECKPOINT_PATH
    )
    feat_save_path = DEFAULT_FEAT_SAVE_PATH
    output_path = DEFAULT_OUTPUT_PATH
    
    # Create output directory
    if not output_path.exists():
        output_path.mkdir(parents=True)
    
    # Load datasets
    dataset = MLNMGraph(
        records_path=data_params['train_records_path'],
        config=data_params,
        phase='train',
        data_root=str(feat_save_path),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    eval_dataset = MLNMGraph(
        records_path=data_params['eval_records_path'],
        config=data_params,
        phase='val',
        data_root=str(feat_save_path)
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False
    )
    
    # Initialize model
    model = Net()
    model = model.cuda()
    
    # Initialize loss function
    loss_func_map = {
        'offset': lambda: OffsetLoss(norm=cfg.TRAIN.RLOSS_NORM),
        'perm': lambda: PermutationLoss(),
        'ce': lambda: CrossEntropyLoss(),
        'focal': lambda: FocalLoss(alpha=0.5, gamma=0.),
        'hung': lambda: PermutationLossHung(),
        'hamming': lambda: HammingLoss(),
        'ilp': lambda: ILP_attention_loss(),
    }
    
    if LOSS_FUNC not in loss_func_map:
        raise ValueError(f'Unknown loss function: {LOSS_FUNC}. '
                        f'Available options: {list(loss_func_map.keys())}')
    criterion = loss_func_map[LOSS_FUNC]()
    
    # Initialize optimizer
    model_params = model.parameters()
    if OPTIMIZER_TYPE == 'sgd':
        optimizer = optim.SGD(
            model_params,
            lr=LEARNING_RATE,
            momentum=cfg.TRAIN.MOMENTUM,
            nesterov=True
        )
    elif OPTIMIZER_TYPE == 'adam':
        optimizer = optim.Adam(model_params, lr=LEARNING_RATE)
    else:
        raise ValueError(f'Unknown optimizer: {OPTIMIZER_TYPE}. '
                        f'Available options: sgd, adam')
    
    # Setup logging
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    experiment_name = f'experiment_{timestamp}'
    
    tfboardwriter = SummaryWriter(
        logdir=str(output_path / 'tensorboard' / f'training_{timestamp}')
    )
    logger = get_logger(str(output_path / f'{experiment_name}.log'))
    
    # Log configuration
    logger.info("Training Configuration:")
    logger.info(f"  Loss function: {LOSS_FUNC}")
    logger.info(f"  Batch size: {BATCH_SIZE}")
    logger.info(f"  Learning rate: {LEARNING_RATE}")
    logger.info(f"  Optimizer: {OPTIMIZER_TYPE}")
    logger.info(f"  Number of epochs: {args.num_epochs}")
    
    # Start training
    model = train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        dataloader=dataloader,
        eval_dataloader=eval_dataloader,
        tfboardwriter=tfboardwriter,
        logger=logger,
        num_epochs=args.num_epochs,
        start_epoch=0,
        checkpoint_path=checkpoint_path,
        resume_path=args.resume,
    )
