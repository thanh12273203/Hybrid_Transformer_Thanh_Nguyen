import os
import yaml
import argparse
import warnings

import torch
import torch.multiprocessing as mp

from src.configs import LorentzParTConfig, TrainConfig
from src.engine import Trainer, MaskedModelTrainer
from src.models import LorentzParT
from src.utils import accuracy_metric_ce, set_seed, setup_ddp, cleanup_ddp
from src.utils.data import JetClassDataset, compute_norm_stats, build_memmap_data, load_memmap_data
from src.utils.viz import plot_history, plot_ssl_history

warnings.filterwarnings('ignore')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LorentzParT from YAML config")

    # Model and configurations arguments
    parser.add_argument('--config-path', type=str, default='./configs/train_LorentzParT.yaml', help="Path to YAML config")
    parser.add_argument('--checkpoint-path', type=str, default=None, help="Checkpoint to restore trainer state")

    # Data loading arguments
    parser.add_argument('--train-data-dir', type=str, default='./data/train_100M', help="Train data folder")
    parser.add_argument('--val-data-dir', type=str, default='./data/val_5M', help="Validation data folder")

    return parser.parse_args()


def main(
    rank: int,
    world_size: int,
    config_path: str,
    checkpoint_path: str = None,
    train_data_dir: str = './data/train_100M',
    val_data_dir: str = './data/val_5M'
):
    # Load the YAML file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_config = LorentzParTConfig.from_dict(config['model'])
    train_config = TrainConfig.from_dict(config['train'])

    # Initialize multi-GPU processing
    setup_ddp(rank, world_size)

    # Read in the data
    X_train, y_train = load_memmap_data(train_data_dir, prefix='train')
    X_val, y_val = load_memmap_data(val_data_dir, prefix='val')
    normalize = [True, False, False, True]
    if rank == 0:
        norm_dict = compute_norm_stats(X_val)
    else:
        norm_dict = None
    
    # Broadcast normalization stats to all processes
    obj_list = [norm_dict]
    torch.distributed.broadcast_object_list(obj_list, src=0)
    norm_dict = obj_list[0]

    # Create the dataset
    if model_config.mask:
        train_dataset = JetClassDataset(X_train, y_train, normalize, norm_dict, mask_mode='biased')
        val_dataset = JetClassDataset(X_val, y_val, normalize, norm_dict, mask_mode='biased')
    else:
        train_dataset = JetClassDataset(X_train, y_train, normalize, norm_dict, mask_mode=None)
        val_dataset = JetClassDataset(X_val, y_val, normalize, norm_dict, mask_mode=None)

    # Initialize the model
    model = LorentzParT(config=model_config).to(rank)

    # Initialize the trainer
    if model_config.mask:
        trainer = MaskedModelTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=rank,
            config=train_config
        )
    else:
        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=rank,
            metric=accuracy_metric_ce,
            config=train_config
        )

    # Resume checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        
        try:
            trainer.load_checkpoint(checkpoint_path)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")

    # Train the model
    history, model = trainer.train()

    # Clean up distributed processing
    cleanup_ddp()

    # Save the training history plot
    output_path = os.path.join(trainer.outputs_dir, f"{trainer.run_name}.png") if train_config.save_fig else None
    if model_config.mask:
        plot_ssl_history(history, save_fig=output_path)
    else:
        plot_history(history, save_fig=output_path)


if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()

    # Reproducibility settings
    set_seed(42)

    # Build memory-mapped data files if they do not exist
    build_memmap_data(args.train_data_dir, prefix='train')
    build_memmap_data(args.val_data_dir, prefix='val')

    # Multi-GPU processing
    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(
            main,
            args=(
                world_size,
                args.config_path,
                args.checkpoint_path,
                args.train_data_dir,
                args.val_data_dir
            ),
            nprocs=world_size
        )
    else:
        # 1 GPU or CPU: run the same code on rank 0
        main(
            rank=0,
            world_size=1,
            config_path=args.config_path,
            checkpoint_path=args.checkpoint_path,
            train_data_dir=args.train_data_dir,
            val_data_dir=args.val_data_dir
        )