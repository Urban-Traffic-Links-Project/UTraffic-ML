"""
Resume Training Script
Resume training from last checkpoint when interrupted
"""

import os
from pathlib import Path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.train_and_compare import main
import argparse


def resume_training(checkpoint_dir='checkpoints/T-GCN'):
    """
    Resume training from checkpoint
    
    Args:
        checkpoint_dir: Directory containing checkpoints
    """
    # Checkpoint directory
    base_dir = Path(__file__).resolve().parents[2] / "checkpoints" / "T-GCN"
    checkpoint_path = os.path.join(base_dir, 'last_checkpoint.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ No checkpoint found at: {checkpoint_path}")
        print("Starting fresh training instead...")
        return False
    
    print(f"\n{'='*80}")
    print(f"Resuming Training from Checkpoint")
    print(f"{'='*80}")
    print(f"Checkpoint: {checkpoint_path}")
    
    import torch
    checkpoint = torch.load(checkpoint_path)
    
    print(f"\nCheckpoint Info:")
    print(f"  Epoch: {checkpoint['epoch'] + 1}")
    print(f"  Best Val Loss: {checkpoint['best_val_loss']:.6f}")
    print(f"  Train Losses: {len(checkpoint['train_losses'])} epochs")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resume T-GCN Training')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/T-GCN',
                       help='Directory containing checkpoints')
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    has_checkpoint = resume_training(args.checkpoint_dir)
    
    # Run main with resume=True
    if has_checkpoint:
        import importlib
        import experiments.train_and_compare as train_module
        
        # Modify config to enable resume
        original_main = train_module.main
        
        def main_with_resume():
            # This will be called instead of original main
            # You can modify the config here
            original_main()
        
        print("\n✓ Resuming training...")
        print("Note: Make sure to set config['resume'] = True in train_and_compare.py")
        print("\nOr run:")
        print("  python experiments/train_and_compare.py")
        print("  with resume=True in config")
    else:
        print("\n✓ No checkpoint found, will start fresh training")