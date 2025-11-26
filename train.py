import torch
import torch.optim as optim
from pathlib import Path
import argparse
import wandb
from tqdm import tqdm

from dataset.dataset import create_dataloader
from model.model import MarthTransformer, MultiHeadLoss

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    loss_components = {k: 0 for k in ['joystick', 'cstick', 'trigger_l', 'trigger_r', 'buttons']}
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch_idx, (states, actions) in enumerate(pbar):
        states = states.to(device)
        actions = {k: v.to(device) for k, v in actions.items()}
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(states)
        
        # Compute loss
        loss, loss_dict = criterion(predictions, actions)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        for k in loss_components.keys():
            loss_components[k] += loss_dict[k].item()
        
        pbar.set_postfix({'loss': loss.item()})
    
    num_batches = len(train_loader)
    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in loss_components.items()}
    
    return avg_loss, avg_components


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    loss_components = {k: 0 for k in ['joystick', 'cstick', 'trigger_l', 'trigger_r', 'buttons']}
    
    correct = {k: 0 for k in ['joystick', 'cstick', 'trigger_l', 'trigger_r']}
    total = 0
    
    with torch.no_grad():
        for states, actions in val_loader:
            states = states.to(device)
            actions = {k: v.to(device) for k, v in actions.items()}
            
            predictions = model(states)
            loss, loss_dict = criterion(predictions, actions)
            
            total_loss += loss.item()
            for k in loss_components.keys():
                loss_components[k] += loss_dict[k].item()
            
            for k in ['joystick', 'cstick', 'trigger_l', 'trigger_r']:
                pred_classes = torch.argmax(predictions[k], dim=1)
                correct[k] += (pred_classes == actions[k].squeeze()).sum().item()
            
            total += states.size(0)
    
    num_batches = len(val_loader)
    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in loss_components.items()}
    accuracies = {k: v / total for k, v in correct.items()}
    
    return avg_loss, avg_components, accuracies


def main(args):
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config=vars(args)
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"Loading games from {args.game_dir}...")
    train_loader, val_loader = create_dataloader(
        game_dir=args.game_dir,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        train_split=args.train_split,
        num_workers=args.num_workers
    )
    
    model = MarthTransformer(
        state_dim=48,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        seq_len=args.seq_len
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    wandb.config.update({"num_parameters": num_params})
    
    # Watch model with wandb
    wandb.watch(model, log="all", log_freq=100)
    
    criterion = MultiHeadLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )
    
    best_val_loss = float('inf')
    epoch_pbar = tqdm(range(args.epochs), desc="Training Progress")
    for epoch in epoch_pbar:
        train_loss, train_components = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        val_loss, val_components, val_accuracies = validate(
            model, val_loader, criterion, device
        )
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'joy_acc': f'{val_accuracies["joystick"]:.3f}'
        })
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'joy_acc': f'{val_accuracies["joystick"]:.3f}'
        })
        
        # Logging
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Val Accuracies - Joystick: {val_accuracies['joystick']:.3f}, "
                  f"C-stick: {val_accuracies['cstick']:.3f}, "
                  f"Trigger_L: {val_accuracies['trigger_l']:.3f}")
        
        # Log to wandb
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/loss_joystick": train_components['joystick'],
            "train/loss_cstick": train_components['cstick'],
            "train/loss_trigger_l": train_components['trigger_l'],
            "train/loss_trigger_r": train_components['trigger_r'],
            "train/loss_buttons": train_components['buttons'],
            "val/loss": val_loss,
            "val/loss_joystick": val_components['joystick'],
            "val/loss_cstick": val_components['cstick'],
            "val/loss_trigger_l": val_components['trigger_l'],
            "val/loss_trigger_r": val_components['trigger_r'],
            "val/loss_buttons": val_components['buttons'],
            "val/accuracy_joystick": val_accuracies['joystick'],
            "val/accuracy_cstick": val_accuracies['cstick'],
            "val/accuracy_trigger_l": val_accuracies['trigger_l'],
            "val/accuracy_trigger_r": val_accuracies['trigger_r'],
            "learning_rate": current_lr
        })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracies': val_accuracies,
                'args': vars(args)
            }
            torch.save(checkpoint, args.checkpoint_path)
            print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
            
            # Save to wandb
            wandb.save(args.checkpoint_path)
    
    # Log final summary
    wandb.run.summary["best_val_loss"] = best_val_loss
    wandb.run.summary["final_train_loss"] = train_loss
    
    wandb.finish()
    print("\n✓ Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Marth behavior cloning model")
    
    # Data
    parser.add_argument("--game_dir", type=str, required=True, help="Path to directory containing .slp files")
    parser.add_argument("--seq_len", type=int, default=16, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--train_split", type=float, default=0.8, help="Train split fraction")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of DataLoader workers (use 0 for debugging)")
    
    # Model
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--dim_feedforward", type=int, default=1024, help="FFN dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Training
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    
    # Logging
    parser.add_argument("--wandb_project", type=str, default="marth-behavior-cloning", help="W&B project name")
    parser.add_argument("--run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/best_model.pt", help="Checkpoint save path")
    
    args = parser.parse_args()
    
    Path(args.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    
    main(args)
