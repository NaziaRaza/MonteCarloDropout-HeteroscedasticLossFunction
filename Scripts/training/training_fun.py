from Scripts.loss_functions.loss_funcs import *
import numpy as np
import os
import time
import torch

device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_mc_dropout_model(
    model, train_dataloader, val_dataloader,
    num_epochs, learning_rate, lam,
    patience=100, min_lr=5e-5, mse_weights=None,
    checkpoint_path="checkpoints/Noisy_train.pt"
):
    os.makedirs("checkpoints", exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=lam)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=55, gamma=0.95)

    # Resume if checkpoint exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        train_loss_list = checkpoint['train_loss_list']
        val_loss_list = checkpoint['val_loss_list']
        best_loss = checkpoint['best_loss']
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed training from epoch {start_epoch}, best loss = {best_loss:.6f}")
    else:
        train_loss_list, val_loss_list = [], []
        best_loss = float('inf')
        start_epoch = 0

    counter = 0
    it = len(train_loss_list)
    start_time = time.time()

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_losses = []

        for X_batch, y_batch in train_dataloader:
            optimizer.zero_grad()

            train_loss = beta_nll_mod_loss(model, X_batch, y_batch, mse_weights=mse_weights)
            train_losses.append(train_loss.item())

            train_loss.backward()
            optimizer.step()

        train_loss_avg = np.mean(train_losses)

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_val, y_val in val_dataloader:
                val_loss = beta_nll_mod_loss(model, X_val, y_val, mse_weights=mse_weights)
                val_losses.append(val_loss.item())

        val_loss_avg = np.mean(val_losses)

        # Early stopping and checkpointing
        if val_loss_avg < best_loss:
            best_loss = val_loss_avg
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

        # Always save latest state
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': best_loss,
            'train_loss_list': train_loss_list,
            'val_loss_list': val_loss_list
        }, checkpoint_path)

        scheduler.step()
        for param_group in optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'], min_lr)

        train_loss_list.append(train_loss_avg)
        val_loss_list.append(val_loss_avg)
        it += 1

        print(
            f"Epoch {epoch} - Iter {it} - "
            f"Train Loss: {train_loss_avg:.6f} - Val Loss: {val_loss_avg:.6f}  "
            f"LR: {param_group['lr']:.6f} - Time: {time.time() - start_time:.1f}s"
        )

    return train_loss_list, val_loss_list