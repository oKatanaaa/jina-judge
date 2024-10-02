import torch
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from .config import TrainConfig


def train_one_epoch(model, dataloader, optimizer, loss_fn, config: TrainConfig):
    device = config.device
    model.train()  # Set the model to training mode
    running_loss = 0.0
    total_batches = 0  # Initialize a counter for batches
    accumulation_steps = config.gradient_accumulation_steps
    # Create a tqdm progress bar with a dynamic total
    pbar = tqdm(desc="Training")

    # Iterate over batches
    for i, batch in enumerate(dataloader):
        total_batches += 1  # Increment the batch counter

        # Move data to the device
        inputs = batch['prompt']
        labels = batch['score'].to(device)
        
        # Forward pass
        outputs = model(inputs)  # Model expects a list of strings, outputs logits
        loss = loss_fn(outputs, labels)

        loss = loss / accumulation_steps  # Scale loss for gradient accumulation

        # Backward pass
        loss.backward()

        # Accumulate gradients for accumulation_steps batches
        if (i + 1) % accumulation_steps == 0:
            # Apply gradient clipping if specified
            if config.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()  # Reset gradients after updating
            
            # Update progress bar with the loss
            pbar.set_postfix(batch=total_batches, loss=loss.item() * accumulation_steps) # show actual loss

            if config.experiment is not None:
                config.experiment.log_metric("batch_loss", loss.item() * accumulation_steps, step=total_batches)

        running_loss += loss.item() * accumulation_steps

    # Check if there are remaining gradients to update
    if total_batches % accumulation_steps != 0:
        # Apply gradient clipping if specified
        if config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()  # Reset gradients after updating

    epoch_loss = running_loss / total_batches if total_batches > 0 else float('inf')
    pbar.close()  # Close the progress bar
    return epoch_loss


def evaluate_model(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = batch['prompt']
            labels = batch['score'].to(device)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # Compute accuracy, precision, recall, and F1-score
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    
    return accuracy, precision, recall, f1


def train_model(model, dataloader, test_dataloader, optimizer, loss_fn, config: TrainConfig, init_f1=0.0):
    device = config.device
    num_epochs = config.epochs
    save_folder = config.output_dir

    if init_f1 > 0:
        print(f"Initial F1-Score: {init_f1:.4f}")
        
    best_f1 = init_f1
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Train for one epoch
        epoch_loss = train_one_epoch(model, dataloader, optimizer, loss_fn, config)
        print(f"Training Loss: {epoch_loss:.4f}")
        
        # Evaluate the model
        accuracy, precision, recall, f1 = evaluate_model(model, test_dataloader, device)
        print(f"Validation Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}\n")

        if config.experiment is not None:
            config.experiment.log_metric("val_accuracy", accuracy, step=epoch)
            config.experiment.log_metric("val_precision", precision, step=epoch)
            config.experiment.log_metric("val_recall", recall, step=epoch)
            config.experiment.log_metric("val_f1", f1, step=epoch)
            config.experiment.log_metric("train_loss", epoch_loss, step=epoch)

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(save_folder, 'best_model.pth'))
            print("Best model saved.")
        
    # Load best model
    model.load_state_dict(torch.load(os.path.join(save_folder, 'best_model.pth'), map_location=device))
    print("Best model loaded.")
    return model
    
