import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging
from utils.metrics import calculate_metrics


def train_and_evaluate(model, train_loader, test_loader, device, num_epochs=100, fold=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    model_path = f'best_model_fold{fold}.pth'

    # Early stopping mechanism
    best_auc = 0.0
    patience = 15
    no_improve = 0

    scaler = torch.amp.GradScaler()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch in progress:
            # Move data to device
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            # Mixed precision
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)
        logging.info(f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {avg_train_loss:.4f}")

        # Validation
        val_metrics, _, _ = evaluate(model, test_loader, device, criterion)
        logging.info(
            f"Epoch [{epoch + 1}/{num_epochs}] Validation || Acc: {val_metrics['acc']:.4f} | F1: {val_metrics['f1']:.4f} | "
            f"AUC: {val_metrics['auc']:.4f} | AUPR: {val_metrics['aupr']:.4f} | "
            f"Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f} | "
            f"MCC: {val_metrics['mcc']:.4f} | Val Loss: {val_metrics['loss']:.4f}")

        # Update learning rate
        scheduler.step(val_metrics['loss'])

        # Early stopping check
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            no_improve = 0
            torch.save(model.state_dict(), model_path)
            logging.info(f"Saved best model, AUC: {best_auc:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                logging.info(f"Early stopping! No improvement in validation AUC for {patience} consecutive epochs")
                break

    # Load best model
    model.load_state_dict(torch.load(model_path, weights_only=True))
    return model, best_auc


def evaluate(model, test_loader, device, criterion=None):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluation"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)

            outputs = model(inputs)

            if criterion is not None:
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)

            probs = F.softmax(outputs, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    metrics, roc_data, pr_data = calculate_metrics(all_labels, all_preds, all_probs, criterion, test_loader)

    if criterion is not None:
        metrics["loss"] = total_loss / len(test_loader.dataset)

    metrics.pop("roc_data", None)
    metrics.pop("pr_data", None)
    return metrics, roc_data, pr_data