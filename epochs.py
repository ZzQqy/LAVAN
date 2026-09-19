import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, auc, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score


def train_epoch(loader, model, loss_function, optimizer, device):
    model.train()
    loss_sum = 0.0
    samples = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for batch in loader:
        *inputs, labels = [item.to(device) for item in batch]
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(*inputs), labels.float())
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(labels)
        samples += len(labels)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return loss_sum / samples, time.perf_counter() - start


@torch.no_grad()
def predict(loader, model, device):
    model.eval()
    probabilities, labels = [], []
    for batch in loader:
        *inputs, target = batch
        probabilities.append(torch.sigmoid(model(*[item.to(device) for item in inputs])).cpu())
        labels.append(target.cpu())
    return torch.cat(labels).numpy(), torch.cat(probabilities).numpy()


def select_threshold(labels, probabilities, lower=0.30, upper=0.70, steps=81):
    thresholds = np.linspace(lower, upper, steps)
    scores = [f1_score(labels, probabilities >= threshold, zero_division=0) for threshold in thresholds]
    index = int(np.argmax(scores))
    return float(thresholds[index]), float(scores[index])


def classification_metrics(labels, probabilities, threshold):
    predictions = probabilities >= threshold
    has_both_classes = len(np.unique(labels)) == 2
    pr_auc = None
    if has_both_classes:
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        pr_auc = float(auc(recall, precision))
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if has_both_classes else None,
        "pr_auc": pr_auc,
    }


@torch.no_grad()
def inference_time(loader, model, device, repeats=50):
    model.eval()
    elapsed = 0.0
    samples = 0
    for batch in loader:
        inputs = [item.to(device) for item in batch[:-1]]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(repeats):
            model(*inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed += time.perf_counter() - start
        samples += len(inputs[0]) * repeats
    return elapsed * 1000 / samples

