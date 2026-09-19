import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from data import load_config, load_data, training_loader
from epochs import classification_metrics, inference_time, predict, select_threshold, train_epoch
from model import LAVAN
from utils import get_device, set_seed


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--config", default="configs/example.json")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    return parser.parse_args()


def save_json(path, value):
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False, allow_nan=False)


def main():
    args = arguments()
    device = get_device(args.device)
    config = load_config(args.config)
    data = load_data(args.data_dir, config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_loader = DataLoader(data.validation, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(data.test, batch_size=args.batch_size, shuffle=False)
    results = []
    for seed in args.seeds:
        set_seed(seed)
        model = LAVAN(
            parameter_groups=data.parameter_groups,
            height_variable=data.height_variable,
            directional_variables=data.directional_variables,
            discrete_variables=data.discrete_variables,
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        best = {"f1": -1.0, "epoch": 0, "threshold": 0.5, "state": None}
        stale_epochs = 0
        epoch_times = []
        for epoch in range(1, args.epochs + 1):
            loss, elapsed = train_epoch(training_loader(data.train, seed, args.batch_size), model, nn.BCEWithLogitsLoss(), optimizer, device)
            epoch_times.append(elapsed)
            labels, probabilities = predict(validation_loader, model, device)
            threshold, score = select_threshold(labels, probabilities)
            print(f"seed={seed} epoch={epoch:03d} loss={loss:.4f} validation_f1={score:.4f} threshold={threshold:.3f}")
            if score > best["f1"]:
                best = {"f1": score, "epoch": epoch, "threshold": threshold, "state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}}
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= args.patience:
                break
        model.load_state_dict(best["state"])
        torch.save({"model_state_dict": best["state"], "seed": seed, "epoch": best["epoch"], "threshold": best["threshold"], "parameter_groups": data.parameter_groups, "config": config}, output_dir / f"lavan_seed_{seed}.pt")
        labels, probabilities = predict(test_loader, model, device)
        result = {"seed": seed, "best_epoch": best["epoch"], "threshold": best["threshold"], "validation_f1": best["f1"], **classification_metrics(labels, probabilities, best["threshold"]), "train_time_per_epoch_s": float(np.mean(epoch_times)), "inference_time_per_sample_ms": inference_time(test_loader, model, device)}
        results.append(result)
        save_json(output_dir / "results.json", results)
    summary = {}
    for metric in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"):
        values = [result[metric] for result in results if result[metric] is not None]
        summary[metric] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))} if len(values) > 1 else None
    save_json(output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()

