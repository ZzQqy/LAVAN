import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset


FREQUENCIES = (1, 2, 4, 8, 16)


@dataclass
class DataBundle:
    train: TensorDataset
    validation: TensorDataset
    test: TensorDataset
    parameter_groups: dict
    directional_variables: tuple
    discrete_variables: tuple
    height_variable: str
    normalization: dict


def load_config(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def _read_feature(data_dir, feature, label_column, window_seconds):
    path = Path(data_dir) / feature["file"]
    frame = pd.read_csv(path)
    if label_column not in frame:
        raise ValueError(f"Missing label column in {path.name}.")
    labels = frame.pop(label_column).to_numpy(dtype=np.int64)
    columns = list(frame.columns)
    if len(columns) > 1 and all(str(column).startswith("t") for column in columns):
        frame = frame[sorted(columns, key=lambda column: int(str(column)[1:]))]
    values = frame.to_numpy(dtype=np.float32)
    frequency = feature["frequency"]
    if values.shape[1] != window_seconds * frequency:
        raise ValueError(f"Unexpected window size for {feature['name']}.")
    if not np.isfinite(values).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError(f"Invalid values or labels in {path.name}.")
    return labels, torch.from_numpy(values).unsqueeze(1)


def load_data(data_dir, config):
    window_seconds = config["window_seconds"]
    label_column = config["label_column"]
    features = config["features"]
    groups = {frequency: [] for frequency in FREQUENCIES}
    names = {frequency: [] for frequency in FREQUENCIES}
    labels = None
    for feature in features:
        frequency = feature["frequency"]
        if frequency not in FREQUENCIES:
            raise ValueError("Unsupported frequency.")
        current_labels, values = _read_feature(data_dir, feature, label_column, window_seconds)
        if labels is None:
            labels = current_labels
        elif not np.array_equal(labels, current_labels):
            raise ValueError("Feature files do not have identical label ordering.")
        groups[frequency].append(values)
        names[frequency].append(feature["name"])
    if labels is None:
        raise ValueError("No features were configured.")
    count = len(labels)
    train_end, validation_end = int(0.6 * count), int(0.8 * count)
    indices = (np.arange(train_end), np.arange(train_end, validation_end), np.arange(validation_end, count))
    split_inputs = [[], [], []]
    normalization = {}
    directional = set(config.get("directional_variables", []))
    discrete = set(config.get("discrete_variables", []))
    for frequency in FREQUENCIES:
        if not names[frequency]:
            for split, index in zip(split_inputs, indices):
                shape = (len(index), 0, window_seconds) if frequency == 1 else (len(index), 0, window_seconds, frequency)
                split.append(torch.zeros(shape))
            continue
        tensor = torch.cat(groups[frequency], dim=1)
        minimum = tensor[indices[0]].amin(dim=(0, 2), keepdim=True)
        maximum = tensor[indices[0]].amax(dim=(0, 2), keepdim=True)
        scale_mask = torch.tensor([name not in directional | discrete for name in names[frequency]]).view(1, -1, 1)
        normalization[str(frequency)] = {"min": minimum, "max": maximum, "scale_mask": scale_mask}
        for split, index in zip(split_inputs, indices):
            raw = tensor[index]
            scaled = 2 * (raw - minimum) / (maximum - minimum + 1e-12) - 1
            values = torch.where(scale_mask, scaled, raw)
            split.append(values if frequency == 1 else values.reshape(len(index), len(names[frequency]), window_seconds, frequency))
    label_tensor = torch.from_numpy(labels)
    datasets = [TensorDataset(*inputs, label_tensor[index]) for inputs, index in zip(split_inputs, indices)]
    return DataBundle(*datasets, names, tuple(directional), tuple(discrete), config["height_variable"], normalization)


def training_loader(dataset, seed, batch_size):
    labels = dataset.tensors[-1].numpy()
    classes = [np.flatnonzero(labels == value) for value in (0, 1)]
    if any(len(indices) == 0 for indices in classes):
        raise ValueError("Training data must contain both classes.")
    generator = np.random.default_rng(seed)
    target = max(map(len, classes))
    balanced = np.concatenate([np.concatenate([indices, generator.choice(indices, target - len(indices), replace=True)]) for indices in classes])
    return DataLoader(Subset(dataset, balanced.tolist()), batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))

