# LAVAN

PyTorch implementation of LAVAN.

## Project Structure

```text
LAVAN/
├── configs/example.json
├── model/
│   ├── __init__.py
│   └── lavan.py
├── data.py
├── epochs.py
├── utils.py
├── run.py
├── requirements.txt
└── .gitignore
```

- `configs/example.json`: Template for dataset configuration.
- `model/lavan.py`: Defines the LAVAN architecture.
- `model/__init__.py`: Exposes the model package interface.
- `data.py`: Loads and preprocesses the data.
- `epochs.py`: Implements training, inference, and metric computation.
- `utils.py`: Provides random-seed and device utilities.
- `run.py`: Main script for training and evaluation.
- `requirements.txt`: Lists project dependencies.
- `.gitignore`: Defines files excluded from Git tracking.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Copy and complete `configs/example.json`, then run:

```bash
python run.py --data-dir /path/to/data --config configs/your_config.json --output-dir outputs
```

Use `--device cuda:1` only when a specific GPU is required.

## Outputs

Results, summaries, and model checkpoints are saved to the output directory.
