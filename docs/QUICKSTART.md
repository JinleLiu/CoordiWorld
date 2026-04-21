# Quickstart

Install:

```bash
conda create -n coordiworld python=3.10 -y
conda activate coordiworld
pip install -e ".[dev]"
```

Validate synthetic data:

```bash
python -m coordiworld.cli.validate_data --dataset synthetic --max-samples 2
```

Validate the standardized JSONL example:

```bash
python -m coordiworld.cli.validate_data --dataset jsonl --config configs/datasets/jsonl_example.yaml --max-samples 2
```

Run synthetic evaluation dry-run:

```bash
bash scripts/run_eval_synthetic.sh
```

Run tests and lint:

```bash
python -m pytest -q
ruff check src tests
```

No real data, GPU, checkpoints, or benchmark wrappers are required for these commands.
