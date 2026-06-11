# OPBDH

Open the Pod Bay Door, Hal: a small command-line tool for running model-backed scripts on RunPod.

OPBDH statically verifies your code, chooses RunPod GPU candidates from a VRAM and hourly-price budget, optionally creates or attaches a network volume for model cache storage, uploads your code, pre-downloads a Hugging Face model, runs your command, monitors it, syncs logs/results into `runpod_results/<run_id>`, and cleans up the pod unless you choose to keep it.

## Install

```bash
pip install opbdh
```

For local development:

```bash
pip install -e ".[dev]"
```

## Configure

Set RunPod credentials:

```bash
export RUNPOD_API_TOKEN="..."
export HF_TOKEN="..." # optional, for private Hugging Face models
```

Create a config interactively:

```bash
opbdh config wizard
```

Or write one directly:

```bash
opbdh config write --model Qwen/Qwen2.5-0.5B-Instruct --code "{cwd}/run.py" --vram-gb 24 --max-spend 5
```

Config is read from `~/.config/opbdh/config.json` plus a local `opbdh.json` or `.opbdh.json`, with local values overriding global ones. String values support `{cwd}`, `{model_id}`, `{model_slug}`, `{run_id}`, `{timestamp}`, `{config_dir}`, and environment variables.

## Run

Use the guided run wizard:

```bash
opbdh run wizard
```

Use the direct path:

```bash
opbdh run now ./run.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --vram-gb 48 \
  --max-dollars-per-hour 2 \
  --max-spend 5
```

There is also a short alias:

```bash
opbdh launch ./run.py --model Qwen/Qwen2.5-0.5B-Instruct --dry-run
```

Your script receives:

- `OPBDH_MODEL_ID`
- `OPBDH_RESULTS_DIR`
- standard Hugging Face cache variables

Write artifacts to `$OPBDH_RESULTS_DIR`; OPBDH syncs remote `logs/` and `results/` back to local `runpod_results/<run_id>`.

## Publish

Recommended path: push this directory to GitHub, create the `opbdh` project on PyPI, and configure PyPI Trusted Publishing for:

- Owner/repository: your GitHub repo
- Workflow file: `publish.yml`
- Environment: `pypi`

Then publish a GitHub Release. The included `.github/workflows/publish.yml` will build and upload without a PyPI password or token.

Manual path:

```bash
python -m pip install -U build twine
python -m build
twine check dist/*
```

Upload to TestPyPI first:

```bash
twine upload --repository testpypi dist/*
```

Then upload to PyPI:

```bash
twine upload dist/*
```
