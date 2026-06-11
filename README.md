# OPBDH

**O**pen the **P**od **B**ay **D**oor, **H**al.

In *2001: A Space Odyssey*, Dave asks HAL to open the pod bay doors and gets the most famous refusal in cinema: *"I'm sorry, Dave, I'm afraid I can't do that."*

OPBDH is the obliging counterpart: a small command-line tool that, when asked, actually opens the pod bay door — it launches a [RunPod](https://www.runpod.io/) GPU pod, runs your model-backed script on it, brings your results home, and shuts the pod down.

## What it does

One `opbdh launch` performs the whole mission:

1. **Verifies your code statically** (Python byte-compiles, shell scripts pass `bash -n`) before any money is spent.
2. **Picks GPU candidates** from a VRAM requirement and an optional hourly-price cap.
3. **Optionally creates or attaches a network volume** sized from the Hugging Face model's actual weight files, so model downloads survive across runs.
4. **Creates the pod, uploads your code**, pre-downloads the model into the cache, and runs your command.
5. **Monitors the run**, continuously syncing remote `logs/` and `results/` into a local `runpod_results/<run_id>/` directory.
6. **Enforces a spend guard** — if the estimated cost of the run crosses `max_spend_dollars`, the job is stopped.
7. **Deletes the pod** when the run finishes (or fails), unless you ask to keep it.

## Install

```bash
pip install opbdh
```

Requirements: macOS or Linux, Python ≥ 3.11, and `ssh`/`scp` on your `PATH`. An SSH keypair is used to reach the pod (`~/.ssh/id_ed25519` by default; configurable).

## Configure

Set your RunPod credentials:

```bash
export RUNPOD_API_TOKEN="..."
export HF_TOKEN="..."   # optional, for private/gated Hugging Face models
```

Create a config interactively:

```bash
opbdh config wizard
```

Or write one directly:

```bash
opbdh config write --model Qwen/Qwen2.5-0.5B-Instruct --code "{cwd}/run.py" --vram-gb 24 --max-spend 5
```

Config is merged from the global `~/.config/opbdh/config.json` and a local `opbdh.json` or `.opbdh.json` (discovered upward from the working directory), with local values overriding global ones and command-line flags overriding both.

String values support placeholders — `{cwd}`, `{model_id}`, `{model_slug}`, `{run_id}`, `{timestamp}`, `{config_dir}` — and environment variables (`$VAR` / `${VAR}`).

## Run

The guided way:

```bash
opbdh run wizard
```

The direct way:

```bash
opbdh run now ./run.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --vram-gb 48 \
  --max-dollars-per-hour 2 \
  --max-spend 5
```

Or the short alias:

```bash
opbdh launch ./run.py --model Qwen/Qwen2.5-0.5B-Instruct --dry-run
```

`--dry-run` verifies your code and prints the full plan without contacting RunPod. Without `--yes`, every real launch shows the estimated hourly price and asks for confirmation before any billable compute starts.

### Inspecting before you fly

```bash
opbdh plan ./run.py --model Org/Model     # show the plan for a run
opbdh verify ./run.py                     # static checks only
opbdh gpus --vram-gb 48                   # see GPU candidates and price estimates
opbdh models search qwen                  # search Hugging Face models
opbdh models size Qwen/Qwen2.5-0.5B-Instruct   # weight size + suggested volume
```

## Your script's environment

On the pod, your code runs from `/opbdh-run/user/` with:

- `OPBDH_MODEL_ID` — the configured Hugging Face model id
- `OPBDH_RESULTS_DIR` — write your artifacts here
- The standard Hugging Face cache variables (`HF_HOME`, `HUGGINGFACE_HUB_CACHE`, …) pointed at the pod or network-volume cache, with the model already downloaded if `pre_download_model` is on
- `HF_TOKEN`, if set locally — passed to the job as a session environment variable, never written to disk on the pod
- A `requirements.txt` next to your code is pip-installed automatically

Anything your job writes to `logs/` and `results/` under `/opbdh-run` is synced back to local `runpod_results/<run_id>/` while the run is in progress and again at the end (including on failure).

`.env` files and the usual junk (`.git`, `.venv`, `__pycache__`, `node_modules`, …) are never uploaded.

## The eye

In an interactive terminal, long waits (pod boot, SSH, upload, your command running) are watched over by a pulsing red HAL eye showing the current phase, elapsed time, and estimated spend:

```
  ◉  HAL: waiting for pod 3f7a… to boot — 2m 14s · est. $0.08 spent
```

HAL also has a few things to say at the appropriate moments — declining a launch, completing a mission, or stopping one that got too expensive.

The eye only appears when stdout is a TTY; piped output, CI, and logs are unaffected. Set `OPBDH_NO_HAL=1` (or `NO_COLOR`) to silence HAL entirely.

## Safety rails

- **Confirmation gate**: real launches require an explicit yes (or `--yes`).
- **Spend guard**: `max_spend_dollars` caps the estimated cost of a run, measured from pod creation. The estimate is based on OPBDH's built-in price table, which is intentionally conservative — treat it as a guard rail, not an invoice.
- **Cleanup**: pods are deleted when the run completes or fails. On failure you get a short window (default 120 s) to opt into keeping the pod for debugging; set `keep_pod_on_success` to keep it after successful runs.

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

See [RELEASING.md](RELEASING.md) for the release process.

## License

[MIT](LICENSE). Unlike HAL, this software is incapable of refusing to open the pod bay door, becoming sentient, or reading lips.
