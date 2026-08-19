# Python API

The CLI is one front-end for opbdh; this is the other.

```python
import opbdh

result = opbdh.launch(
    "./train",
    model="Qwen/Qwen2.5-7B-Instruct",
    vram_gb=48,
    max_spend=5,
)
print(result.outputs_dir)   # runpod_results/<run_id>/results
```

Configuration layers exactly as it does for the CLI: keyword overrides beat a
local `opbdh.json`, which beats `~/.config/opbdh/config.json`. Any
`OpbdhConfig` field can be passed as a keyword, plus the CLI-style aliases
`model` (→ `model_id`), `max_spend` (→ `max_spend_dollars`), and
`min_ram_per_gpu` (→ `min_ram_per_gpu_gb`).

**These functions never prompt.** `launch()` does not ask for confirmation
before spending money, and a failed run always deletes its pod instead of
asking whether to keep it alive for debugging. Treat calling `launch()` as
having already said yes.

## `launch(code, **overrides)`

Plan and run in one call: rent a pod, upload the code, run it, stream
`logs/` and `results/` home, delete the pod.

```python
result = opbdh.launch("./train", model="Qwen/Qwen2.5-7B-Instruct", vram_gb=48)

result.run_id        # "20260819-013743"
result.pod_id
result.gpu_type_id   # the GPU actually allocated
result.results_dir   # runpod_results/<run_id>
result.logs_dir      # .../logs   — remote stdout/stderr
result.outputs_dir   # .../results — whatever the job wrote to $OPBDH_RESULTS_DIR
result.returncode
```

Useful keywords beyond the config fields:

| Keyword | Default | Meaning |
| --- | --- | --- |
| `dry_run` | `False` | Log the plan and return `None` without contacting the provider |
| `progress` | `False` | Draw the HAL status line (the CLI uses `True`) |
| `on_event` | `None` | Callback receiving each `RunEvent` |
| `config_file` | discovered | Explicit local config path |
| `run_id` | timestamp | Name the run yourself |

Errors:

- `MaxSpendReached` — the spend guard tripped mid-run. Results synced so far
  are still on disk.
- `RuntimeError` — the remote job exited non-zero, or the provider could not
  satisfy the request. `logs_dir` holds the remote stderr.
- `ValueError` — the code failed static verification, or no GPU in the
  catalogue matches the request. Nothing was rented.

## Watching progress

```python
def show(event):
    print(f"[{event.kind}] {event.message}")

opbdh.launch("./train", model="...", on_event=show)
```

`event.kind` is `"status"` for stage changes, `"output"` for remote stdout
captured after a failure, and `"error"` for remote stderr. To collect rather
than print:

```python
events, sink = opbdh.collect_events()
opbdh.launch("./train", model="...", on_event=sink)
print(opbdh.event_messages(events, kind="error"))
```

## Planning without spending

`plan()` does everything `launch()` does up to the point of renting: it
statically verifies the code, picks GPU candidates, and sizes any network
volume.

```python
plan = opbdh.plan("./train", model="Qwen/Qwen2.5-7B-Instruct", vram_gb=48)
summary = opbdh.summarize(plan)      # JSON-friendly dict

summary["gpu_candidates"]            # ordered GPU ids to try
summary["estimated_hourly_dollars"]
summary["command"]                   # what will run on the pod
```

`opbdh.configure(...)` returns just the resolved `OpbdhConfig` if you want to
inspect the layered settings without building a plan.

## Sizing helpers

```python
opbdh.verify("./train")                    # VerificationResult, .ok / .errors
opbdh.search_models("qwen instruct")       # list[str] of hub ids

size = opbdh.estimate_model_size("Qwen/Qwen2.5-7B-Instruct")
size.size_gb                               # weights, in GB (None if unknown)
opbdh.suggest_volume_gb(size)              # network volume big enough to cache it

est = opbdh.estimate_memory("Qwen/Qwen2.5-7B-Instruct", "lora", context_len=4096)
est.total_vram_gb
est.host_ram_gb

opbdh.gpu_options(vram_gb=48, max_dollars_per_hour=2.0)   # list[GpuOffer]
```

Goals for `estimate_memory` are in `opbdh.GOALS`: `inference`, `lora`,
`qlora`, `full`.

## What the remote script sees

Unchanged from the CLI. The code path is uploaded (a directory in full, or a
single file plus a sibling `requirements.txt`), extracted to `/opbdh-run/user`,
and run with:

- `OPBDH_MODEL_ID` — the model id
- `OPBDH_RESULTS_DIR` — write outputs here to have them synced home
- Hugging Face cache variables, plus `HF_TOKEN` when set locally
