# Supervised fine-tuning

Install the optional local authoring dependency and open the console:

```bash
pip install "opbdh[ft]"
opbdh ft
```

The console can import files or create examples interactively. It then lets you manage tags, choose the tag groups for a run, select or create a training recipe, inspect the automatically sized pod, and pass the job to OPBDH's normal confirmation and spend guard.

## Native TOML format

One base-model example per file is deliberately small:

```toml
input = "Translate hello to Italian"
output = "Ciao"
tags = ["translation", "short"]
```

Use multiline TOML strings for longer prose. For a chat model, `input` is a list of messages; `output` is the assistant response to train:

```toml
output = "Ciao! Come posso aiutarti?"
tags = ["italian", "greeting"]

[[input]]
role = "system"
content = "Reply in Italian."

[[input]]
role = "user"
content = "Hello!"
```

A single TOML file can contain multiple examples:

```toml
[[examples]]
input = "2 + 2"
output = "4"
tags = ["math"]

[[examples]]
input = "3 + 5"
output = "8"
tags = ["math"]
```

For a chat example inside that form, use `[[examples.input]]` for each message. OPBDH also accepts one JSON object, a JSON array, an `{ "examples": [...] }` wrapper, or JSONL with one object per line. A directory input recursively loads `.toml`, `.json`, and `.jsonl` files, so one-file-per-example and many-examples-per-file layouts can be mixed.

## Imports

Convert provider data to editable TOML without launching anything:

```bash
opbdh ft:import training.jsonl --format openai
opbdh ft:import anthropic-a.jsonl anthropic-b.jsonl --format anthropic -o examples.toml
opbdh ft:import training.jsonl --tag imported --tag support
```

`--format auto` is the default. OpenAI `messages` rows and `prompt`/`completion` rows are supported. Anthropic message content blocks, a separate `system` field, and legacy `Human:`/`Assistant:` prompt/completion rows are normalized. The output extension controls whether `ft:import` writes TOML, JSON, or JSONL; its default is `<source>.opbdh.toml`.

Files passed directly to `opbdh ft --data ...` go through the same importer and are copied into the managed dataset, so later runs do not depend on the original files.

## Tags

Every example may have zero or more tags. After files are loaded, the interactive console offers:

- bulk add or remove, optionally limited to an existing tag group;
- rename a tag everywhere;
- edit the tags on one example;
- clear all tags;
- choose which groups participate in the next run.

Choosing several groups uses “match any” semantics. With no group selected, every example—including untagged examples—is trained. Non-interactively, repeat `--tag` to select groups:

```bash
opbdh ft --tag support --tag concise --dry-run
```

## Training recipes

The editable dataset is shared, while training choices are saved as independent named recipes. A recipe contains its model, fine-tuning technique, selected tag groups, GPU count, resource overrides, hyperparameters, provider, and spend limits. The interactive console can switch, create, clone, rename, reset, and delete recipes without duplicating or modifying the dataset.

New recipes start with sensible technique-specific defaults: LoRA and QLoRA use a `2e-4` learning rate, rank 16, alpha 32, and dropout 0.05, while a Full recipe uses `2e-5`; all default to three epochs, a 2,048-token maximum length, batch size 1 per GPU, eight gradient-accumulation steps, packing off, and seed 42. Creating a recipe keeps the current model and infrastructure choices but starts from all examples. Cloning is the way to preserve every current parameter and tag selection for a controlled variation.

Recipes also work non-interactively. A missing name is created; an existing name is restored before flag overrides are applied:

```bash
opbdh ft --recipe qlora-default --method qlora --dry-run
opbdh ft --recipe full-2gpu --method full --gpu-count 2 --learning-rate 1e-5
```

Each recipe gets a separate generated job directory and results directory, so experimenting with several techniques on one dataset does not overwrite another recipe's artifacts.

## Launching without the console

```bash
opbdh ft \
  --data ./examples \
  --model Qwen/Qwen3-8B \
  --model-type chat \
  --recipe lora-4gpu \
  --method lora \
  --gpu-count 4 \
  --max-spend 10
```

Useful overrides include `--recipe`, `--epochs`, `--learning-rate`, `--max-length`, `--batch-size`, `--gradient-accumulation`, `--lora-r`, `--lora-alpha`, `--lora-dropout`, `--packing`, `--seed`, `--vram-gb`, `--provider`, `--max-dollars-per-hour`, `--dry-run`, and `--yes`. The interactive advanced editor exposes the same core parameters.

LoRA is the default. QLoRA loads the base model in 4-bit NF4 for a lower per-GPU memory floor. Full fine-tuning is available for models that fit. The generated program uses completion-only loss for both base and conversational prompt/completion datasets; chat models use their tokenizer chat template to add model-specific control tokens.

For multiple GPUs, OPBDH requests one multi-GPU pod and launches TRL through Accelerate. The default strategy is data parallel: it improves throughput and multiplies the effective batch size, but every GPU must fit one model replica. Automatic sizing therefore does not divide the model's memory requirement by the GPU count.

## Persistent project files

OPBDH keeps these under the directory where the fine-tune project was created:

```text
.opbdh/
├── finetune.json          # named recipes and shared-dataset metadata
└── finetune/
    ├── data.toml          # editable normalized examples
    ├── jobs/<recipe>/     # generated runner, selected JSONL rows, and requirements
    └── results/<recipe>/<run-id>/
                            # synced adapter/model, metrics, and logs
```

Run `opbdh ft` from that directory or a child directory to reuse the project. The generated pod installs Accelerate, Datasets, PEFT, TRL, and BitsAndBytes when QLoRA is selected; those GPU training packages are not installed on the local machine.
