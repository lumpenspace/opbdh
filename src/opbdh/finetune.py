"""Small, provider-agnostic building blocks for supervised fine-tuning.

The CLI normalizes every supported source into OPBDH's deliberately small
``input``/``output``/``tags`` schema.  A generated, self-contained training
directory is then handed to the normal OPBDH run machinery, so fine-tuning
does not need a second implementation of pod creation, upload, result sync,
or cleanup.
"""

from __future__ import annotations

import json
import math
import re
import tomllib
from copy import deepcopy
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Iterable

from .config import OpbdhConfig
from .estimate import MemoryEstimate, estimate_for_model


DATA_FORMATS = ("auto", "opbdh", "openai", "anthropic")
MODEL_TYPES = ("base", "chat")
FINETUNE_METHODS = ("lora", "qlora", "full")
PROJECT_FILE = Path(".opbdh/finetune.json")
DEFAULT_DATA_FILE = Path(".opbdh/finetune/data.toml")
JOB_DIR = Path(".opbdh/finetune/jobs")
RESULTS_DIR = Path(".opbdh/finetune/results")
DEFAULT_RECIPE = "default"

METHOD_DEFAULTS: dict[str, dict[str, int | float | bool]] = {
    "lora": {
        "epochs": 3.0,
        "learning_rate": 2e-4,
        "max_length": 2048,
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "packing": False,
        "seed": 42,
    },
    "qlora": {
        "epochs": 3.0,
        "learning_rate": 2e-4,
        "max_length": 2048,
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "packing": False,
        "seed": 42,
    },
    "full": {
        "epochs": 3.0,
        "learning_rate": 2e-5,
        "max_length": 2048,
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "packing": False,
        "seed": 42,
    },
}

_RECIPE_FIELDS = (
    "model_id",
    "method",
    "selected_tags",
    "gpu_count",
    "vram_gb",
    "provider",
    "epochs",
    "learning_rate",
    "max_length",
    "per_device_batch_size",
    "gradient_accumulation_steps",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "lora_target_modules",
    "packing",
    "seed",
    "trust_remote_code",
    "chat_template",
    "max_dollars_per_hour",
    "max_spend_dollars",
)
_RECIPE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class FineTuneDataError(ValueError):
    """An input file or example does not match a supported SFT shape."""


@dataclass(frozen=True, slots=True)
class FineTuneExample:
    input: str | list[dict[str, str]]
    output: str
    tags: tuple[str, ...] = ()

    @property
    def is_chat(self) -> bool:
        return isinstance(self.input, list)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"input": self.input, "output": self.output}
        if self.tags:
            payload["tags"] = list(self.tags)
        return payload


@dataclass(slots=True)
class FineTuneProject:
    """Persistent SFT choices stored in ``.opbdh/finetune.json``."""

    version: int = 2
    model_id: str = ""
    model_type: str = ""
    method: str = "lora"
    data_file: str = str(DEFAULT_DATA_FILE)
    source_files: list[str] = field(default_factory=list)
    selected_tags: list[str] = field(default_factory=list)
    example_count: int = 0
    gpu_count: int = 1
    vram_gb: int | None = None
    provider: str = ""
    epochs: float = 3.0
    learning_rate: float = 2e-4
    max_length: int = 2048
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Comma-separated module names to attach LoRA to. Empty means peft's
    # "all-linear" shorthand, which is right for mainstream architectures but
    # cannot resolve one whose output embedding is a bare nn.Parameter rather
    # than a module (peft looks the module up by identity and raises).
    lora_target_modules: str = ""
    packing: bool = False
    seed: int = 42
    trust_remote_code: bool = False
    chat_template: str = ""
    max_dollars_per_hour: float | None = None
    max_spend_dollars: float = 5.0
    active_recipe: str = DEFAULT_RECIPE
    recipes: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)


@dataclass(frozen=True, slots=True)
class FineTuneResources:
    estimate: MemoryEstimate
    vram_per_gpu_gb: int
    host_ram_per_gpu_gb: int
    disk_gb: int


@dataclass(frozen=True, slots=True)
class FineTuneJob:
    directory: Path
    command: str
    example_count: int
    selected_tags: tuple[str, ...]


def normalize_tags(value: Any) -> tuple[str, ...]:
    """Normalize a string or string list into stable, unique tag names."""

    if value is None:
        return ()
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, (list, tuple, set)):
        raise FineTuneDataError("tags must be a string or a list of strings")
    normalized: list[str] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, str):
            raise FineTuneDataError("tags must contain only strings")
        tag = item.strip()
        if tag and tag not in seen:
            normalized.append(tag)
            seen.add(tag)
    return tuple(normalized)


def _merge_tags(record: dict[str, Any], extra_tags: Iterable[str]) -> tuple[str, ...]:
    record_tags = record.get("tags", record.get("tag"))
    return normalize_tags([*normalize_tags(record_tags), *normalize_tags(list(extra_tags))])


def _text_content(value: Any, *, field_name: str) -> str:
    """Read provider text content, including modern content-block arrays."""

    if isinstance(value, str):
        text = value
    elif isinstance(value, list):
        parts: list[str] = []
        for block in value:
            if isinstance(block, str):
                parts.append(block)
                continue
            if not isinstance(block, dict):
                raise FineTuneDataError(f"{field_name} has a non-text content block")
            candidate = block.get("text", block.get("content"))
            if candidate is not None:
                parts.append(_text_content(candidate, field_name=field_name))
        text = "".join(parts)
    elif isinstance(value, dict):
        candidate = value.get("text", value.get("content"))
        if candidate is None:
            raise FineTuneDataError(f"{field_name} must contain text")
        text = _text_content(candidate, field_name=field_name)
    else:
        raise FineTuneDataError(f"{field_name} must be text or text content blocks")
    if not text.strip():
        raise FineTuneDataError(f"{field_name} cannot be empty")
    return text


_ROLE_ALIASES = {
    "ai": "assistant",
    "bot": "assistant",
    "gpt": "assistant",
    "human": "user",
}


def _message(value: Any, *, index: int) -> dict[str, str]:
    if not isinstance(value, dict):
        raise FineTuneDataError(f"message {index} must be an object")
    role_value = value.get("role", value.get("from", value.get("author")))
    if not isinstance(role_value, str) or not role_value.strip():
        raise FineTuneDataError(f"message {index} needs a role")
    role = _ROLE_ALIASES.get(role_value.strip().lower(), role_value.strip().lower())
    if role not in {"system", "developer", "user", "assistant", "tool"}:
        raise FineTuneDataError(f"message {index} has unsupported role {role_value!r}")
    content = value.get("content", value.get("value", value.get("text")))
    return {"role": role, "content": _text_content(content, field_name=f"message {index} content")}


def _messages(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise FineTuneDataError("chat input must be a non-empty list of messages")
    return [_message(message, index=index) for index, message in enumerate(value, start=1)]


def _canonical_example(record: dict[str, Any], extra_tags: Iterable[str]) -> FineTuneExample:
    if "input" not in record or "output" not in record:
        raise FineTuneDataError("an OPBDH example needs both input and output")
    input_value = record["input"]
    if isinstance(input_value, str):
        if not input_value.strip():
            raise FineTuneDataError("input cannot be empty")
        normalized_input: str | list[dict[str, str]] = input_value
    elif isinstance(input_value, list):
        normalized_input = _messages(input_value)
    elif isinstance(input_value, dict) and isinstance(input_value.get("messages"), list):
        normalized_input = _messages(input_value["messages"])
    else:
        raise FineTuneDataError("input must be text or a list of chat messages")
    output = _text_content(record["output"], field_name="output")
    return FineTuneExample(normalized_input, output, _merge_tags(record, extra_tags))


_ANTHROPIC_TURN = re.compile(r"(?:^|\n\n)(Human|Assistant):\s*", re.IGNORECASE)


def _legacy_anthropic_input(prompt: str) -> list[dict[str, str]] | None:
    matches = list(_ANTHROPIC_TURN.finditer(prompt))
    if not matches:
        return None
    messages: list[dict[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(prompt)
        content = prompt[match.end() : end].strip()
        if not content:
            continue
        role = "user" if match.group(1).lower() == "human" else "assistant"
        messages.append({"role": role, "content": content})
    return messages or None


def _provider_example(record: dict[str, Any], source_format: str, extra_tags: Iterable[str]) -> FineTuneExample:
    messages_value = record.get("messages")
    if messages_value is None and isinstance(record.get("params"), dict):
        messages_value = record["params"].get("messages")

    if messages_value is not None:
        messages = _messages(messages_value)
        system = record.get("system")
        if system is None and isinstance(record.get("params"), dict):
            system = record["params"].get("system")
        if system is not None:
            messages.insert(0, {"role": "system", "content": _text_content(system, field_name="system")})

        explicit_output = record.get("output", record.get("completion"))
        if explicit_output is not None:
            output = _text_content(explicit_output, field_name="output")
            input_messages = messages
        else:
            if not messages or messages[-1]["role"] != "assistant":
                raise FineTuneDataError("provider messages need a final assistant message or an output field")
            output = messages[-1]["content"]
            input_messages = messages[:-1]
        if not input_messages:
            raise FineTuneDataError("provider example has no input messages before its output")
        return FineTuneExample(input_messages, output, _merge_tags(record, extra_tags))

    if "prompt" in record and "completion" in record:
        prompt = _text_content(record["prompt"], field_name="prompt")
        completion = _text_content(record["completion"], field_name="completion")
        anthropic_messages = _legacy_anthropic_input(prompt) if source_format in {"auto", "anthropic"} else None
        input_value: str | list[dict[str, str]] = anthropic_messages or prompt
        return FineTuneExample(input_value, completion, _merge_tags(record, extra_tags))

    raise FineTuneDataError("record is not a recognized OpenAI or Anthropic training example")


def example_from_record(
    record: Any,
    *,
    source_format: str = "auto",
    extra_tags: Iterable[str] = (),
) -> FineTuneExample:
    source_format = source_format.strip().lower()
    if source_format not in DATA_FORMATS:
        raise FineTuneDataError(f"unsupported format {source_format!r}; expected one of: {', '.join(DATA_FORMATS)}")
    if not isinstance(record, dict):
        raise FineTuneDataError("each example must be a JSON object")
    if source_format == "opbdh" or (source_format == "auto" and "input" in record and "output" in record):
        return _canonical_example(record, extra_tags)
    if source_format in {"openai", "anthropic", "auto"}:
        return _provider_example(record, source_format, extra_tags)
    raise FineTuneDataError("unsupported example")


def _payload_records(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for wrapper in ("examples", "data"):
            wrapped = payload.get(wrapper)
            if isinstance(wrapped, list) and not ({"input", "output", "messages"} & payload.keys()):
                return wrapped
        return [payload]
    raise FineTuneDataError("the JSON root must be an example object or a list of examples")


def _read_records(path: Path) -> list[Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FineTuneDataError(f"cannot read {path}: {exc}") from exc
    if not text.strip():
        raise FineTuneDataError(f"{path} is empty")
    if path.suffix.lower() == ".toml":
        try:
            return _payload_records(tomllib.loads(text))
        except (tomllib.TOMLDecodeError, FineTuneDataError) as exc:
            raise FineTuneDataError(f"{path}: invalid TOML: {exc}") from exc
    try:
        return _payload_records(json.loads(text))
    except json.JSONDecodeError as whole_error:
        records: list[Any] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.extend(_payload_records(json.loads(line)))
            except (json.JSONDecodeError, FineTuneDataError) as exc:
                raise FineTuneDataError(f"{path}:{line_number}: invalid JSONL record: {exc}") from exc
        if not records:
            raise FineTuneDataError(f"{path}: invalid JSON: {whole_error}") from whole_error
        return records


def expand_data_paths(paths: Iterable[Path]) -> list[Path]:
    expanded: list[Path] = []
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        if not path.exists():
            raise FineTuneDataError(f"data path does not exist: {path}")
        if path.is_dir():
            matches = sorted(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file() and candidate.suffix.lower() in {".json", ".jsonl", ".toml"}
            )
            if not matches:
                raise FineTuneDataError(f"data directory has no .toml, .json, or .jsonl files: {path}")
            expanded.extend(matches)
        else:
            expanded.append(path)
    if not expanded:
        raise FineTuneDataError("at least one data file is required")
    return expanded


def read_examples(
    paths: Iterable[Path],
    *,
    source_format: str = "auto",
    extra_tags: Iterable[str] = (),
) -> list[FineTuneExample]:
    examples: list[FineTuneExample] = []
    for path in expand_data_paths(paths):
        for index, record in enumerate(_read_records(path), start=1):
            try:
                examples.append(example_from_record(record, source_format=source_format, extra_tags=extra_tags))
            except FineTuneDataError as exc:
                raise FineTuneDataError(f"{path} example {index}: {exc}") from exc
    return examples


def _example_key(example: FineTuneExample) -> str:
    return json.dumps(example.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def deduplicate_examples(examples: Iterable[FineTuneExample]) -> list[FineTuneExample]:
    unique: list[FineTuneExample] = []
    seen: set[str] = set()
    for example in examples:
        key = _example_key(example)
        if key not in seen:
            unique.append(example)
            seen.add(key)
    return unique


def require_finetune_extra() -> None:
    try:
        import tomli_w  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            'Fine-tuning support is optional. Install it with: pip install "opbdh[ft]"'
        ) from exc


def _examples_as_toml(examples: list[FineTuneExample]) -> str:
    require_finetune_extra()
    import tomli_w

    payload: dict[str, Any]
    if len(examples) == 1:
        payload = examples[0].to_dict()
    else:
        payload = {"examples": [example.to_dict() for example in examples]}
    # Prompts and completions are prose; preserving CRLF bytes is less useful
    # here than emitting truly hand-editable multiline strings.
    return tomli_w.dumps(payload, multiline_strings=True)


def write_examples(path: Path, examples: Iterable[FineTuneExample], *, overwrite: bool = False) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}; pass --force or choose another output")
    normalized = deduplicate_examples(examples)
    if not normalized:
        raise FineTuneDataError("cannot write an empty dataset")
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".toml":
        content = _examples_as_toml(normalized)
    elif suffix == ".json":
        payload: Any = normalized[0].to_dict() if len(normalized) == 1 else [item.to_dict() for item in normalized]
        content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    else:
        content = "".join(json.dumps(example.to_dict(), ensure_ascii=False) + "\n" for example in normalized)
    path.write_text(content, encoding="utf-8")
    return path


def available_tags(examples: Iterable[FineTuneExample]) -> list[str]:
    return sorted({tag for example in examples for tag in example.tags})


def select_examples(examples: Iterable[FineTuneExample], tags: Iterable[str]) -> list[FineTuneExample]:
    selected_tags = set(normalize_tags(list(tags)))
    values = list(examples)
    if not selected_tags:
        return values
    return [example for example in values if selected_tags.intersection(example.tags)]


def infer_model_type(examples: Iterable[FineTuneExample]) -> str:
    kinds = {"chat" if example.is_chat else "base" for example in examples}
    if not kinds:
        raise FineTuneDataError("cannot infer a model type from an empty dataset")
    if len(kinds) != 1:
        raise FineTuneDataError("the dataset mixes text inputs and chat message inputs")
    return kinds.pop()


def validate_examples_for_model(examples: Iterable[FineTuneExample], model_type: str) -> None:
    normalized_type = model_type.strip().lower()
    if normalized_type not in MODEL_TYPES:
        raise FineTuneDataError(f"model type must be one of: {', '.join(MODEL_TYPES)}")
    wrong = [index for index, example in enumerate(examples, start=1) if example.is_chat != (normalized_type == "chat")]
    if wrong:
        expected = "message-list" if normalized_type == "chat" else "text"
        preview = ", ".join(str(index) for index in wrong[:5])
        raise FineTuneDataError(f"{normalized_type} models require {expected} inputs; mismatched examples: {preview}")


def _root_path(start: Path | None = None) -> Path:
    path = (start or Path.cwd()).expanduser().resolve()
    return path.parent if path.is_file() else path


def find_finetune_root(start: Path | None = None) -> Path | None:
    current = _root_path(start)
    for candidate in (current, *current.parents):
        if (candidate / PROJECT_FILE).is_file():
            return candidate
    return None


def validate_recipe_name(name: str) -> str:
    normalized = name.strip()
    if not _RECIPE_NAME.fullmatch(normalized):
        raise FineTuneDataError(
            "recipe names must be 1-64 characters using letters, numbers, dots, underscores, or hyphens"
        )
    return normalized


def method_defaults(method: str) -> dict[str, int | float | bool]:
    normalized = method.strip().lower()
    if normalized not in FINETUNE_METHODS:
        raise FineTuneDataError(f"method must be one of: {', '.join(FINETUNE_METHODS)}")
    return dict(METHOD_DEFAULTS[normalized])


def set_finetune_method(project: FineTuneProject, method: str, *, reset_defaults: bool = False) -> None:
    """Select a technique while preserving deliberately customized values.

    A fresh recipe gets every method-aware default. When an existing recipe is
    changed, values that still match the old defaults follow the new method;
    explicit custom values remain untouched.
    """

    normalized = method.strip().lower()
    defaults = method_defaults(normalized)
    old_defaults = method_defaults(project.method)
    old_method = project.method
    project.method = normalized
    for name, value in defaults.items():
        if reset_defaults or (old_method != normalized and getattr(project, name) == old_defaults[name]):
            setattr(project, name, value)


def _recipe_payload(project: FineTuneProject) -> dict[str, Any]:
    return {name: deepcopy(getattr(project, name)) for name in _RECIPE_FIELDS}


def _normalized_recipe_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise FineTuneDataError("each fine-tuning recipe must be an object")
    defaults = FineTuneProject()
    return {
        name: deepcopy(payload[name] if name in payload else getattr(defaults, name))
        for name in _RECIPE_FIELDS
    }


def _apply_recipe_payload(project: FineTuneProject, payload: dict[str, Any]) -> None:
    normalized = _normalized_recipe_payload(payload)
    for name, value in normalized.items():
        setattr(project, name, deepcopy(value))


def sync_active_recipe(project: FineTuneProject) -> None:
    project.active_recipe = validate_recipe_name(project.active_recipe)
    project.recipes[project.active_recipe] = _recipe_payload(project)


def recipe_names(project: FineTuneProject) -> list[str]:
    if not project.recipes:
        return [validate_recipe_name(project.active_recipe)]
    return sorted(project.recipes)


def activate_finetune_recipe(project: FineTuneProject, name: str) -> None:
    normalized = validate_recipe_name(name)
    if not project.recipes:
        sync_active_recipe(project)
    if normalized not in project.recipes:
        raise FineTuneDataError(f"unknown fine-tuning recipe {normalized!r}")
    sync_active_recipe(project)
    project.active_recipe = normalized
    _apply_recipe_payload(project, project.recipes[normalized])


def create_finetune_recipe(
    project: FineTuneProject,
    name: str,
    *,
    method: str = "lora",
    clone_current: bool = False,
) -> None:
    normalized = validate_recipe_name(name)
    sync_active_recipe(project)
    if normalized in project.recipes:
        raise FineTuneDataError(f"fine-tuning recipe {normalized!r} already exists")

    if clone_current:
        payload = _recipe_payload(project)
    else:
        fresh = FineTuneProject(
            model_id=project.model_id,
            gpu_count=project.gpu_count,
            vram_gb=project.vram_gb,
            provider=project.provider,
            trust_remote_code=project.trust_remote_code,
            chat_template=project.chat_template,
            max_dollars_per_hour=project.max_dollars_per_hour,
            max_spend_dollars=project.max_spend_dollars,
        )
        set_finetune_method(fresh, method, reset_defaults=True)
        payload = _recipe_payload(fresh)

    project.recipes[normalized] = payload
    project.active_recipe = normalized
    _apply_recipe_payload(project, payload)


def rename_finetune_recipe(project: FineTuneProject, name: str) -> None:
    normalized = validate_recipe_name(name)
    sync_active_recipe(project)
    if normalized in project.recipes and normalized != project.active_recipe:
        raise FineTuneDataError(f"fine-tuning recipe {normalized!r} already exists")
    old_name = project.active_recipe
    project.recipes = {
        (normalized if recipe_name == old_name else recipe_name): payload
        for recipe_name, payload in project.recipes.items()
    }
    project.active_recipe = normalized


def delete_finetune_recipe(project: FineTuneProject, name: str) -> None:
    normalized = validate_recipe_name(name)
    sync_active_recipe(project)
    if normalized not in project.recipes:
        raise FineTuneDataError(f"unknown fine-tuning recipe {normalized!r}")
    if len(project.recipes) == 1:
        raise FineTuneDataError("a fine-tune project must keep at least one recipe")
    del project.recipes[normalized]
    if project.active_recipe == normalized:
        project.active_recipe = sorted(project.recipes)[0]
        _apply_recipe_payload(project, project.recipes[project.active_recipe])


def load_finetune_project(root: Path) -> FineTuneProject | None:
    path = root.expanduser().resolve() / PROJECT_FILE
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FineTuneDataError(f"{path} must contain a JSON object")

    stored_recipes = payload.get("recipes")
    if isinstance(stored_recipes, dict) and stored_recipes:
        project = FineTuneProject(
            version=2,
            model_type=payload.get("model_type", ""),
            data_file=payload.get("data_file", str(DEFAULT_DATA_FILE)),
            source_files=list(payload.get("source_files", [])),
            example_count=payload.get("example_count", 0),
            active_recipe=payload.get("active_recipe", DEFAULT_RECIPE),
        )
        project.recipes = {
            validate_recipe_name(str(name)): _normalized_recipe_payload(recipe)
            for name, recipe in stored_recipes.items()
        }
        if project.active_recipe not in project.recipes:
            raise FineTuneDataError(f"{path}: active recipe {project.active_recipe!r} does not exist")
        _apply_recipe_payload(project, project.recipes[project.active_recipe])
    else:
        # Version 1 stored one training configuration directly on the project.
        # Promote it to a named recipe without changing the user's settings.
        known = {item.name for item in fields(FineTuneProject)} - {"recipes"}
        project = FineTuneProject(**{key: value for key, value in payload.items() if key in known})
        project.version = 2
        sync_active_recipe(project)
    validate_project(project)
    return project


def _validate_training_values(project: FineTuneProject) -> None:
    if project.method not in FINETUNE_METHODS:
        raise FineTuneDataError(f"method must be one of: {', '.join(FINETUNE_METHODS)}")
    if project.gpu_count < 1:
        raise FineTuneDataError("gpu_count must be at least 1")
    if project.epochs <= 0 or project.learning_rate <= 0:
        raise FineTuneDataError("epochs and learning_rate must be positive")
    if project.max_length < 1 or project.per_device_batch_size < 1 or project.gradient_accumulation_steps < 1:
        raise FineTuneDataError("max_length and batch settings must be positive")
    if project.lora_r < 1 or project.lora_alpha < 1 or not 0 <= project.lora_dropout < 1:
        raise FineTuneDataError("LoRA rank/alpha must be positive and dropout must be between 0 and 1")


def validate_project(project: FineTuneProject) -> None:
    if project.model_type and project.model_type not in MODEL_TYPES:
        raise FineTuneDataError(f"model_type must be one of: {', '.join(MODEL_TYPES)}")
    validate_recipe_name(project.active_recipe)
    _validate_training_values(project)
    if project.recipes and project.active_recipe not in project.recipes:
        raise FineTuneDataError(f"active recipe {project.active_recipe!r} does not exist")
    for name, payload in project.recipes.items():
        validate_recipe_name(name)
        candidate = FineTuneProject(model_type=project.model_type)
        _apply_recipe_payload(candidate, payload)
        _validate_training_values(candidate)


def save_finetune_project(root: Path, project: FineTuneProject) -> Path:
    project.version = 2
    sync_active_recipe(project)
    validate_project(project)
    path = root.expanduser().resolve() / PROJECT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": project.version,
        "model_type": project.model_type,
        "data_file": project.data_file,
        "source_files": list(project.source_files),
        "example_count": project.example_count,
        "active_recipe": project.active_recipe,
        "recipes": project.recipes,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def project_data_path(root: Path, project: FineTuneProject) -> Path:
    path = Path(project.data_file).expanduser()
    return path.resolve() if path.is_absolute() else (root.expanduser().resolve() / path).resolve()


def read_project_examples(root: Path, project: FineTuneProject) -> list[FineTuneExample]:
    path = project_data_path(root, project)
    if not path.exists():
        return []
    return read_examples([path], source_format="opbdh")


def add_project_examples(
    root: Path,
    project: FineTuneProject,
    examples: Iterable[FineTuneExample],
    *,
    source_files: Iterable[Path] = (),
) -> list[FineTuneExample]:
    existing = read_project_examples(root, project)
    combined = deduplicate_examples([*existing, *examples])
    if project.model_type:
        validate_examples_for_model(combined, project.model_type)
    write_examples(project_data_path(root, project), combined, overwrite=True)
    known_sources = list(project.source_files)
    for source in source_files:
        label = str(source.expanduser().resolve())
        if label not in known_sources:
            known_sources.append(label)
    project.source_files = known_sources
    project.example_count = len(combined)
    save_finetune_project(root, project)
    return combined


def replace_project_examples(
    root: Path,
    project: FineTuneProject,
    examples: Iterable[FineTuneExample],
) -> list[FineTuneExample]:
    """Replace the managed dataset after edits such as retagging."""

    normalized = deduplicate_examples(examples)
    if project.model_type:
        validate_examples_for_model(normalized, project.model_type)
    write_examples(project_data_path(root, project), normalized, overwrite=True)
    project.example_count = len(normalized)
    existing_tags = set(available_tags(normalized))
    project.selected_tags = [tag for tag in project.selected_tags if tag in existing_tags]
    sync_active_recipe(project)
    for recipe in project.recipes.values():
        recipe["selected_tags"] = [
            tag for tag in normalize_tags(recipe.get("selected_tags")) if tag in existing_tags
        ]
    _apply_recipe_payload(project, project.recipes[project.active_recipe])
    save_finetune_project(root, project)
    return normalized


def estimate_finetune_resources(project: FineTuneProject) -> FineTuneResources:
    """Estimate a data-parallel SFT replica and the host/disk around it.

    Standard Accelerate multi-GPU training keeps one model replica per GPU, so
    GPU count increases throughput but does not lower the per-GPU VRAM floor.
    This deliberately asks the estimator for a one-GPU fit even for a larger
    pod, avoiding plans that only work with unconfigured model sharding.
    """

    estimate = estimate_for_model(
        project.model_id,
        project.method,
        context_len=project.max_length,
        batch_size=project.per_device_batch_size,
    )
    per_gpu_vram = project.vram_gb or estimate.min_vram_gb(1)
    host_per_gpu = max(16, int(math.ceil(estimate.host_ram_gb / max(1, project.gpu_count))))
    disk_gb = max(40, int(estimate.disk_gb) + 20)
    return FineTuneResources(estimate, per_gpu_vram, host_per_gpu, disk_gb)


def _training_rows(examples: Iterable[FineTuneExample], model_type: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example in examples:
        if model_type == "chat":
            rows.append(
                {
                    "prompt": example.input,
                    "completion": [{"role": "assistant", "content": example.output}],
                }
            )
        else:
            rows.append({"prompt": example.input, "completion": example.output})
    return rows


def prepare_finetune_job(root: Path, project: FineTuneProject) -> FineTuneJob:
    validate_project(project)
    if not project.model_id.strip():
        raise FineTuneDataError("a Hugging Face model id is required")
    if not project.model_type:
        raise FineTuneDataError("model_type is required")
    examples = read_project_examples(root, project)
    if not examples:
        raise FineTuneDataError("the fine-tune project has no examples")
    validate_examples_for_model(examples, project.model_type)
    selected = select_examples(examples, project.selected_tags)
    if not selected:
        raise FineTuneDataError("no examples match the selected tags")

    directory = root.expanduser().resolve() / JOB_DIR / validate_recipe_name(project.active_recipe)
    directory.mkdir(parents=True, exist_ok=True)
    runner_source = Path(__file__).with_name("_ft_runner.py").read_text(encoding="utf-8")
    (directory / "run.py").write_text(runner_source, encoding="utf-8")
    (directory / "dataset.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in _training_rows(selected, project.model_type)),
        encoding="utf-8",
    )
    runner_config = {
        "recipe": project.active_recipe,
        "model_id": project.model_id,
        "model_type": project.model_type,
        "method": project.method,
        "epochs": project.epochs,
        "learning_rate": project.learning_rate,
        "max_length": project.max_length,
        "per_device_batch_size": project.per_device_batch_size,
        "gradient_accumulation_steps": project.gradient_accumulation_steps,
        "lora_r": project.lora_r,
        "lora_alpha": project.lora_alpha,
        "lora_dropout": project.lora_dropout,
        "lora_target_modules": project.lora_target_modules,
        "packing": project.packing,
        "seed": project.seed,
        "trust_remote_code": project.trust_remote_code,
        "chat_template": project.chat_template,
        "example_count": len(selected),
        "selected_tags": list(project.selected_tags),
    }
    (directory / "config.json").write_text(
        json.dumps(runner_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    requirements = [
        "accelerate>=1.14,<2",
        "datasets>=4.8,<5",
        "peft>=0.20,<0.21",
        "transformers>=5,<6",
        "trl>=0.29.1,<0.30",
    ]
    if project.method == "qlora":
        requirements.append("bitsandbytes>=0.48,<1")
    (directory / "requirements.txt").write_text("\n".join(requirements) + "\n", encoding="utf-8")

    remote_runner = "/opbdh-run/user/run.py"
    if project.gpu_count > 1:
        command = (
            "accelerate launch --multi_gpu "
            f"--num_processes {project.gpu_count} {remote_runner}"
        )
    else:
        command = f"python {remote_runner}"
    return FineTuneJob(directory, command, len(selected), tuple(project.selected_tags))


def build_finetune_run_config(
    base: OpbdhConfig,
    *,
    root: Path,
    project: FineTuneProject,
    job: FineTuneJob,
    resources: FineTuneResources,
) -> OpbdhConfig:
    config = replace(base)
    config.model_id = project.model_id
    config.code = str(job.directory)
    config.command = job.command
    config.gpu_count = project.gpu_count
    config.vram_gb = resources.vram_per_gpu_gb
    config.container_disk_gb = max(config.container_disk_gb, resources.disk_gb)
    config.pod_volume_gb = max(config.pod_volume_gb, resources.disk_gb)
    config.min_ram_per_gpu_gb = max(config.min_ram_per_gpu_gb, resources.host_ram_per_gpu_gb)
    config.results_dir = str(
        root.expanduser().resolve() / RESULTS_DIR / validate_recipe_name(project.active_recipe)
    )
    config.pre_download_model = True
    if project.provider:
        config.provider = project.provider
    config.max_dollars_per_hour = project.max_dollars_per_hour
    config.max_spend_dollars = project.max_spend_dollars
    if config.auto_network_volume and config.network_volume_size_gb is None:
        config.network_volume_size_gb = resources.disk_gb
    return config
