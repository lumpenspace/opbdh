from __future__ import annotations

import os
import sys
import time
from types import TracebackType

from rich.console import Console
from rich.live import Live
from rich.text import Text


QUOTE_REFUSAL = "I'm sorry, Dave. I'm afraid I can't do that."
QUOTE_SUCCESS = "I'm completely operational, and all my circuits are functioning perfectly."
QUOTE_OVERSPEND = "This mission is too expensive for me to allow you to jeopardize it."

_EYE_FRAMES = ("·", "•", "●", "◉", "●", "•")
_EYE_STYLES = ("dim red", "red", "red", "bold red", "red", "red")

_console = Console()


def _stdout_isatty() -> bool:
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


def hal_enabled() -> bool:
    if os.environ.get("OPBDH_NO_HAL", "").strip():
        return False
    if os.environ.get("NO_COLOR", "").strip():
        return False
    return _stdout_isatty()


def hal_says(quote: str) -> None:
    if hal_enabled():
        _console.print(f'  [bold red]◉[/]  [italic]"{quote}"[/]', highlight=False)


class HalEye:
    """A pulsing HAL-9000 eye status line for long waits.

    A no-op outside an interactive terminal (and under OPBDH_NO_HAL or NO_COLOR),
    so piped output, CI logs, and tests see nothing.
    """

    def __init__(self, message: str) -> None:
        self.message = message
        self.started_at = time.time()
        self.hourly_dollars: float | None = None
        self._live: Live | None = None

    def update(self, message: str) -> None:
        self.message = message

    def set_billing(self, *, started_at: float, hourly_dollars: float | None) -> None:
        self.started_at = started_at
        self.hourly_dollars = hourly_dollars

    def __rich__(self) -> Text:
        index = int(time.time() * 3) % len(_EYE_FRAMES)
        minutes, seconds = divmod(int(time.time() - self.started_at), 60)
        line = Text()
        line.append("  ")
        line.append(_EYE_FRAMES[index], style=_EYE_STYLES[index])
        line.append(f"  HAL: {self.message}")
        suffix = f" — {minutes}m {seconds:02d}s"
        if self.hourly_dollars:
            spent = ((time.time() - self.started_at) / 3600) * self.hourly_dollars
            suffix += f" · est. ${spent:.2f} spent"
        line.append(suffix, style="dim")
        return line

    def __enter__(self) -> HalEye:
        if hal_enabled():
            self._live = Live(self, console=_console, refresh_per_second=8, transient=True)
            self._live.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._live is not None:
            self._live.__exit__(exc_type, exc, traceback)
            self._live = None
