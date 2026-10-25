"""Read and parse the NVIDIA model list exposed by the Pi CLI."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

PI_MODEL_LIST_COMMAND = ("pi", "--list-models", "nvidia")
PI_MODEL_LIST_TIMEOUT_SECONDS = 15
_PI_HEADER = ("provider", "model", "context", "max-out", "thinking", "images")


class PiModelListError(RuntimeError):
    """Raised when Pi cannot provide a valid non-empty NVIDIA model list."""


@dataclass(frozen=True, slots=True)
class PiModel:
    """One model parsed from ``pi --list-models nvidia``."""

    upstream_id: str
    display_name: str
    context_window: int
    max_output_tokens: int
    reasoning: bool
    input: tuple[str, ...]


def parse_pi_model_list(output: str) -> tuple[PiModel, ...]:
    """Parse the human-readable table emitted by ``pi --list-models nvidia``."""

    lines = output.splitlines()
    header_index = _find_header(lines)
    if header_index is None:
        raise PiModelListError("Pi model output did not contain the expected table header")

    models: list[PiModel] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(lines[header_index + 1 :], start=header_index + 2):
        if not line.strip():
            continue
        columns = line.split()
        if len(columns) != len(_PI_HEADER):
            raise PiModelListError(
                f"Pi model output line {line_number} has {len(columns)} columns; "
                f"expected {len(_PI_HEADER)}"
            )
        provider, model_id, context, max_output, thinking, images = columns
        if provider != "nvidia":
            continue
        if model_id in seen_ids:
            raise PiModelListError(f"Pi model output contains duplicate model {model_id}")
        seen_ids.add(model_id)
        modalities = ("text", "image") if images == "yes" else ("text",)
        models.append(
            PiModel(
                upstream_id=model_id,
                display_name=_display_name(model_id),
                context_window=_parse_token_count(context, line_number),
                max_output_tokens=_parse_token_count(max_output, line_number),
                reasoning=_parse_yes_no(thinking, "thinking", line_number),
                input=modalities,
            )
        )

    if not models:
        raise PiModelListError("Pi returned an empty NVIDIA model list")
    return tuple(models)


def run_pi_model_list(
    *,
    executable: str | None = None,
    timeout_seconds: float = PI_MODEL_LIST_TIMEOUT_SECONDS,
) -> tuple[PiModel, ...]:
    """Run Pi's NVIDIA model-list command and parse its output."""

    pi_executable = executable or shutil.which("pi")
    if pi_executable is None:
        raise PiModelListError(
            "Pi command not found; maintain ~/.config/claude-nim/models.json manually"
        )

    environment = os.environ.copy()
    environment["NO_COLOR"] = "1"
    environment["TERM"] = "dumb"
    environment.pop("NVIDIA_API_KEY", None)
    try:
        completed = subprocess.run(
            [pi_executable, *PI_MODEL_LIST_COMMAND[1:]],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=environment,
        )
    except subprocess.TimeoutExpired as exc:
        raise PiModelListError(
            f"Pi model-list command timed out after {timeout_seconds:g} seconds"
        ) from exc
    except OSError as exc:
        raise PiModelListError(f"could not run Pi model-list command: {exc}") from exc

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        if len(detail) > 300:
            detail = detail[:300] + "..."
        message = f"Pi model-list command exited with status {completed.returncode}"
        if detail:
            message += f": {detail}"
        raise PiModelListError(message)
    return parse_pi_model_list(completed.stdout)


def _find_header(lines: list[str]) -> int | None:
    for index, line in enumerate(lines):
        if tuple(line.split()) == _PI_HEADER:
            return index
    return None


def _parse_yes_no(value: str, field: str, line_number: int) -> bool:
    if value == "yes":
        return True
    if value == "no":
        return False
    raise PiModelListError(
        f"Pi model output line {line_number} has invalid {field} value {value!r}"
    )


def _parse_token_count(value: str, line_number: int) -> int:
    try:
        if value.endswith("M"):
            number = Decimal(value[:-1])
            scaled = number * 1_000_000
        elif value.endswith("K"):
            number = Decimal(value[:-1])
            scaled = number * 1_000
        else:
            number = Decimal(value)
            scaled = number
    except InvalidOperation as exc:
        raise PiModelListError(
            f"Pi model output line {line_number} has invalid token count {value!r}"
        ) from exc
    if scaled <= 0:
        raise PiModelListError(
            f"Pi model output line {line_number} has invalid token count {value!r}"
        )

    if value.endswith("K") and number != number.to_integral_value():
        # Pi rounds binary token counts to one decimal K. Round down to a 1024-token
        # boundary so Claude Code never receives an overstated context limit.
        return int(scaled / 1_024) * 1_024
    return int(scaled)


def _display_name(model_id: str) -> str:
    tail = model_id.rsplit("/", 1)[-1]
    return f"{tail} (NIM)"
