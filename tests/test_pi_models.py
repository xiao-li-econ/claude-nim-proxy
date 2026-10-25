from __future__ import annotations

import subprocess

import pytest

from claude_nim_proxy.pi_models import (
    PiModelListError,
    parse_pi_model_list,
    run_pi_model_list,
)

PI_OUTPUT = (
    "provider  model                                          context  max-out  thinking  images\n"
    "nvidia    deepseek-ai/deepseek-v4.1-flash                1M       384K     yes       yes\n"
    "nvidia    google/gemma-3-12b-it                          131.1K   16.4K    no        yes\n"
    "nvidia    mistralai/mistral-7b-instruct-v0.3             65.5K    65.5K    no        no\n"
)


def test_parse_pi_model_list_reads_ids_and_capabilities() -> None:
    models = parse_pi_model_list(PI_OUTPUT)

    assert [model.upstream_id for model in models] == [
        "deepseek-ai/deepseek-v4.1-flash",
        "google/gemma-3-12b-it",
        "mistralai/mistral-7b-instruct-v0.3",
    ]
    assert models[0].context_window == 1_000_000
    assert models[0].max_output_tokens == 384_000
    assert models[0].reasoning is True
    assert models[0].input == ("text", "image")
    assert models[1].context_window == 131_072
    assert models[1].max_output_tokens == 16_384
    assert models[2].input == ("text",)


def test_parse_pi_model_list_rejects_empty_output() -> None:
    with pytest.raises(PiModelListError, match="empty NVIDIA model list"):
        parse_pi_model_list(
            "provider  model  context  max-out  thinking  images\n"
        )


def test_parse_pi_model_list_rejects_missing_header() -> None:
    with pytest.raises(PiModelListError, match="expected table header"):
        parse_pi_model_list("nvidia    model\n")


def test_parse_pi_model_list_rejects_malformed_rows() -> None:
    with pytest.raises(PiModelListError, match="expected 6"):
        parse_pi_model_list(
            "provider  model  context  max-out  thinking  images\n"
            "nvidia    model  1M\n"
        )


def test_run_pi_model_list_uses_fixed_arguments_and_scrubs_nvidia_key(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(args, 0, stdout=PI_OUTPUT, stderr="")

    monkeypatch.setattr("claude_nim_proxy.pi_models.shutil.which", lambda _: "/usr/bin/pi")
    monkeypatch.setattr("claude_nim_proxy.pi_models.subprocess.run", fake_run)
    monkeypatch.setenv("NVIDIA_API_KEY", "must-not-be-passed")

    models = run_pi_model_list()

    assert len(models) == 3
    assert captured["args"] == ["/usr/bin/pi", "--list-models", "nvidia"]
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    environment = kwargs["env"]
    assert isinstance(environment, dict)
    assert "NVIDIA_API_KEY" not in environment
    assert environment["NO_COLOR"] == "1"


def test_run_pi_model_list_reports_missing_command(monkeypatch) -> None:
    monkeypatch.setattr("claude_nim_proxy.pi_models.shutil.which", lambda _: None)

    with pytest.raises(PiModelListError, match="Pi command not found"):
        run_pi_model_list()


def test_run_pi_model_list_rejects_nonzero_exit(monkeypatch) -> None:
    monkeypatch.setattr("claude_nim_proxy.pi_models.shutil.which", lambda _: "/usr/bin/pi")
    monkeypatch.setattr(
        "claude_nim_proxy.pi_models.subprocess.run",
        lambda args, **kwargs: subprocess.CompletedProcess(args, 2, stdout="", stderr="broken"),
    )

    with pytest.raises(PiModelListError, match="status 2"):
        run_pi_model_list()


def test_run_pi_model_list_times_out(monkeypatch) -> None:
    monkeypatch.setattr("claude_nim_proxy.pi_models.shutil.which", lambda _: "/usr/bin/pi")

    def timeout(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(args, 15)

    monkeypatch.setattr("claude_nim_proxy.pi_models.subprocess.run", timeout)

    with pytest.raises(PiModelListError, match="timed out"):
        run_pi_model_list()


def test_run_pi_model_list_does_not_use_shell(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured.update(kwargs)
        return subprocess.CompletedProcess(args, 0, stdout=PI_OUTPUT, stderr="")

    monkeypatch.setattr("claude_nim_proxy.pi_models.shutil.which", lambda _: "/usr/bin/pi")
    monkeypatch.setattr("claude_nim_proxy.pi_models.subprocess.run", fake_run)

    run_pi_model_list()

    assert "shell" not in captured
