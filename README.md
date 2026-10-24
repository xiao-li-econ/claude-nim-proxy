# claude-nim-proxy

A planned thin local translator that lets Claude Code use NVIDIA NIM models through the Anthropic Messages API.

The project is currently in the planning/setup stage. See [`plan.md`](plan.md) for the architecture, protocol mappings, CLI design, credential handling, testing strategy, and implementation phases.

## Development with uv

This project uses [uv](https://docs.astral.sh/uv/) for Python version management, dependency resolution, locking, and the development environment.

```bash
# Create/sync the environment and install all development dependencies
uv sync --all-groups

# Run the test suite
uv run pytest

# Lint and type-check
uv run ruff check .
uv run mypy
```

`uv.lock` is committed so dependency resolution remains reproducible across machines.
