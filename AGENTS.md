# Project Instructions

## Source of truth

- Read `plan.md` in full before changing project code. Treat it as the product and protocol specification.
- Check the implementation tracker in `plan.md` before starting work; it records the current phase, validation evidence, commits, and blockers.
- The decisions in `plan.md` section 19 are confirmed by the user and are binding for implementation.
- Use the current repository and Git history as the source of truth when `plan.md` and the implementation diverge. Update the plan and tracker when implementation reveals a necessary specification change.

## Work in bounded phases

- Work on the current tracker phase only unless the user explicitly changes the scope.
- For protocol and translation behavior, write focused tests before or alongside implementation.
- Prefer the smallest change that satisfies the phase acceptance criteria. Do not introduce frameworks, abstractions, or dependencies not justified by the plan.
- Do not make live NVIDIA inference, authentication, or other potentially billable requests without explicit approval for that task.
- Do not weaken tests or skip validation to make a phase appear complete. Record genuine blockers in the tracker instead.

## Development workflow

- Use `uv` for dependency, environment, test, lint, and type-check commands.
- Keep `uv.lock` synchronized with dependency changes.
- Before considering a phase complete, run:

  ```bash
  uv sync --all-groups
  uv run pytest
  uv run ruff check .
  uv run mypy
  ```

- Keep changes compatible with Python 3.11+ and the strict typing configuration in `pyproject.toml`.

## Implementation tracker

- Update the tracker in `plan.md` whenever a phase starts, completes, becomes blocked, changes scope, gains or loses a decision, or changes its validation evidence.
- Record the relevant test/lint/type-check result and the phase's Git commit when one exists.
- A phase is complete only when its acceptance criteria are met, required validation passes, and the tracker reflects the final state.

## Documentation

- Keep `README.md` usage instructions synchronized with implemented CLI behavior.
- Keep `plan.md` focused on the agreed design and implementation contract; move operational details to the README as implementation matures.
