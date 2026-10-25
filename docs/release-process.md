# Release process

A release is an immutable snapshot: one Git tag, one built distribution (wheel + sdist),
and release notes. The Git tag is the only trigger; publishing to PyPI and creating the
GitHub release are automated from it.

## Versioning policy

The project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html):

| Change | Bump |
|---|---|
| Bug fix, documentation, internal cleanup | `patch` |
| New feature or behavior change | `minor` |
| Breaking change | `major` (reserved for `1.0.0` and later) |

While the version is below `1.0.0`, minor releases may contain breaking changes. The
active model JSON (`~/.config/claude-nim/models.json`) and `config.toml` belong to the
user: an upgrade never rewrites them.

`pyproject.toml` holds the single source of truth for the version. `claude_nim_proxy.
__version__` is read from the installed package metadata, and `claude-nim --version`
prints it.

## One-time setup (owner only)

1. Create the PyPI project `claude-nim-proxy` (the name must be available).
2. On PyPI, configure **Trusted Publisher** for the GitHub repository:
   - owner: `xiao-li-econ`, repository: `claude-nim-proxy`;
   - workflow name: `release.yml`, workflow file: `.github/workflows/release.yml`.
   Trusted publishing uses short-lived OIDC tokens, so no API token has to be stored.
   As a fallback, create a PyPI API token, add it as the `PYPI_API_TOKEN` secret in the
   `pypi` GitHub environment, and add `password: ${{ secrets.PYPI_API_TOKEN }}` to the
   publish step.

## Release checklist

1. **Prepare on `main`.** Land the merged pull requests, update `CHANGELOG.md` (move the
   entries under `Unreleased` into a new version section and set the release date), update
   the tracker in `plan.md`, and make sure the README is accurate.
2. **Bump the version.**

   ```bash
   uv version --bump minor   # or patch / major
   uv version --show
   ```

   This updates `pyproject.toml` and re-syncs the environment.
3. **Validate locally.**

   ```bash
   uv sync --all-groups
   uv run pytest -q
   uv run ruff check .
   uv run mypy
   uv build
   ```

4. **Open a pull request** with the version bump and changelog update, review it, and
   merge it into `main`. The CI workflow must pass.
5. **Tag the merge commit.**

   ```bash
   git checkout main
   git pull --ff-only
   git tag -a v0.2.0 -m "Release v0.2.0"
   git push origin v0.2.0
   ```

   Tags are immutable; never move or delete a published tag.
6. **Watch the release workflow.** `.github/workflows/release.yml` runs the full check
   suite, verifies that the tag matches the project version, builds the distributions,
   publishes them to PyPI, and creates a GitHub release whose notes are taken from the
   matching `CHANGELOG.md` section. If a step fails, fix the problem and re-run the
   failed workflow run.

   The workflow is safe to re-run: uploads use `skip-existing`, and the GitHub release
   step refreshes the notes of an existing release instead of failing. To (re)run it for
   an existing tag without pushing a new tag, use **Run workflow** on the `Release`
   workflow and provide the tag (for example `v0.1.0`); the workflow checks out that
   tag.
7. **Verify the published artifact.**

   ```bash
   uv tool install claude-nim-proxy==0.2.0
   claude-nim --version
   pip index versions claude-nim-proxy
   ```

8. **Announce.** The GitHub release page is the announcement; link it from the README
   "What's new" section if the release is significant.

## Maintenance notes

- GitHub Actions are pinned to exact versions (`actions/checkout@v7.0.1`,
  `astral-sh/setup-uv@v10.3.2`, `pypa/gh-action-pypi-publish@release/v1`). Bump them
  deliberately, and confirm that a new `astral-sh/setup-uv` major tag exists before
  moving to it — the project does not publish a floating tag for every major.
- The CI workflow pins the `uv` binary to the version that produced `uv.lock`
  (`0.9.2`). Update both together when upgrading uv.

## Installing a released version

```bash
uv tool install claude-nim-proxy            # newest release
uv tool install "claude-nim-proxy==0.2.0"   # pinned release
uv tool upgrade claude-nim-proxy
```

Installing straight from GitHub (`uv tool install git+https://github.com/xiao-li-econ/
claude-nim-proxy.git`) always tracks the tip of `main` and is meant for contributors and
users who want unreleased changes.
