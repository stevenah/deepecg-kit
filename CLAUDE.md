# CLAUDE.md

Development commands, workflow, and release process are documented in
[CONTRIBUTING.md](CONTRIBUTING.md). Run bare `make` for the target list.

Notes for agents:
- Use `make check` (read-only) before declaring work done; `make fix` for style.
- Always invoke tools via `uv run` / the Makefile — never bare `ruff`/`pytest`
  from PATH.
- `tests/test_state_dict_compat.py` guards checkpoint compatibility: never
  rename model module attributes; regenerate its fixture only for a
  deliberate breaking release.
