# acdc-jax

Differentiable atmospheric molecular-cluster dynamics in Python/JAX: a
faithful port of [ACDC](https://github.com/tolenius/ACDC), validated term by
term against the Fortran reference.

- **Physics** — the birth–death equations and every rate expression, with
  Perl/Fortran line citations.
- **Fidelity** — each place the port could differ from upstream, the flag
  that controls it, and the upstream defects found along the way (F1–F23).
- **Validation** — the acceptance gates and why they are what they are.
- **API** — the modules, in pipeline order.
- **Plan** — the phase-by-phase record, each slice one commit.

```bash
uv sync --extra dev
uv run pytest
uv run python scripts/make_figures.py
```
