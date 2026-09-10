"""Regenerate every figure in figures/ from scratch.

Usage::

    uv run python scripts/make_figures.py
    uv run python scripts/make_figures.py --only parity

Each figure answers one question. The caption printed alongside names the
command that produced it, so a figure in a talk can always be traced back.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from acdc_jax import (  # noqa: E402
    config,
    free_energy,
    pathways,
    rates,
    rhs,
    sensitivity,
    solve,
)
from acdc_jax.boundary import BoundarySystem  # noqa: E402
from acdc_jax.clusterset import parse_cluster_set  # noqa: E402
from acdc_jax.reactions import enumerate_reactions  # noqa: E402
from acdc_jax.system import build_system  # noqa: E402
from acdc_jax.thermo import parse_dipole_file, parse_energy_file  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
INPUTS = REPO / "fortran/src/Perl_input"
GOLDENS = REPO / "validation/goldens"
FIGURES = REPO / "figures"

# Validated categorical palette, first slots in fixed order. Never cycled:
# a fifth series folds into "other" or gets its own facet.
# Checked with the dataviz validator (light surface #fcfcfb): all gates pass;
# aqua sits below 3:1 contrast, so it is always directly labelled.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
VIOLET = "#4a3aa7"

# Diverging pair for signed quantities: two poles that read as opposite,
# with a neutral -- never a hue -- at the midpoint.
POS, NEG, MID = BLUE, "#e34948", "#f0efec"

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8983"
SURFACE = "#fcfcfb"
GRID = "#e6e5e1"

CM3 = 1e6  # cm^-3 -> m^-3


def style() -> None:
    """Recessive axes and grid; ink for all text."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "medium",
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "text.color": INK,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 2.0,
            "figure.dpi": 130,
        }
    )


def build():
    cluster_set = parse_cluster_set(INPUTS / "input_ANnarrow_neutral_neg_pos.inp")
    system = build_system(cluster_set)
    boundary = BoundarySystem(cluster_set)
    reactions = enumerate_reactions(system, boundary)
    energies = parse_energy_file(
        INPUTS / "HS298.15K_example.txt", cluster_set.molecule_names
    )
    dipoles = parse_dipole_file(
        INPUTS / "dip_pol_298.15K_example.txt", cluster_set.molecule_names
    )
    inputs = rates.build_rate_inputs(
        system, cluster_set, energies, dipoles, reactions=reactions
    )
    return system, reactions, inputs


def _tidy(ax, *, grid_axis: str = "both") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------


def figure_parity(model) -> str:
    """J vs sulfuric acid, port against the Fortran reference.

    The headline validation. A line for the continuous port result, markers
    for the discrete Fortran grid points, so agreement is read as markers
    sitting on the line rather than as two lines to disentangle. The residual
    panel below carries the number that matters.
    """
    system, reactions, inputs = model
    path = GOLDENS / "steadystate.npz"
    goldens = np.load(path)
    conditions, reference_j = goldens["conditions"], goldens["j"]

    # Hold everything but [A] at the bundled example's values.
    c_n, temperature, cs_ref, ipr = 1e15, 280.0, 1e-3, 3.0 * CM3
    mask = (
        (conditions[:, 1] == c_n)
        & (conditions[:, 2] == temperature)
        & (conditions[:, 3] == cs_ref)
    )
    ref_c_a = conditions[mask, 0]
    ref_values = reference_j[mask]
    order = np.argsort(ref_c_a)
    ref_c_a, ref_values = ref_c_a[order], ref_values[order]

    dense = np.logspace(np.log10(ref_c_a.min()), np.log10(ref_c_a.max()), 12)
    port = sensitivity.sweep(
        system, reactions, inputs, dense, c_n, temperature, cs_ref, ipr
    )
    port_at_ref = sensitivity.sweep(
        system, reactions, inputs, ref_c_a, c_n, temperature, cs_ref, ipr
    )
    residual = np.abs(port_at_ref - ref_values) / ref_values

    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(6.2, 5.4), height_ratios=[3, 1], sharex=True
    )

    top.plot(dense / CM3, port / CM3, color=BLUE, zorder=3, label="acdc-jax")
    top.plot(
        ref_c_a / CM3,
        ref_values / CM3,
        "o",
        markersize=7,
        markerfacecolor="none",
        markeredgecolor=ORANGE,
        markeredgewidth=2,
        zorder=4,
        label="Fortran ACDC",
    )
    top.set_xscale("log")
    top.set_yscale("log")
    top.set_ylabel("formation rate $J$  (cm$^{-3}$ s$^{-1}$)")
    top.set_title(
        "Steady-state particle formation rate\n"
        "sulfuric acid–ammonia, 280 K, [NH$_3$] = 10$^9$ cm$^{-3}$",
        loc="left",
        color=INK,
    )
    _tidy(top)
    top.legend(loc="lower right")

    # Direct labels as well as the legend: identity is never colour-alone.
    top.annotate(
        "acdc-jax",
        xy=(dense[-3] / CM3, port[-3] / CM3),
        xytext=(6, -14),
        textcoords="offset points",
        color=INK_SECONDARY,
        fontsize=8,
    )

    bottom.plot(
        ref_c_a / CM3, residual, "o-", color=INK_MUTED, markersize=4, linewidth=1.2
    )
    bottom.axhline(1e-5, color=ORANGE, linewidth=1.2, linestyle="--")
    bottom.annotate(
        "acceptance gate  1e-5",
        xy=(ref_c_a[0] / CM3, 1e-5),
        xytext=(2, 4),
        textcoords="offset points",
        color=ORANGE,
        fontsize=7.5,
    )
    bottom.set_yscale("log")
    bottom.set_ylim(1e-9, 3e-5)
    bottom.set_xlabel("[H$_2$SO$_4$]  (cm$^{-3}$)")
    bottom.set_ylabel("relative\ndifference")
    _tidy(bottom, grid_axis="y")

    figure.tight_layout()
    out = FIGURES / "01_parity_j_vs_sulfuric_acid.png"
    figure.savefig(out, bbox_inches="tight")
    plt.close(figure)
    return (
        f"{out.name}: worst relative difference {residual.max():.1e} "
        f"across {len(ref_c_a)} reference points"
    )


def figure_distribution(model) -> str:
    """Steady-state cluster concentrations against size, split by charge.

    Concentration against mass diameter rather than cluster index: the index
    is an implementation detail, size is the physical coordinate and the one
    the coagulation sink and the grow-out criteria are defined on.
    """
    system, reactions, inputs = model
    coefficients = rhs.assemble(
        system, reactions, inputs, 280.0, 1e-3, 3.0 * CM3, 3.0 * CM3
    )
    c0 = solve.set_vapours(
        system, np.zeros(system.n_equations), {"1A": 1e13, "1N": 1e15}
    )
    result = solve.steady_state_by_integration(system, coefficients, c0)
    concentration = np.asarray(result.concentrations[: system.n_clusters])
    diameter = inputs.diameter_nm
    charge = np.asarray(system.charges)

    figure, ax = plt.subplots(figsize=(6.2, 4.2))

    for label, value, colour in (
        ("neutral", 0, BLUE),
        ("negative", -1, ORANGE),
        ("positive", 1, AQUA),
    ):
        pick = (charge == value) & (concentration > 0)
        ax.plot(
            diameter[pick],
            concentration[pick] / CM3,
            "o",
            markersize=7,
            color=colour,
            markeredgecolor=SURFACE,
            markeredgewidth=1.4,  # 2px surface ring on overlapping marks
            label=label,
            zorder=3,
        )

    ax.set_yscale("log")
    # Clip the bottom: species below ~1e-9 cm^-3 are numerically zero (far
    # under the integrator's absolute tolerance) and stretch the axis over
    # twelve empty decades. The count of omitted species is stated so the
    # clipping is not silent.
    floor = 1e-9
    omitted = int(((concentration / CM3) < floor).sum())
    ax.set_ylim(floor, None)
    if omitted:
        ax.annotate(
            f"{omitted} species below 10$^{{-9}}$ cm$^{{-3}}$ not shown",
            xy=(0.02, 0.03),
            xycoords="axes fraction",
            color=INK_MUTED,
            fontsize=7.5,
        )
    ax.set_xlabel("mass diameter  (nm)")
    ax.set_ylabel("concentration  (cm$^{-3}$)")
    ax.set_title(
        "Steady-state cluster population\n"
        "54 clusters, 280 K, [H$_2$SO$_4$] = 10$^7$ cm$^{-3}$",
        loc="left",
        color=INK,
    )
    _tidy(ax)
    ax.legend(loc="upper right", title=None)

    # Aqua is below the 3:1 contrast floor, so it carries a direct label.
    positive = (charge == 1) & (concentration > 0)
    if positive.any():
        i = np.argmax(np.where(positive, concentration, 0))
        ax.annotate(
            "positive",
            xy=(diameter[i], concentration[i] / CM3),
            xytext=(8, 2),
            textcoords="offset points",
            color=INK_SECONDARY,
            fontsize=8,
        )

    figure.tight_layout()
    out = FIGURES / "02_cluster_distribution.png"
    figure.savefig(out, bbox_inches="tight")
    plt.close(figure)
    span = concentration[concentration > 0]
    return f"{out.name}: concentrations span {span.max() / span.min():.1e}"


def figure_sensitivity(model) -> str:
    """dJ/d(Delta-H) per cluster -- the capability the reference lacks.

    Signed, so a diverging pair with a neutral midpoint. Horizontal bars
    because the category labels are cluster formulae and need reading.
    Only the top twelve: past that the values are noise and the labels
    collide.
    """
    system, reactions, inputs = model
    conditions = sensitivity.Conditions(
        c_a=1e13, c_n=1e15, temperature=280.0, cs_ref=1e-3, ipr=3.0 * CM3
    )
    result = sensitivity.sensitivity_to_free_energies(
        system, reactions, inputs, conditions
    )
    relative = np.asarray(result["relative_dh"])

    # Pick the twelve most influential, then re-sort by SIGNED value so the
    # diverging chart reads as two blocks. Sorting by magnitude makes the
    # signs alternate down the axis, which looks like noise.
    top = np.argsort(-np.abs(relative))[:12]
    top = top[np.argsort(relative[top])]
    values = relative[top]
    labels = [system.labels[i] for i in top]

    figure, ax = plt.subplots(figsize=(6.2, 4.6))
    colours = [POS if v > 0 else NEG for v in values]
    ax.barh(
        np.arange(len(values)),
        values,
        color=colours,
        height=0.68,
        zorder=3,
        edgecolor=SURFACE,
        linewidth=1.2,  # 2px surface gap between adjacent bars
    )
    ax.axvline(0, color=INK_MUTED, linewidth=1.0, zorder=2)
    ax.set_yticks(np.arange(len(values)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(r"$\partial \ln J\,/\,\partial \Delta H$   (per kcal mol$^{-1}$)")
    ax.set_title(
        "Which cluster free energies control the formation rate\n"
        "gradient through the steady-state solve, validated to 2e-7\n"
        "blue: raising $\\Delta H$ raises $J$      red: lowers it",
        loc="left",
        color=INK,
    )
    _tidy(ax, grid_axis="x")

    # Room for the outward direct labels at both extremes; without the
    # margin the negative one lands on top of the y-tick labels.
    reach = np.abs(values).max()
    ax.set_xlim(-reach * 1.35, reach * 1.35)

    # Selective direct labels: the two extremes only, never every bar. With
    # the signed sort these are the first and last rows.
    for rank in (0, len(values) - 1):
        value = values[rank]
        ax.annotate(
            f"{value:+.3f}",
            xy=(value, rank),
            xytext=(6 if value > 0 else -6, 0),
            textcoords="offset points",
            va="center",
            ha="left" if value > 0 else "right",
            color=INK_SECONDARY,
            fontsize=8,
        )

    figure.tight_layout()
    out = FIGURES / "03_free_energy_sensitivity.png"
    figure.savefig(out, bbox_inches="tight")
    plt.close(figure)
    return f"{out.name}: largest |dlnJ/dH| = {np.abs(relative).max():.3f} per kcal/mol"


def figure_stability(model) -> str:
    """Evaporation rate against cluster size, with the growth rate as a line.

    The system-adequacy check the ACDC QuickGuide prescribes: the largest
    clusters in the set must be stable, meaning their evaporation rate sits
    well below the rate at which they gain another acid molecule. If the
    biggest clusters still evaporate quickly, the cluster set is too small
    and the formation rate is an artefact of where it was truncated.

    A first version plotted evaporation against collision frequency as a
    scatter with a 1:1 line. That form fails: the acid collision frequency
    is nearly constant across clusters (~1e-2 s^-1 -- collision rates vary
    weakly with size) while evaporation spans ten orders, so every point
    collapses onto a vertical strip in an empty panel. Size on the x-axis
    with the frequency as a horizontal reference is the same comparison,
    legibly.
    """
    system, reactions, inputs = model
    temperature = 280.0
    collision = np.asarray(rates.collision_coefficients(inputs, temperature))

    parents = np.array([e.k for e in reactions.evaporations])
    di = np.array([e.i for e in reactions.evaporations])
    dj = np.array([e.j for e in reactions.evaporations])
    evaporation = np.asarray(
        rates.evaporation_for_pairs(inputs, collision, parents, di, dj, temperature)
    )

    total = free_energy.total_evaporation_rate(system.n_clusters, parents, evaporation)

    acid = system.labels.index("1A")
    c_acid = 1e13  # [H2SO4] = 1e7 cm^-3
    growth = collision[:, acid] * c_acid
    diameter = inputs.diameter_nm
    charge = np.asarray(system.charges)

    figure, ax = plt.subplots(figsize=(6.4, 4.4))

    ax.axhspan(1e-20, float(np.median(growth)), color=GRID, alpha=0.55, zorder=0)
    ax.axhline(
        float(np.median(growth)),
        color=INK_MUTED,
        linewidth=1.4,
        linestyle="--",
        zorder=2,
    )

    for label, value, colour in (
        ("neutral", 0, BLUE),
        ("negative", -1, ORANGE),
        ("positive", 1, AQUA),
    ):
        pick = (charge == value) & (total > 0)
        ax.plot(
            diameter[pick],
            total[pick],
            "o",
            markersize=7,
            color=colour,
            markeredgecolor=SURFACE,
            markeredgewidth=1.4,
            label=label,
            zorder=3,
        )

    ax.set_yscale("log")
    ax.set_ylim(3e-7, 1e6)
    ax.set_xlabel("mass diameter  (nm)")
    ax.set_ylabel("total evaporation rate  (s$^{-1}$)")
    ax.set_title(
        "Cluster stability at 280 K\n"
        "the system-adequacy check from the ACDC QuickGuide",
        loc="left",
        color=INK,
    )
    _tidy(ax)
    ax.legend(loc="upper right")

    # Blended transform: x in axes fraction, y in data. Anchoring x to
    # diameter.min() put this label at 0.39 nm -- the generic charger ion,
    # which is off the left of the visible range -- so it was clipped away.
    ax.annotate(
        "acid collision frequency at [H$_2$SO$_4$] = 10$^7$ cm$^{-3}$",
        xy=(0.02, float(np.median(growth))),
        xycoords=ax.get_yaxis_transform(),
        xytext=(0, 6),
        textcoords="offset points",
        color=INK_SECONDARY,
        fontsize=7.5,
    )
    ax.annotate(
        "stable: grows faster than it evaporates",
        xy=(0.03, 0.06),
        xycoords="axes fraction",
        color=INK_SECONDARY,
        fontsize=8,
    )
    # Aqua sits below the contrast floor, so it takes a direct label too.
    positive = (charge == 1) & (total > 0)
    if positive.any():
        i = int(np.argmax(np.where(positive, diameter, 0)))
        ax.annotate(
            "positive",
            xy=(diameter[i], total[i]),
            xytext=(-6, 8),
            textcoords="offset points",
            ha="right",
            color=INK_SECONDARY,
            fontsize=8,
        )

    figure.tight_layout()
    out = FIGURES / "04_cluster_stability.png"
    figure.savefig(out, bbox_inches="tight")
    plt.close(figure)
    stable = float((total[total > 0] < np.median(growth)).mean())
    return f"{out.name}: {stable:.0%} of evaporating clusters are below the growth rate"


# ---------------------------------------------------------------------------
T_QUICKGUIDE = 280.0
"""The setup QuickGuide's conditions: [H2SO4] = 5e6 cm^-3, [NH3] = 100 ppt,
280 K, coagulation sink 1e-3 1/s, ion production 3 cm^-3 s^-1."""


def _quickguide_coefficients(model):
    system, reactions, inputs = model
    return rhs.assemble(system, reactions, inputs, T_QUICKGUIDE, 1e-3, 3e6, 3e6)


def _quickguide_state(model):
    """Steady state at the QuickGuide's conditions."""
    from acdc_jax.thermo import reference_number_density

    system, _, _ = model
    co = _quickguide_coefficients(model)
    c_n = 100e-12 * reference_number_density(config.P_ATM, T_QUICKGUIDE)
    c0 = solve.set_vapours(
        system, np.zeros(system.n_equations), {"1A": 5e6 * CM3, "1N": c_n}
    )
    result = solve.solve_steady_state(system, co, c0=c0)
    return co, np.asarray(result.concentrations)


def _acid_base(system, label: str) -> tuple[int, int]:
    """(acid, base) chart coordinates of a label -- in or out of the set --
    with the bisulfate ion folded into the acid count and the proton ignored."""
    from acdc_jax import labels as label_module

    counts = np.array(label_module.composition_vector(label, system.order))
    return int(free_energy.folded_count(system, counts, "A")), int(
        free_energy.folded_count(system, counts, "N")
    )


def figure_pathway(model) -> str:
    """The neutral growth pathway in composition space -- the QuickGuide's
    slide 6, from the port's own flux tracking.

    Each arrow is a significant flux into a cluster (>= 5% of its inflow),
    from the growing collider; the arrow out of the set is the exit channel.
    Colour is the flux on a log scale; the main route is drawn heavier. The
    dashed lines mark the largest neutral cluster in the set: everything
    beyond them is 'out'. What the reader should take away is the same
    conclusion the QuickGuide draws for this system: growth runs along and
    below the acid = base diagonal, so a boundary that lets clusters with
    more acid than base grow out is a defensible one.
    """
    system, _, _ = model
    co, c = _quickguide_state(model)
    pw = pathways.track_pathways(system, co, c, charge=0)

    from acdc_jax import labels as label_module

    def charge_of(label: str) -> int:
        if label in system.labels:
            return system.charges[system.labels.index(label)]
        counts = label_module.composition_vector(label, system.order)
        return int(
            sum(n * q for n, q in zip(counts, system.charges_of_molecule, strict=True))
        )

    # MATLAB draws an arrow only between two clusters of the wanted charge
    # (track_fluxes.m :313-345); cross-charge sources get a text box there
    # and are simply left out here.
    edges = [
        e
        for e in list(pw.edges) + list(pw.exits)
        if e.start in system.labels
        and charge_of(e.start) == 0
        and charge_of(e.end) == 0
    ]
    values = np.array([e.value for e in edges]) / CM3
    vmin, vmax = values.min(), values.max()
    norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "flux", [BLUE, VIOLET, NEG]
    )
    route = set(zip(pw.main_route[:-1], pw.main_route[1:], strict=True))

    figure, ax = plt.subplots(figsize=(6.0, 5.2))
    for edge, value in zip(edges, values, strict=True):
        (x0, y0), (x1, y1) = (
            _acid_base(system, edge.start),
            _acid_base(system, edge.end),
        )
        on_route = (edge.start, edge.end) in route
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={
                "arrowstyle": "-|>",
                "color": cmap(norm(value)),
                "lw": 3.2 if on_route else 1.4,
                "shrinkA": 6,
                "shrinkB": 6,
                "alpha": 1.0 if on_route else 0.8,
            },
            zorder=3 if on_route else 2,
        )
    neutral = [i for i in range(system.n_clusters) if system.charges[i] == 0]
    max_acid = max(_acid_base(system, system.labels[i])[0] for i in neutral)
    max_base = max(_acid_base(system, system.labels[i])[1] for i in neutral)
    ax.axvline(max_acid + 0.5, color=INK, ls="--", lw=1.2)
    ax.axhline(max_base + 0.5, color=INK, ls="--", lw=1.2)
    ax.plot([0, max_acid + 1], [0, max_acid + 1], color=INK_MUTED, ls=":", lw=1)
    reach_a = max([max_acid] + [_acid_base(system, e.end)[0] for e in edges])
    reach_b = max([max_base] + [_acid_base(system, e.end)[1] for e in edges])
    ax.set_xlim(-0.5, reach_a + 0.5)
    ax.set_ylim(-0.5, reach_b + 0.5)
    ax.set_xticks(range(reach_a + 1))
    ax.set_yticks(range(reach_b + 1))
    ax.set_xlabel("H$_2$SO$_4$ molecules")
    ax.set_ylabel("NH$_3$ molecules")
    ax.set_aspect("equal")
    _tidy(ax)
    ax.grid(True, alpha=0.5)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = figure.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("flux (cm$^{-3}$ s$^{-1}$)")
    ax.set_title(
        "Neutral growth pathways\n"
        "[H$_2$SO$_4$] = 5·10$^6$ cm$^{-3}$, [NH$_3$] = 100 ppt, 280 K",
        fontsize=9,
        loc="left",
    )
    route_text = " → ".join(pw.main_route)
    figure.text(
        0.0,
        -0.14,
        "\n".join(textwrap.wrap(f"main route: {route_text}", width=64)),
        transform=ax.transAxes,
        fontsize=8,
        color=INK_SECONDARY,
        va="top",
    )
    ax.annotate(
        "dashed: edge of the cluster set (growth beyond is counted as J)",
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        color=INK_SECONDARY,
        fontsize=8,
    )
    figure.tight_layout()
    out = FIGURES / "05_growth_pathway.png"
    figure.savefig(out, bbox_inches="tight")
    plt.close(figure)
    return f"{out.name}: main route {' -> '.join(pw.main_route)}"


def figure_evaporation_map(model) -> str:
    """Total evaporation rate of every neutral cluster on the (acid, base)
    grid -- the QuickGuide's slide 3, the first check of a cluster set.

    The reader is looking for the largest clusters (top right) to be cold:
    if the corner still evaporates at 1e2 s^-1 or more, the set is too small
    and J is an artefact of where it was cut. Cells print log10 of the rate.
    """
    system, _, _ = model
    co = _quickguide_coefficients(model)
    temperature = T_QUICKGUIDE
    total = free_energy.total_evaporation_rate(
        system.n_clusters, co.evaporation_k, co.evaporation_rate
    )
    grid = free_energy.composition_grid(system, total, "A", "N", charge=0)
    log_grid = np.log10(np.where(grid > 0, grid, np.nan))

    figure, ax = plt.subplots(figsize=(6.0, 5.0))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "evap", [BLUE, MID, ORANGE]
    )
    image = ax.imshow(
        log_grid.T,
        origin="lower",
        cmap=cmap,
        vmin=float(np.nanmin(log_grid)),
        vmax=float(np.nanmax(log_grid)),
        aspect="equal",
    )
    for a in range(grid.shape[0]):
        for b in range(grid.shape[1]):
            if np.isnan(grid[a, b]):
                continue
            value = grid[a, b]
            if value <= 0:
                continue
            text = f"$10^{{{int(np.round(np.log10(value)))}}}$"
            ax.text(a, b, text, ha="center", va="center", fontsize=8, color=INK)
    ax.set_xticks(range(grid.shape[0]))
    ax.set_yticks(range(grid.shape[1]))
    ax.set_xlabel("H$_2$SO$_4$ molecules")
    ax.set_ylabel("NH$_3$ molecules")
    ax.set_title(
        "Total evaporation rate Σγ (s$^{-1}$) of neutral clusters "
        f"at {temperature:g} K",
        fontsize=9,
        loc="left",
    )
    cbar = figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log$_{10}$ Σγ (s$^{-1}$)")
    for spine in ax.spines.values():
        spine.set_visible(False)
    figure.tight_layout()
    out = FIGURES / "06_evaporation_map.png"
    figure.savefig(out, bbox_inches="tight")
    plt.close(figure)
    low, high = np.nanmin(log_grid), np.nanmax(log_grid)
    return f"{out.name}: evaporation spans 10^{low:.0f} to 10^{high:.0f} s^-1"


FIGURE_FUNCTIONS = {
    "parity": figure_parity,
    "distribution": figure_distribution,
    "sensitivity": figure_sensitivity,
    "stability": figure_stability,
    "pathway": figure_pathway,
    "evaporation": figure_evaporation_map,
}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=sorted(FIGURE_FUNCTIONS))
    args = parser.parse_args(argv[1:])

    FIGURES.mkdir(exist_ok=True)
    style()
    model = build()

    names = [args.only] if args.only else list(FIGURE_FUNCTIONS)
    for name in names:
        print(f"  {FIGURE_FUNCTIONS[name](model)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
