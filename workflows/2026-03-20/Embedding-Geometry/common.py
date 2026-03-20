"""Helpers: load HF embedding matrix, build (king−queen)-aligned 2D basis, project."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def load_embedding_matrix(
    repo_id: str,
    *,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    trust_remote_code: bool = True,
):
    """Return tokenizer and full token embedding matrix ``(vocab_size, hidden_dim)``."""
    tokenizer = AutoTokenizer.from_pretrained(
        repo_id, trust_remote_code=trust_remote_code
    )
    model = AutoModel.from_pretrained(repo_id, trust_remote_code=trust_remote_code)
    model.eval()
    emb = model.get_input_embeddings().weight.detach().to(dtype=dtype, device=device)
    return tokenizer, emb


def token_id_for_string(tokenizer, text: str) -> int:
    """Require ``text`` to map to exactly one token id (e.g. BPE ``\" king\"``)."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(
            f"Expected a single token for {text!r}, got token ids {ids}. "
            "Try another spelling or a leading space for GPT-style tokenizers."
        )
    return int(ids[0])


# Used when a highlight entry is only a string (no explicit matplotlib color).
DEFAULT_HIGHLIGHT_COLORS: tuple[str, ...] = (
    "tab:red",
    "tab:blue",
    "tab:green",
    "tab:orange",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:olive",
    "tab:cyan",
    "tab:gray",
)


def resolve_highlight_token_ids(
    tokenizer,
    specs: Sequence[str | tuple[str, str]],
) -> list[tuple[str, int, str]]:
    """
    Turn plot highlight specs into ``(token_text, token_id, color)``.

    Each spec is either a single-token string (e.g. ``\" king\"``) or
    ``(token_string, matplotlib_color)``.
    """
    out: list[tuple[str, int, str]] = []
    for i, spec in enumerate(specs):
        if isinstance(spec, tuple):
            text, color = spec[0], spec[1]
        else:
            text = spec
            color = DEFAULT_HIGHLIGHT_COLORS[i % len(DEFAULT_HIGHLIGHT_COLORS)]
        tid = token_id_for_string(tokenizer, text)
        out.append((text, tid, color))
    return out


def x_unit_from_pair(
    emb_matrix: torch.Tensor, idx_a: int, idx_b: int
) -> np.ndarray:
    """Unit vector along ``(e_a - e_b)`` in embedding space, shape ``(hidden_dim,)``."""
    e_a = emb_matrix[idx_a]
    e_b = emb_matrix[idx_b]
    x = e_a - e_b
    x = x / (x.norm() + 1e-12)
    return x.cpu().numpy().astype(np.float64, copy=False)


def default_y_orthogonal_to_x(x_unit: np.ndarray) -> np.ndarray:
    """
    Second axis: unit vector orthogonal to ``x_unit``, via Gram–Schmidt from a
    standard basis vector (pick dimension where ``|x_i|`` is smallest).

    If ``x`` is the king−queen direction, any ``y`` here satisfies
    ``e_a·y = e_b·y``, so those two tokens share the same vertical coordinate.
    """
    x = np.asarray(x_unit, dtype=np.float64).reshape(-1)
    x = x / (np.linalg.norm(x) + 1e-12)
    i_min = int(np.argmin(np.abs(x)))
    y_raw = np.zeros_like(x)
    y_raw[i_min] = 1.0
    y = y_raw - (float(y_raw @ x)) * x
    y = y / (np.linalg.norm(y) + 1e-12)
    return y


def orthonormal_basis_from_pair(
    emb_matrix: torch.Tensor, idx_a: int, idx_b: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    x axis: unit vector along (e_a - e_b).
    y axis: any unit vector orthogonal to x (built from a standard basis vector).
    """
    x_dir = x_unit_from_pair(emb_matrix, idx_a, idx_b)
    y_dir = default_y_orthogonal_to_x(x_dir)
    return x_dir, y_dir


def project_onto_x_perp(v: np.ndarray, x_unit: np.ndarray) -> np.ndarray:
    """Return ``v - (v·x)x`` with ``x`` unit; not normalized (lies in ``x^⊥``)."""
    x = np.asarray(x_unit, dtype=np.float64).reshape(-1)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    x = x / (np.linalg.norm(x) + 1e-12)
    return v - (float(v @ x)) * x


def normalize_or_raise(u: np.ndarray, eps: float = 1e-12, msg: str = "") -> np.ndarray:
    n = np.linalg.norm(u)
    if n < eps:
        raise ValueError(msg or f"expected non-zero vector, got norm {n!r}")
    return u / n


def spin_axis_w(
    x_unit: np.ndarray,
    y_unit: np.ndarray,
    rng: np.random.Generator,
    *,
    max_tries: int = 32,
) -> np.ndarray:
    """
    Random unit ``w`` with ``w·x = 0`` and ``w·y = 0`` (axis for rotating ``y``
    in a great circle inside ``x^⊥`` while keeping ``x`` fixed).
    """
    x = np.asarray(x_unit, dtype=np.float64).reshape(-1)
    y = np.asarray(y_unit, dtype=np.float64).reshape(-1)
    x = normalize_or_raise(x)
    y = normalize_or_raise(y)
    d = x.shape[0]
    for _ in range(max_tries):
        r = rng.standard_normal(d)
        u = project_onto_x_perp(r, x)
        w_raw = u - (float(u @ y)) * y
        nw = np.linalg.norm(w_raw)
        if nw >= 1e-9:
            return (w_raw / nw).astype(np.float64, copy=False)
    raise RuntimeError("spin_axis_w: failed to find non-degenerate w; try another seed")


def y_at_angle(y_unit: np.ndarray, w_unit: np.ndarray, phi: float) -> np.ndarray:
    """``cos(phi)*y + sin(phi)*w``; requires ``y``, ``w`` unit and ``y·w = 0``."""
    y = np.asarray(y_unit, dtype=np.float64).reshape(-1)
    w = np.asarray(w_unit, dtype=np.float64).reshape(-1)
    out = np.cos(phi) * y + np.sin(phi) * w
    return out / (np.linalg.norm(out) + 1e-12)


def iter_spin_frames(
    y_unit: np.ndarray,
    w_unit: np.ndarray,
    n_frames: int,
    *,
    full_turn: bool = True,
) -> Iterator[np.ndarray]:
    """
    Yield ``y(phi)`` for ``phi`` linearly spaced in ``[0, 2π)`` if ``full_turn`` and
    ``n_frames > 1``, else a single frame at ``phi=0``.
    """
    y0 = np.asarray(y_unit, dtype=np.float64).reshape(-1)
    w0 = np.asarray(w_unit, dtype=np.float64).reshape(-1)
    if n_frames <= 0:
        return
    if n_frames == 1:
        yield y_at_angle(y0, w0, 0.0)
        return
    if full_turn:
        phis = np.linspace(0.0, 2.0 * np.pi, n_frames, endpoint=False)
    else:
        phis = np.linspace(0.0, 2.0 * np.pi, n_frames, endpoint=True)
    for phi in phis:
        yield y_at_angle(y0, w0, float(phi))


def project_embeddings_2d(
    emb_matrix: torch.Tensor, x_dir: np.ndarray, y_dir: np.ndarray
) -> np.ndarray:
    """Project every token embedding row to R^2: columns are dot products with x, y."""
    e = emb_matrix.cpu().numpy()
    xs = e @ x_dir
    ys = e @ y_dir
    return np.column_stack([xs, ys])


def make_phase_dial(
    ax_parent,
    phi0: float,
    *,
    n_spins: int = 1,
    inset_rect: tuple[float, float, float, float] = (0.69, 0.70, 0.29, 0.27),
    ray_radius: float = 0.88,
):
    """
    Inset schematic for the y-spin phase φ ∈ [0, 2π]: ``n_spins`` circle outlines in a row.
    The swept wedge + ray appear only in the circle for the active spin. Starting layout
    uses ``phi0`` for spin 0. Returns a dict for :func:`update_phase_dial`.
    """
    from matplotlib.patches import Circle, Wedge

    n_spins = max(1, int(n_spins))
    ray_ref = float(ray_radius)

    dia = ax_parent.inset_axes(inset_rect)
    dia.set_aspect("equal")
    dia.set_xticks([])
    dia.set_yticks([])
    for spine in dia.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_edgecolor("0.4")
    dia.set_facecolor((1, 1, 1, 0.92))

    centers = [(float(i), 0.0) for i in range(n_spins)]
    outline_r = 0.42
    wedge_r = outline_r * min(ray_ref / 1.0, 0.95)
    pad = outline_r + 0.14
    x0, x1 = -pad, (n_spins - 1) + pad
    y0, y1 = -0.64, 0.64
    caption_txt = "0 → 2π per spin"

    dia.set_xlim(x0, x1)
    dia.set_ylim(y0, y1)

    circles = []
    wedges = []
    rays = []

    phi_deg0 = float(np.rad2deg(phi0))

    for i in range(n_spins):
        cx, cy = centers[i]
        circ = Circle(
            (cx, cy),
            radius=outline_r,
            facecolor="none",
            edgecolor="0.35",
            lw=1.15,
            zorder=2,
        )
        dia.add_patch(circ)
        circles.append(circ)

        wedge = Wedge(
            (cx, cy),
            wedge_r,
            0.0,
            max(phi_deg0, 1e-6),
            width=0,
            facecolor=(0.95, 0.42, 0.12, 0.32),
            edgecolor="none",
            zorder=1,
        )
        dia.add_patch(wedge)
        wedges.append(wedge)

        phi_i = phi0 if i == 0 else 0.0
        ray_line, = dia.plot(
            [cx, cx + wedge_r * np.cos(phi_i)],
            [cy, cy + wedge_r * np.sin(phi_i)],
            color="tab:red",
            lw=2.3,
            solid_capstyle="round",
            zorder=4,
        )
        rays.append(ray_line)

        if i != 0:
            wedge.set_visible(False)
            ray_line.set_visible(False)
        elif phi0 < 1e-7:
            wedge.set_visible(False)

        if n_spins > 1:
            dia.text(
                cx,
                cy - outline_r - 0.11,
                str(i + 1),
                ha="center",
                va="top",
                fontsize=9,
                color="0.45",
            )

    dia.set_title("rotation directions and phase φ", fontsize=11, color="0.15", pad=6)
    # Use axes coordinates (0–1), not data coords: with aspect="equal", the displayed
    # y-limits are often expanded after set_ylim, so data (x, y) for text no longer match
    # the visual bottom — see Axes.set_aspect / Axes.text(..., transform=ax.transAxes).
    dia.text(
        0.5,
        -0.15,
        caption_txt,
        transform=dia.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color="0.45",
        clip_on=False,
    )

    return {
        "ax": dia,
        "n_spins": n_spins,
        "centers": centers,
        "circles": circles,
        "wedges": wedges,
        "rays": rays,
        "wedge_r": wedge_r,
        "ray_radius": ray_ref,
    }


def update_phase_dial(dial: dict, phi: float, spin_index: int = 0) -> None:
    """Update wedge + ray for the active spin (``spin_index``) from ``make_phase_dial``."""
    r = float(dial["wedge_r"])
    n = int(dial["n_spins"])
    si = int(np.clip(spin_index, 0, max(n - 1, 0)))
    centers = dial["centers"]
    wedges = dial["wedges"]
    rays = dial["rays"]
    phi_deg = float(np.rad2deg(phi))

    for i in range(n):
        wd = wedges[i]
        ray = rays[i]
        cx, cy = centers[i]
        if i != si:
            wd.set_visible(False)
            ray.set_visible(False)
            continue

        ray.set_visible(True)
        ray.set_data([cx, cx + r * np.cos(phi)], [cy, cy + r * np.sin(phi)])
        if phi < 1e-7:
            wd.set_visible(False)
        else:
            wd.set_visible(True)
            wd.set_theta1(0.0)
            wd.set_theta2(min(phi_deg, 360.0))
