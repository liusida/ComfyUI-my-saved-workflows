import sys
from pathlib import Path
from contextlib import contextmanager

import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None  # optional; install tqdm for progress bars

@contextmanager
def p(path):
    sys.path.insert(0, str(path)); yield; sys.path.pop(0)

with p(Path.cwd() / "user/default/workflows/2026-03-20/Embedding-Geometry"):
    try:
        sys.modules.pop('common')
    except KeyError:
        pass

    from common import (
        default_y_orthogonal_to_x,
        load_embedding_matrix,
        make_phase_dial,
        resolve_highlight_token_ids,
        spin_axis_w,
        token_id_for_string,
        update_phase_dial,
        x_unit_from_pair,
        y_at_angle,
    )

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    # HTMLWriter embeds each frame as base64 PNG; default limit is 20 (MB). Past that,
    # matplotlib SILENTLY stops appending frames — the slider "last" frame is not your
    # real last frame, so loop jumps badly even when geometry says first==last.
    "animation.embed_limit": 512,
})

REPO = "Qwen/Qwen3.5-0.8B"
TOKEN_A = " king"
TOKEN_B = " queen"

# First & last frame per spin match (phi=0 and phi=2π → same projection); loop playback is seamless.
N_FRAMES_PER_SPIN = 300
N_SPINS = 3
SPIN_SEED = 1
ANIMATION_FPS = 12
# Figure size in inches (matplotlib)
FIG_WIDTH_IN = 12
FIG_HEIGHT_IN = 7
# ``to_jshtml`` rasterizes each frame; more DPI + ``snap=False`` on scatter reduces stripe Moiré.
FIG_DPI = 160
# Also write each animation frame to PNG here (same pixels as ``savefig`` at ``FIG_DPI``).
PNG_FRAME_DIR = Path("/workspace/plots/spin-embedding-space")
# Background token cloud; lower = less solid blob in the dense center (try 0.04–0.12)
BG_SCATTER_ALPHA = 0.01
# Phase-dial inset in parent axes coords [x0, y0, width, height] (0–1), top-right
PHASE_INSET_RECT = (0.69, 0.67, 0.29, 0.27)
PHASE_RAY_RADIUS = 0.99  # inside unit circle in dial data coords

# Edit here: starred tokens (string or (string, matplotlib color))
HIGHLIGHTS = [
    (" king", "#d62728"),
    (" queen", "#1f77b4"),
    (" man", "#2ca02c"),
    (" woman", "#9467bd"),
    (" table", "#8c564b"),
    (" sun", "#ffbf00"),
    (" moon", "#7f7f7f"),
]
label_offsets = {
    " king":  (8, 8),
    " queen": (8, 8),
    " man":   (8, 0),
    " woman": (8, 8),
    " table": (10, 10),
    " sun":   (0, -16),
    " moon":  (-2, -10),
}


def _draw_highlights(ax, pts, highlight_rows, label_offsets):
    star_scatters = []
    annotations = []
    for text, tid, color in highlight_rows:
        x, y = pts[tid, 0], pts[tid, 1]
        dx, dy = label_offsets.get(text, (8, 8))
        sc = ax.scatter(
            [x], [y],
            s=120,
            c=color,
            marker="*",
            zorder=5,
        )
        star_scatters.append(sc)
        ann = ax.annotate(
            text.strip(),
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=18,
            color=color,
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy >= 0 else "top",
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc="white",
                ec="none",
                alpha=0.85,
            ),
        )
        annotations.append(ann)
    return star_scatters, annotations


torch.set_grad_enabled(False)

tokenizer, emb = load_embedding_matrix(REPO)
tid_a = token_id_for_string(tokenizer, TOKEN_A)
tid_b = token_id_for_string(tokenizer, TOKEN_B)
highlight_rows = resolve_highlight_token_ids(tokenizer, HIGHLIGHTS)

# float64 projections: HF weights are float32; keeping E in float32 for matmul can make
# y-coordinates slightly “grainer” than necessary (not stripes by itself, but avoids doubt).
E = np.asarray(emb.cpu().numpy(), dtype=np.float64)
x_dir = x_unit_from_pair(emb, tid_a, tid_b)
y_start = default_y_orthogonal_to_x(x_dir)
xs = E @ x_dir.astype(np.float64, copy=False)
rng = np.random.default_rng(SPIN_SEED)
ws = [spin_axis_w(x_dir, y_start, rng) for _ in range(N_SPINS)]
total_frames = N_SPINS * N_FRAMES_PER_SPIN


def _spin_k_phi(fi: int) -> tuple[int, int, float]:
    """Spin segment index s, frame index k within segment, φ ∈ [0, 2π] (inclusive endpoints)."""
    fi_i = int(fi)
    s = fi_i // N_FRAMES_PER_SPIN
    k = fi_i % N_FRAMES_PER_SPIN
    if N_FRAMES_PER_SPIN <= 1:
        return s, k, 0.0
    phi = 2.0 * np.pi * k / (N_FRAMES_PER_SPIN - 1)
    return s, k, phi


def coords_for_frame(fi: int) -> np.ndarray:
    s, k, phi = _spin_k_phi(fi)
    # Pin endpoints to y_start exactly (float cos/sin at 2π can drift from y_start).
    if N_FRAMES_PER_SPIN <= 1 or k == 0 or k == N_FRAMES_PER_SPIN - 1:
        y_dir = y_start
    else:
        y_dir = y_at_angle(y_start, ws[s], phi)
    ys = E @ y_dir
    # y ⊥ x ⇒ e_king·y = e_queen·y; subtract so king & queen stay at y=0 while spinning
    ys = ys - ys[tid_a]
    return np.column_stack([xs, ys])


# Numerical confirmation: first vs last frame & each spin segment (expect 0.0)
_loop_d = float(np.max(np.abs(coords_for_frame(0) - coords_for_frame(total_frames - 1))))
print(f"[loop] global first vs last max |Δpt| = {_loop_d:.3e}", flush=True)
for _s in range(N_SPINS):
    _a = _s * N_FRAMES_PER_SPIN
    _b = _a + N_FRAMES_PER_SPIN - 1
    _d = float(np.max(np.abs(coords_for_frame(_a) - coords_for_frame(_b))))
    print(f"[loop] spin {_s} first vs last max |Δpt| = {_d:.3e}", flush=True)


# Axis limits from bounds over all frames
_bounds_iter = range(total_frames)
if tqdm is not None:
    _bounds_iter = tqdm(
        _bounds_iter, total=total_frames, desc="Bounds", unit="frame",
    )
xmin = xmax = ymin = ymax = None
for fi in _bounds_iter:
    pts = coords_for_frame(fi)
    x0, x1 = float(pts[:, 0].min()), float(pts[:, 0].max())
    y0, y1 = float(pts[:, 1].min()), float(pts[:, 1].max())
    xmin = x0 if xmin is None else min(xmin, x0)
    xmax = x1 if xmax is None else max(xmax, x1)
    ymin = y0 if ymin is None else min(ymin, y0)
    ymax = y1 if ymax is None else max(ymax, y1)
dx = xmax - xmin
dy = ymax - ymin
pad = 0.05 * max(dx, dy, 1e-9)
xlim = (xmin - pad, xmax + pad)
ylim = (ymin - pad, ymax + pad)

fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=FIG_DPI)
pts0 = coords_for_frame(0)
# Banding on *some* frames only usually isn’t “rounded data”: the cloud rotates with φ,
# so the dense core beats against the pixel grid (Moiré). ``snap=False`` helps; so does DPI.
bg = ax.scatter(
    pts0[:, 0],
    pts0[:, 1],
    s=2,
    alpha=BG_SCATTER_ALPHA,
    rasterized=False,
    snap=False,
    linewidths=0,
    edgecolors="none",
    animated=True,
)
star_scatters, annotations = _draw_highlights(
    ax, pts0, highlight_rows, label_offsets,
)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("x — unit along king − queen")
ax.set_ylabel("y — spin ⊥ x\n(king & queen at 0)")
_s0, _, _phi0 = _spin_k_phi(0)
ax.set_title(
    f"Embeddings 2D — {REPO}   |   spin {_s0 + 1}/{N_SPINS}   "
    f"φ = {_phi0:.3f} rad ({_phi0 / np.pi:.3f}π)",
)

_phase_dial = make_phase_dial(
    ax,
    _phi0,
    n_spins=N_SPINS,
    inset_rect=PHASE_INSET_RECT,
    ray_radius=PHASE_RAY_RADIUS,
)

artists_spin = (bg, *star_scatters, *annotations)

_jshtml_frame_bar: list = []


def _update(fi: int):
    fi_i = int(fi)
    if tqdm is not None:
        if not _jshtml_frame_bar:
            _jshtml_frame_bar.append(
                tqdm(total=total_frames, desc="jshtml frames", unit="frame"),
            )
        _jshtml_frame_bar[0].update(1)
    elif total_frames > 0:
        if fi_i in (0, total_frames - 1) or (
            total_frames > 20 and fi_i % max(1, total_frames // 20) == 0
        ):
            print(f"jshtml frame {fi_i + 1}/{total_frames}", flush=True)

    _s, _k, _phi = _spin_k_phi(fi_i)
    ax.set_title(
        f"Embeddings 2D — {REPO}   |   spin {_s + 1}/{N_SPINS}   "
        f"φ = {_phi:.3f} rad ({_phi / np.pi:.3f}π)",
    )

    update_phase_dial(_phase_dial, _phi, _s)

    pts = coords_for_frame(fi_i)
    bg.set_offsets(np.ascontiguousarray(pts, dtype=np.float64))
    for i, (_text, tid, color) in enumerate(highlight_rows):
        x, y = pts[tid, 0], pts[tid, 1]
        star_scatters[i].set_offsets(np.array([[x, y]]))
        annotations[i].xy = (x, y)
    return artists_spin


_jshtml_frame_bar.clear()
# Tighter than default pad (~1.08); equal aspect can still leave side bands (matplotlib limitation).
plt.tight_layout(pad=0.35)
anim = FuncAnimation(
    fig,
    _update,
    frames=total_frames,
    interval=1000 / ANIMATION_FPS,
    blit=False,
)

PNG_FRAME_DIR.mkdir(parents=True, exist_ok=True)
_png_iter = range(total_frames)
if tqdm is not None:
    _png_iter = tqdm(_png_iter, total=total_frames, desc="PNG frames", unit="frame")
for _fi in _png_iter:
    _update(_fi)
    fig.savefig(
        PNG_FRAME_DIR / f"frame_{_fi:06d}.png",
        dpi=FIG_DPI,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )

# Full HTML+JS string for notebook / Comfy downstream (e.g. IPython.display.HTML(Result))
Result = anim.to_jshtml(fps=ANIMATION_FPS, default_mode="loop")
for _b in _jshtml_frame_bar:
    _b.close()
_jshtml_frame_bar.clear()
plt.close(fig)
