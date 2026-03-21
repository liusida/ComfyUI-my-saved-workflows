#### Cell: 1 ####
"""
PCA-2 embedding plot: fit plane on ``PCA_SUBSET_TOKENS``, project full vocab, ``plt.show()``.
"""
import sys
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import to_hex

@contextmanager
def p(path):
    sys.path.insert(0, str(path))
    yield
    sys.path.pop(0)


try:
    _WORKFLOW_DIR = Path(__file__).resolve().parent
except NameError:
    _cand = (
        Path.cwd() / "workflows/2026-03-21",
        Path.cwd() / "user/default/workflows/2026-03-21",
    )
    _WORKFLOW_DIR = next(
        (c for c in _cand if (c / "Embedding-Numbers" / "common.py").is_file()),
        _cand[0],
    )

with p(_WORKFLOW_DIR / "Embedding-Numbers"):
    try:
        sys.modules.pop("common")
    except KeyError:
        pass
    from common import (
        draw_highlights,
        load_embedding_matrix,
        pca2_orthonormal_basis,
        resolve_highlight_token_ids,
        token_id_for_string,
    )

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

REPO = "Qwen/Qwen3.5-0.8B"

torch.set_grad_enabled(False)

tokenizer, emb = load_embedding_matrix(REPO)

#### Cell: 2 ####

FIG_WIDTH_IN = 10
FIG_HEIGHT_IN = 10
FIG_DPI = 100
BG_SCATTER_ALPHA = 0.01

_TAB20 = plt.colormaps["tab20"].colors

# --- Token categories (first principles) ---
# 1) Each category is one *semantic field* (mutually interpretable under one color).
# 2) Same number of tokens per category → comparable density / legend weight (here: 4 each).
# 3) Strings must be *single tokenizer pieces* for your model (BPE: often leading space
#    before a word piece; digits often bare). If encode() returns >1 id, change spelling.
# 4) PCA_SUBSET_TOKENS: pick any subset (≥2) from the union—e.g. 3 or 4 tokens from
#    different categories. Order fixes the PCA fit.

TOKENS_PER_CATEGORY = 4

TOKEN_CATEGORIES: list[tuple[str, str, tuple[str, ...]]] = [
    # Decimal digit type (numeric symbol)
    (
        "digits",
        to_hex(_TAB20[0]),
        ("0", "1", "2", "3", "4", "5", "6", "9"),
    ),
    # Letter-as-token (continuation after space in many BPE vocabularies)
    (
        "letters",
        to_hex(_TAB20[2]),
        (" a", " b", " c", " d"),
    ),
    # Major city names as single tokens (leading space = word-initial piece)
    (
        "cities",
        to_hex(_TAB20[4]),
        (" Paris", " Tokyo", " Beijing", " London"),
    ),
    # Common concrete nouns (animals)
    (
        "animals",
        to_hex(_TAB20[6]),
        (" dog", " cat", " bird", " fish"),
    ),
    # Basic color words
    (
        "colors",
        to_hex(_TAB20[8]),
        (" red", " blue", " green", " white"),
    ),
    # High-frequency action verbs (often single-piece)
    (
        "verbs",
        to_hex(_TAB20[10]),
        (" go", " get", " see", " make"),
    ),
    # Food / drink nouns
    (
        "food",
        to_hex(_TAB20[12]),
        (" water", " bread", " meat", " rice"),
    ),
    # Calendar / duration units
    (
        "time",
        to_hex(_TAB20[14]),
        (" day", " year", " week", " hour"),
    ),
    #
    (
        "",
        to_hex(_TAB20[16]),
        (" yourselves", " myself", " themselves", " himself"),
    ),
    #
    (
        "",
        to_hex(_TAB20[18]),
        ("\n", "-", " theirs"),
    ),
    
]


# Tokens whose embeddings define PCA-2 (length ≥2; often 3–4). Edit freely.
PCA_SUBSET_TOKENS = (" yourselves", "0", " year")
PCA_TOKEN_COLOR = "black"  # stars for tokens that define the PCA plane

_token_to_color: dict[str, str] = {}
for _cat, color, toks in TOKEN_CATEGORIES:
    for t in toks:
        if t in _token_to_color:
            raise ValueError(f"token {t!r} appears in more than one category")
        _token_to_color[t] = color

_all_highlight_tokens = frozenset(_token_to_color)
_missing = [t for t in PCA_SUBSET_TOKENS if t not in _all_highlight_tokens]
if _missing:
    raise ValueError(
        f"PCA_SUBSET_TOKENS must be drawn from TOKEN_CATEGORIES; missing: {_missing}"
    )
if len(PCA_SUBSET_TOKENS) < 2:
    raise ValueError(
        f"PCA_SUBSET_TOKENS needs at least 2 tokens for PCA-2, got {len(PCA_SUBSET_TOKENS)}"
    )

_pca_star_tokens = frozenset(PCA_SUBSET_TOKENS)

# Stars: category color, except PCA-defining tokens (black).
HIGHLIGHTS: list[tuple[str, str]] = [
    (
        t,
        PCA_TOKEN_COLOR if t in _pca_star_tokens else _token_to_color[t],
    )
    for _cat, _c, toks in TOKEN_CATEGORIES
    for t in toks
]

label_offsets: dict[str, tuple[int, int]] = {}

highlight_rows = resolve_highlight_token_ids(tokenizer, HIGHLIGHTS)

E = np.asarray(emb.cpu().numpy(), dtype=np.float64)
pca_token_ids = [token_id_for_string(tokenizer, t) for t in PCA_SUBSET_TOKENS]
E_subset = E[pca_token_ids]
# pc1, pc2, evr1, evr2 = pca2_orthonormal_basis(E_subset)
# DEBUG: two random orthonormal directions in embedding space (not PCA).
_d = int(E.shape[1])
_rng = np.random.default_rng()
_a = _rng.standard_normal(_d)
pc1 = _a / (np.linalg.norm(_a) + 1e-12)
_b = _rng.standard_normal(_d)
_b = _b - float(_b @ pc1) * pc1
pc2 = _b / (np.linalg.norm(_b) + 1e-12)
evr1 = float("nan")
evr2 = float("nan")
print(
    f"[debug] random orthonormal plane (PCA commented)  |  subset {PCA_SUBSET_TOKENS}",
    flush=True,
)

pts = np.column_stack([E @ pc1, E @ pc2])
pad = 0.05 * max(
    float(pts[:, 0].max() - pts[:, 0].min()),
    float(pts[:, 1].max() - pts[:, 1].min()),
    1e-9,
)

fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=FIG_DPI)
ax.scatter(
    pts[:, 0],
    pts[:, 1],
    s=2,
    alpha=BG_SCATTER_ALPHA,
    snap=False,
    linewidths=0,
    edgecolors="none",
)
draw_highlights(ax, pts, highlight_rows, label_offsets)
# ax.set_xlim(float(pts[:, 0].min()) - pad, float(pts[:, 0].max()) + pad)
# ax.set_ylim(float(pts[:, 1].min()) - pad, float(pts[:, 1].max()) + pad)
scale = 0.8
ax.set_xlim(-scale, scale)
ax.set_ylim(-scale, scale)
ax.set_aspect("equal", adjustable="box")
if np.isfinite(evr1):
    ax.set_xlabel(f"PC1  (≈ {evr1:.2f})")
    ax.set_ylabel(f"PC2  (≈ {evr2:.2f})")
    ax.set_title(
        f"Embeddings 2D — {REPO} \nPCA-2 on {PCA_SUBSET_TOKENS}",
    )
else:
    ax.set_xlabel("u1  (random)")
    ax.set_ylabel("u2  (random ⊥ u1)")
    ax.set_title(
        f"Embeddings 2D — {REPO} \nrandom 2D slice (debug)",
    )
plt.tight_layout(pad=0.35)
plt.show()


#### Cell: 3 ####
norms = np.linalg.norm(E, axis=1)
k = min(20, int(norms.size))
top_tids = np.argsort(-norms)[:k]
print(f"[top {k} rows by embedding L2 norm]", flush=True)
for rank, tid in enumerate(top_tids, start=1):
    tid = int(tid)
    piece = tokenizer.convert_ids_to_tokens(tid)
    text = tokenizer.decode([tid])
    print(
        f"{rank:2d}  id={tid:6d}  norm={float(norms[tid]):.6g}  "
        f"decode={text!r}  piece={piece!r}",
        flush=True,
    )