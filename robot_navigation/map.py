"""
    Fixed named wall layouts and cell-probability heat maps.
"""
import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle


# Rows are written top to bottom for readability. '#' denotes an occupied unit
# square. Their coordinate conversion below reverses rows to increasing y.
_TEMPLATES = (
    ("staggered walls", (
        "..#...",
        "..#...",
        "###.#.",
        "....#.",
        ".####.",
        "......",
    )),
    ("slalom corridor", (
        ".####.",
        "....#.",
        ".#..#.",
        ".#....",
        ".####.",
        "......",
    )),
    ("open pocket", (
        ".####.",
        ".#..#.",
        ".#..#.",
        ".#....",
        ".###..",
        "......",
    )),
    ("offset doorways", (
        "###.##",
        "......",
        ".####.",
        ".#....",
        ".#..#.",
        "...##.",
    )),
    ("fork and ledge", (
        "..#...",
        ".###..",
        "...#..",
        ".#.#..",
        ".#....",
        ".####.",
    )),
    ("opposing bends", (
        "###...",
        "..#.#.",
        "..#.#.",
        "....#.",
        ".####.",
        "......",
    )),
    ("broken rooms", (
        ".##.##",
        ".#..#.",
        ".#....",
        ".####.",
        "....#.",
        "......",
    )),
    ("stepped baffles", (
        "###...",
        "..#...",
        "..###.",
        "....#.",
        ".####.",
        "......",
    )),
    ("winding passage", (
        ".####.",
        ".#....",
        ".#.##.",
        "...#..",
        "##.#..",
        "......",
    )),
)


def make_wall_maps(*, dtype=torch.float64, device="cpu"):
    """Return cell centres [36,2], occupied masks [9,6,6], and map labels.

    Cells use y-major order: x varies fastest, from -2.5 to 2.5. Each of the
    nine templates provides one map, without rotations. The returned masks
    cover only the interior; the plotting function adds the outer cells
    (centres with x or y equal to +/-3.5) without wall outlines.
    """
    centers = torch.tensor(
        [(x, y) 
            for y in (-2.5, -1.5, -.5, .5, 1.5, 2.5)
            for x in (-2.5, -1.5, -.5, .5, 1.5, 2.5)
        ],
        dtype=dtype, device=device,
    )
    layouts, labels = [], []
    for name, rows in _TEMPLATES:
        if len(rows) != 6 or any(len(row) != 6 or set(row) - {'.', '#'} for row in rows):
            raise ValueError("Every map template must be six rows of six '.'/'#' characters")

        base = torch.tensor(
            [[c == '#' for c in row] for row in reversed(rows)],
            dtype=torch.bool, device=device
        )
        
        layouts.append(base)
        labels.append(f"{name}")
    maps = torch.stack(layouts)
    return centers, maps, tuple(labels)

def _save(fig, save_path):
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")

def plot_wall_map_catalog(
    save_path=None,
    *,
    layout_name,
    start_point,
    target_point,
    cell_probabilities=None,
    cmap="YlOrRd",
    wall_linewidth=2.5,
):
    """
    Plot a fixed wall layout, cell occupancy probabilities, start and target.

    Parameters
    ----------
    layout_name : str
        The name of a fixed layout in ``_TEMPLATES`` (case-insensitive).
        This selects the true wall cells, drawn with thick black outlines.
    save_path : str or pathlib.Path, optional
        Save the figure here as well as returning it.
    start_point, target_point : array-like or torch.Tensor, shape (2,)
        Coordinates (y_1, y_2) in the outer ring of the 8 x 8 grid.
        Both coordinates must lie in [-4, 4], and at least one must have
        absolute value >= 3. Cell boundaries are included. Raise ValueError
        for points outside this ring, malformed points, or nonfinite values.
    cell_probabilities : array-like or torch.Tensor, optional
        Wall occupancy probabilities with shape (8, 8) or (64,). Row zero
        is the bottom row (y_2 = -3.5); y_1 varies fastest when flattened.
        Each entry lies in [0, 1]; entries need not sum to one.
        If omitted, assign 0.5 to every interior cell. The outer cells are
        known to have no walls, so their probabilities are always fixed at 0,
        overriding any supplied outer-cell values without modifying the input.
        For compatibility, an interior
        array with shape (6, 6) or (36,) is also accepted: its bottom row has
        y_2 = -2.5, and the outer ring is filled with 0.
    cmap : str or matplotlib.colors.Colormap
        Base heat-map colour map, softened by blending 35% white into its
        colours. The colour scale is always fixed at [0, 1].
    wall_linewidth : float
        Width of true wall cell boundaries in points (default 2.5).

    Returns
    -------
    matplotlib.figure.Figure
        The figure. Call ``plt.show()`` to display it in a script.

    Examples
    --------
    >>> fig = plot_wall_map_catalog(
    ...     "staggered walls", start_point=(-3.5, -3.5), target_point=(3.5, 3.5))
    >>> fig = plot_wall_map_catalog(
    ...     "open pocket", start_point=(-3.5, 0), target_point=(3.5, 0),
    ...     cell_probabilities=p)

    All 64 cells in [-4, 4] x [-4, 4] are shown. The named true map determines
    the wall outlines in the interior 6 x 6 cells only; it does not determine
    the interior probabilities. The outer ring is known to be wall-free.
    """

    templates = dict(_TEMPLATES)
    labels = tuple(templates)
    if not isinstance(layout_name, str) or layout_name.strip().casefold() not in templates:
        raise ValueError(f"Unknown wall map {layout_name!r}. Choose from: {', '.join(labels)}")
    layout_name = layout_name.strip().casefold()
    # Convert only the selected fixed layout to coordinates increasing in y.
    walls = np.array(
        [[cell == '#' for cell in row] for row in reversed(templates[layout_name])],
        dtype=bool,
    )

    if not np.isfinite(wall_linewidth) or wall_linewidth <= 0:
        raise ValueError("wall_linewidth must be a positive finite number")

    def as_array(values):
        # Detach tensors and transfer to CPU so GPU/training outputs also work.
        if torch.is_tensor(values):
            values = values.detach().to(device="cpu", dtype=torch.float64).numpy()
        return np.asarray(values, dtype=float)

    def outer_point(values, argument):
        try:
            point = as_array(values)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{argument} must be a finite coordinate pair (y_1, y_2)") from error
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError(f"{argument} must be a finite coordinate pair (y_1, y_2)")
        if np.any(np.abs(point) > 4) or np.all(np.abs(point) < 3):
            raise ValueError(
                f"{argument} must lie in the outer ring: both coordinates in "
                "[-4, 4] and at least one with absolute value >= 3"
            )
        return point

    start = outer_point(start_point, "start_point")
    target = outer_point(target_point, "target_point")

    if cell_probabilities is None:
        probabilities = np.full((8, 8), 0.5)
    else:
        probabilities = as_array(cell_probabilities)
        if probabilities.shape not in ((8, 8), (64,), (6, 6), (36,)):
            raise ValueError(
                "cell_probabilities must have shape (8, 8) or (64,), "
                "or interior shape (6, 6) or (36,)"
            )
        if not np.all(np.isfinite(probabilities)) or np.any(
            (probabilities < 0) | (probabilities > 1)
        ):
            raise ValueError("cell_probabilities must contain finite probabilities in [0, 1]")
        if probabilities.size == 36:
            interior = probabilities.reshape(6, 6)
            probabilities = np.zeros((8, 8))
            probabilities[1:-1, 1:-1] = interior
        else:
            probabilities = probabilities.reshape(8, 8).copy()

    # The outer cells are known wall-free, regardless of the interior belief.
    probabilities[[0, -1], :] = 0.0
    probabilities[:, [0, -1]] = 0.0

    # Soften the palette itself so the heat map and colour bar match.
    base_cmap = plt.get_cmap(cmap)
    colours = base_cmap(np.linspace(0, 1, 256))
    colours[:, :3] = 0.65 * colours[:, :3] + 0.35
    pale_cmap = ListedColormap(colours, name=f"{base_cmap.name}_pale")

    fig, ax = plt.subplots(figsize=(8, 7.5), constrained_layout=True)
    image = ax.imshow(
        probabilities,
        origin="lower",
        extent=(-4, 4, -4, 4),
        interpolation="nearest",
        cmap=pale_cmap,
        vmin=0,
        vmax=1,
        zorder=0,
    )
    ax.set(
        xlim=(-4, 4), ylim=(-4, 4),
        xticks=np.arange(-4, 5), yticks=np.arange(-4, 5),
        xlabel=r"$y_1$", ylabel=r"$y_2$", title=layout_name,
    )
    ax.set_aspect("equal")
    ax.set_axisbelow(False)
    ax.grid(True, color="0.75", linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Outline every occupied unit cell, including boundaries shared by walls.
    for row, col in np.argwhere(walls):
        ax.add_patch(Rectangle(
            (col - 3, row - 3), 1, 1,
            fill=False, edgecolor="black", linewidth=wall_linewidth,
            zorder=3, clip_on=False,
        ))

    for row in range(8):
        for col in range(8):
            probability = probabilities[row, col]
            red, green, blue, _ = image.cmap(image.norm(probability))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            ax.text(
                col - 3.5, row - 3.8, f"{probability:.2f}",
                ha="center", va="center", fontsize=9,
                color="black" if luminance > 0.5 else "white", zorder=4,
            )

    start_marker = ax.scatter(
        start[0], start[1], marker="o", s=100, color="royalblue",
        edgecolors="white", linewidths=1.2, zorder=5, clip_on=False,
        label="Starting point",
    )
    target_marker = ax.scatter(
        target[0], target[1], marker="*", s=220, color="limegreen",
        edgecolors="black", linewidths=0.8, zorder=6, clip_on=False,
        label="Target point",
    )

    fig.colorbar(image, ax=ax, shrink=0.88, pad=0.04, label="P(wall in cell)")
    ax.legend(
        handles=[Patch(facecolor="none", edgecolor="black",
                       linewidth=wall_linewidth, label="True wall cell"),
                 start_marker, target_marker],
        loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False, ncol=3,
    )
    _save(fig, save_path)
    return fig
