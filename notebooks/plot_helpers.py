import numpy as np
import matplotlib.pyplot as plt

import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.offline as pyo

from cycler import cycler


def apply_plot_style():
    """Sets a clean publication-ready style: no grid, inward ticks, tight layout."""

    plt.style.use("seaborn-v0_8-white")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "figure.autolayout": True,
            "savefig.bbox": "tight",
            "axes.grid": False,
            "axes.edgecolor": "black",
            "axes.linewidth": 1.0,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.top": True,
            "ytick.right": True,
            "legend.fontsize": 9,
            "legend.frameon": True,
            "legend.edgecolor": "black",
            "legend.fancybox": False,
            "figure.dpi": 100,
            "figure.figsize": (5, 4),
        }
    )


def create_triangle_surface(
    p1, p2, p3, facecolor="rgba(173,216,230,0.5)", linecolor="black"
):
    x = [p1[0], p2[0], p3[0]]
    y = [p1[1], p2[1], p3[1]]
    z = [p1[2], p2[2], p3[2]]

    mesh = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=[0, 0],
        j=[1, 2],
        k=[2, 1],
        color=facecolor,
        opacity=0.2,
        flatshading=True,
        showscale=False,
    )

    # Edges
    edges = [(p1, p2), (p2, p3), (p3, p1)]
    lines = []
    for a, b in edges:
        lines.append(
            go.Scatter3d(
                x=[a[0], b[0]],
                y=[a[1], b[1]],
                z=[a[2], b[2]],
                mode="lines",
                line=dict(color=linecolor, width=2, dash="dash"),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    return [mesh] + lines


def visualize_grid(grid):
    vertices = grid.vertices
    elements = grid.elements

    fig = ff.create_trisurf(
        x=vertices[0, :],
        y=vertices[1, :],
        z=vertices[2, :],
        simplices=elements.T,
        color_func=elements.shape[1] * ["lightblue"],
    )

    fig.update_layout(
        scene=dict(aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z")
    )

    fig.data = tuple(trace for trace in fig.data if not isinstance(trace, go.Scatter3d))
    pyo.iplot(fig)


def add_convergence_triangle(
    ax, x_ref, y_ref, p, size_x=0.5, text_offset=0.2, shift_y=3
):
    x0 = x_ref
    x1 = x_ref * size_x
    y0 = y_ref * shift_y
    y1 = y0 * size_x**p

    ax.plot(
        [x0, x1, x1, x0], [y0, y0, y1, y0], color="black", linestyle="--", marker="None"
    )
    ax.text(x1 * (size_x**0.5), y1 * (1.0 + text_offset), f"${p}$", fontsize=10)


def plot_convergence_curves(ax, epsilons, **error_curves):
    color_cycle = [
        "#ff0000",
        "#0042E1",
        "#ff7f00",
        "#8900ff",
        "#4daf4a",
        "#ffff33",
        "#a65628",
    ]
    linestyle_cycle = ["--", "--", "--", "--", "--", "--", "--"]
    marker_cycle = ["^", "s", "D", "o", "v", "x", "+"]

    custom_cycler = (
        cycler("color", color_cycle)
        + cycler("linestyle", linestyle_cycle)
        + cycler("marker", marker_cycle)
    )

    ax.set_prop_cycle(custom_cycler)

    for label, data in error_curves.items():
        ax.loglog(epsilons, data, label=label, markersize=5)

    ax.set_xlabel(r"$\varepsilon$", fontsize=12)
    ax.set_title(r"Absolute Error ($L^\infty$)", fontsize=12)
    ax.legend()


def plot_scattered_fields(ax, t, epsilon, **fields):
    color_cycle = [
        "#e41a1c",
        "#377eb8",
        "#ff7f00",
        "#8900ff",
        "#4daf4a",
        "#ffff33",
        "#ff00ff",
    ]
    linestyle_cycle = ["-", "--", "-.", ":", "-", "--", "-."]
    custom_cycler = cycler("color", color_cycle) + cycler("linestyle", linestyle_cycle)

    ax.set_prop_cycle(custom_cycler)

    for label, data in fields.items():
        ax.plot(t, np.real(data), label=label)

    ax.set_xlabel(r"$t$", fontsize=12)
    ax.set_title(
        rf"Time evolution of the scattered field computed for $\varepsilon$={epsilon}",
        fontsize=12,
    )
    ax.legend()


def animate_matrices(
    ax, matrices, xmin, xmax, ymin, ymax, _vmin=None, _vmax=None, title=""
):

    im = ax.imshow(
        np.real(matrices[0].T),
        cmap="viridis",
        extent=[xmin, xmax, ymin, ymax],
        vmin=_vmin,
        vmax=_vmax,
        origin="lower",
    )

    ax.set_title(title)

    def update(frame):
        im.set_array(np.real(matrices[frame].T))
        return [im]

    return im, update
