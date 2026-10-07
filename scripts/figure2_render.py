"""Figure 2 panel layout, curve styles, reward arrows and event marks."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.ticker import LogLocator, MaxNLocator, MultipleLocator, NullLocator
from matplotlib.transforms import Affine2D, ScaledTranslation
from matplotlib.legend_handler import HandlerPatch
from PIL import Image

def scale_episode_axis(x):
    return np.asarray(x) / 1e4

IEEE_COLUMN_WIDTH_IN = 252 / 72.27

IEEE_FIG_HEIGHT_IN = 3.3

IEEE_STYLES = {
    "unsafe": dict(color="#aa3377", lw=0.9, ls=(0, (1.0, 1.3))),
    "unified": dict(color="#009e73", lw=0.8, ls=(0, (4.0, 1.3, 1.0, 1.3))),
    "penalty": dict(color="#0072b2", lw=0.8, ls=(0, (3.2, 1.3))),
    "proposed": dict(color="#d55e00", lw=1.2, ls="-"),
}

IEEE_ELLIPSE_PT = (10.0, 14.0)

IEEE_ELLIPSE_CLEARANCE_PT = 1.5

IEEE_RC = {
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{mathptmx}",  # Times, as in the IEEEtran body text
    "font.family": "serif",
    "font.serif": ["Times"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.5,
    "axes.edgecolor": "#262626",
    "axes.axisbelow": True,
    "axes.titlepad": 2.5,
    "axes.labelpad": 2.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 2.2,
    "ytick.major.size": 2.2,
    "xtick.minor.size": 1.3,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4,
    "xtick.major.pad": 2.0,
    "ytick.major.pad": 1.6,
    "grid.color": "#e3e3e3",
    "grid.linewidth": 0.35,
    "lines.solid_capstyle": "round",
    "lines.dash_capstyle": "butt",
    "legend.frameon": False,
}

class EllipseLegendHandler(HandlerPatch):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        h = 1.2 * height
        w = h * IEEE_ELLIPSE_PT[0] / IEEE_ELLIPSE_PT[1]
        patch = Ellipse((0.5 * width - xdescent, 0.5 * height - ydescent), w, h, fill=False)
        self.update_prop(patch, orig_handle, legend)
        patch.set_transform(trans)
        return [patch]

METRICS = [("global_cost", r"Global cost $C_g$"), ("local_cost", r"Local cost $C_l$")]

REWARD = ("reward", r"Reward $R$")

LEGEND_BAND_PT = 25.0

BAND_ALPHA = 0.16

LOG_FLOOR = 1e-3

SYMLOG_LINTHRESH = 1.0

SYMLOG_LINSCALE = 0.5

YLIM_FROM_EPISODE = 300

MPC_LINE_STYLE = dict(color="#222222", lw=0.8, ls=(0, (4.0, 2.0)))

MPC_LABEL = "Distributed MPC (theoretical optimum)"

SOFT_STYLES = {
    "unsafe": dict(color="#a3abb0", lw=0.9, ls="-"),
    "unified": dict(color="#4fa477", lw=0.85, ls=(0, (4.0, 1.3, 1.0, 1.3))),
    "penalty": dict(color="#4f7fbf", lw=0.85, ls=(0, (3.2, 1.3))),
    "proposed": dict(color="#c44e52", lw=1.25, ls="-", marker="o", markersize=2.4,
                     markerfacecolor="white", markeredgewidth=0.7),
}

PALETTES = {"paper": (IEEE_STYLES, "_paper"), "soft": (SOFT_STYLES, "")}

EVENT_STYLES = {
    "conflict": ("Safety conflict", dict(edgecolor="#2f7f57", lw=0.8, ls=(0, (2.2, 1.4)))),
    "violation": ("Severe violation", dict(edgecolor="#6b7378", lw=0.8, ls=(0, (2.2, 1.4)))),
}

GAP_ARROW_COLOR = "#444444"

def mark_event(fig, ax, x, y, kind):
    """Fixed-size ellipse (sized in points) centered on the data point (x, y)."""
    to_display = Affine2D().scale(1 / 72) + fig.dpi_scale_trans + ScaledTranslation(x, y, ax.transData)
    w, h = IEEE_ELLIPSE_PT
    ax.add_patch(Ellipse((0, 0), w, h, fill=False, transform=to_display, zorder=10, **EVENT_STYLES[kind][1]))


def limit_values(values):
    """Use the post-transient curve when available, otherwise the full short run."""
    return values[YLIM_FROM_EPISODE:] if len(values) > YLIM_FROM_EPISODE else values


def plot(stats, out, palette="soft", yscale="symlog", gap_vs=None, mpc_refs=None, events=(), use_tex=False, xlim=None):
    """Render the current paper layout using supplied seed statistics and MPC references."""
    scenarios = tuple(stats)
    styles = {method: style for method, style in PALETTES[palette][0].items() if method in stats[scenarios[0]]}
    legend_order = [method for method in ("proposed", "penalty", "unified", "unsafe") if method in styles]
    labels = dict(proposed="Proposed", penalty="MAPPO-PF", unified="MAPPO-Lag", unsafe="MAPPO-Vanilla")
    metrics = [REWARD] + METRICS
    height = IEEE_FIG_HEIGHT_IN if len(scenarios) == 3 else 1.25 * len(scenarios) + 0.5
    mpc_line = mpc_refs is not None
    band = LEGEND_BAND_PT if (events or mpc_line) else 13.0
    max_episodes = max(len(values["reward"][0]) for methods in stats.values() for values in methods.values())
    xlim = xlim or (0.0, max((max_episodes - 1) / 1e4, 1e-4))

    with plt.rc_context(dict(IEEE_RC, **{"text.usetex": use_tex, "font.serif": ["Times"] if use_tex else ["Liberation Serif", "DejaVu Serif"]})):
        fig = plt.figure(figsize=(IEEE_COLUMN_WIDTH_IN, height), layout="constrained")
        fig.get_layout_engine().set(
            w_pad=1.5 / 72, h_pad=1.5 / 72, wspace=0.02, hspace=0.0,
            rect=(0, 0, 1, 1 - band / (72 * height)),
        )
        axes = fig.subplots(len(scenarios), len(metrics), sharex=True, squeeze=False)

        for r, scenario in enumerate(scenarios):
            for c, (metric, title) in enumerate(metrics):
                ax = axes[r, c]
                for method, style in styles.items():
                    mean, lo, hi = stats[scenario][method][metric]
                    x = scale_episode_axis(np.arange(len(mean)))
                    step = 10 if len(mean) > YLIM_FROM_EPISODE else 1
                    ax.fill_between(x[::step], lo[::step], hi[::step], color=style["color"],
                                    alpha=BAND_ALPHA, linewidth=0, zorder=1)
                    kw = dict(style)
                    if "marker" in kw:
                        kw["markevery"] = (125, 250) if len(mean) >= 2500 else max(1, len(mean) // (10 * step))
                    if len(mean) == 1:
                        kw.setdefault("marker", "o")
                        kw.pop("markevery", None)
                    ax.plot(x[::step], mean[::step], zorder=3, **kw)
                if metric == "reward":
                    # linear, same rule as the single-seed figure's reward column
                    centers = [limit_values(stats[scenario][m][metric][0]) for m in styles]
                    lo, hi = min(c_.min() for c_ in centers), max(c_.max() for c_ in centers)
                    if gap_vs:
                        # double arrow at the right end: mean of the last 2,000 episodes,
                        # labelled with the lead in percent of the baseline's |reward|
                        top = stats[scenario]["proposed"][metric][0][-2000:].mean()
                        bottom = stats[scenario][gap_vs][metric][0][-2000:].mean()
                        # drawn in the open part of the late, flat curves, between Proposed's markers
                        x_arrow = xlim[0] + 0.75 * (xlim[1] - xlim[0])
                        ax.annotate("", xy=(x_arrow, top), xytext=(x_arrow, bottom), zorder=6,
                                    arrowprops=dict(arrowstyle="<->", color=GAP_ARROW_COLOR, lw=0.7,
                                                    shrinkA=0, shrinkB=0, mutation_scale=5))
                        label = dict(fontsize=7, color=styles["proposed"]["color"], zorder=6,
                                     bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8))
                        span = max(hi, top) - min(lo, bottom)
                        # relative lead over the baseline; |.| because C3 rewards are negative
                        text = (f"$+{(top - bottom) / abs(bottom) * 100:.0f}\\%$" if bottom != 0
                                else f"${top - bottom:+.2g}$")
                        if (top - bottom) < 0.4 * span:  # small gap: label under the lower curve
                            ax.text(x_arrow, bottom - 0.07 * span, text, ha="center", va="top", **label)
                        else:  # wide gap: label left of the arrow, between the curves, inside the panel
                            ax.text(x_arrow - 0.06, (top + bottom) / 2, text, ha="right", va="center", **label)
                    if mpc_line:
                        ref = mpc_refs[scenario]
                        ax.axhline(ref, zorder=2, **MPC_LINE_STYLE)
                        lo, hi = min(lo, ref), max(hi, ref)
                    span = hi - lo if hi > lo else max(abs(lo), 1.0)
                    ax.set_ylim(lo - 0.06 * span, hi + 0.08 * span)
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10], min_n_ticks=3))
                elif yscale == "symlog":
                    ax.set_yscale("symlog", linthresh=SYMLOG_LINTHRESH, linscale=SYMLOG_LINSCALE)
                    top = max(limit_values(stats[scenario][m][metric][0]).max() for m in styles)
                    decades = int(np.ceil(np.log10(max(top, 1.0) * 1.6)))
                    ax.set_ylim(-0.04, 10 ** decades)
                    ticks = [0.0] + [10.0 ** k for k in range(0, decades + 1)]
                    ax.set_yticks(ticks)
                    ax.set_yticklabels(["$0$"] + [f"$10^{{{k}}}$" for k in range(0, decades + 1)])
                    ax.yaxis.set_minor_locator(NullLocator())
                elif yscale == "log":
                    ax.set_yscale("log")
                    # a little room under the floor keeps curves at 1e-3 off the bottom spine
                    ax.set_ylim(bottom=LOG_FLOOR / 10 ** 0.15)
                    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
                    ax.yaxis.set_minor_locator(NullLocator())
                else:
                    # Same rule as the single-seed figure, applied to the mean curves.
                    top = max(limit_values(stats[scenario][m][metric][0]).max() for m in styles)
                    top = max(top, 1e-3)
                    ax.set_ylim(-0.05 * top, 1.08 * top)
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10], min_n_ticks=3))
                ax.xaxis.set_major_locator(MultipleLocator(1.0) if xlim[1] >= 1.0 else MaxNLocator(nbins=3))
                ax.xaxis.set_minor_locator(MultipleLocator(0.5) if xlim[1] >= 1.0 else NullLocator())
                ax.set_xlim(*xlim)
                ax.grid(True, which="major")
                if r == 0:
                    ax.set_title(title)
                if c == 0:
                    ax.set_ylabel(scenario)

        marked = {}
        for scenario, metric, method, episode, kind in events:
            r = scenarios.index(scenario)
            c = [m for m, _ in metrics].index(metric)
            y = stats[scenario][method][metric][0]
            if episode >= len(y):
                continue
            lo_i, hi_i = max(episode - 800, 0), min(episode + 800, len(y))
            i = lo_i + int(np.argmax(y[lo_i:hi_i]))
            mark_event(fig, axes[r, c], scale_episode_axis(i), y[i], kind)
            marked[(r, c)] = max(marked.get((r, c), -np.inf), y[i])

        fig.supxlabel(r"Episode ($\times 10^4$)", fontsize=8)

        # Widen the y-range just enough that each ellipse stays inside its panel.
        fig.draw_without_rendering()
        for (r, c), y_event in marked.items():
            ax = axes[r, c]
            if ax.get_yscale() == "symlog":
                continue  # the marked spikes sit well inside these panels
            bottom, top = ax.get_ylim()
            axes_height_pt = ax.get_window_extent().height * 72 / fig.dpi
            frac = (IEEE_ELLIPSE_PT[1] / 2 + IEEE_ELLIPSE_CLEARANCE_PT) / axes_height_pt
            if ax.get_yscale() == "log":
                need = 10 ** ((np.log10(y_event) - frac * np.log10(bottom)) / (1 - frac))
            else:
                need = (y_event - frac * bottom) / (1 - frac)
            ax.set_ylim(bottom, max(top, need))

        handles = [Line2D([], [], **styles[m]) for m in legend_order]
        long_labels = sum(len(v) for v in labels.values()) > 40  # keep the method legend compact
        fig.legend(handles, [labels[m] for m in legend_order],
                   loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4,
                   handlelength=1.0 if long_labels else 1.3, handletextpad=0.3 if long_labels else 0.35,
                   columnspacing=0.45 if long_labels else 0.85, borderaxespad=0.15, borderpad=0.15)
        second_row, second_labels = [], []
        if mpc_line:
            second_row.append(Line2D([], [], **MPC_LINE_STYLE))
            second_labels.append(MPC_LABEL)
        if marked:
            used = [k for k in EVENT_STYLES if any(e[4] == k for e in events)]
            second_row += [Ellipse((0, 0), 1, 1, fill=False, **EVENT_STYLES[k][1]) for k in used]
            second_labels += [EVENT_STYLES[k][0] for k in used]
        if second_row:
            fig.legend(second_row, second_labels,
                       loc="upper center", bbox_to_anchor=(0.5, 1.0 - 11.5 / (72 * height)),
                       ncol=len(second_row), handlelength=1.3, handletextpad=0.35, columnspacing=1.6,
                       borderaxespad=0.15, borderpad=0.15, handler_map={Ellipse: EllipseLegendHandler()})

        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out.with_suffix(".pdf"))
        fig.savefig(out.with_suffix(".png"), dpi=600)
        plt.close(fig)

    # IEEE's graphics checker expects RGB raster files without an alpha channel.
    png_path = out.with_suffix(".png")
    with Image.open(png_path) as im:
        im.convert("RGB").save(png_path, dpi=(600, 600))
    return stats
