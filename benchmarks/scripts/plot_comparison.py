"""
Generate a multi-panel benchmark visualization comparing
Context Stuffing vs Archivist Memory System across corpus scales.
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.ticker import FuncFormatter

# ── Load data ────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).parent.parent.parent
data_path = ROOT / ".benchmarks" / "full_comparison.json"
if not data_path.exists():
    print(f"ERROR: {data_path} not found. Run run_full_comparison.sh first.", file=sys.stderr)
    sys.exit(1)

data = json.loads(data_path.read_text())
stuffing = {s["memory_scale"]: s for s in data["stuffing_summaries"]}
archivist_full = {
    s["memory_scale"]: s
    for s in data["archivist_summaries"]
    if s["variant"] == "full_pipeline"
}

SCALES      = ["small", "medium", "large", "stress"]
SCALE_LABELS = ["Small\n56 files", "Medium\n253 files", "Large\n543 files", "Stress\n1,523 files"]
FILE_COUNTS  = [56, 253, 543, 1523]
CORPUS_TOK   = [8_681, 24_262, 44_351, 108_847]
OVERFLOW_IDX = 2  # large + stress overflow (index 2 and 3)

# ── Theme ────────────────────────────────────────────────────────────────────
BG       = "#0d1117"
PANEL    = "#161b22"
GRID     = "#21262d"
STUFFING = "#f97316"   # orange
ARCH     = "#22d3ee"   # cyan
ARCH2    = "#a78bfa"   # purple (vector_only secondary)
OVERFLOW = "#ef444430" # red transparent
WHITE    = "#f0f6fc"
MUTED    = "#8b949e"
GREEN    = "#4ade80"
YELLOW   = "#fbbf24"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    GRID,
    "axes.labelcolor":   WHITE,
    "axes.titlecolor":   WHITE,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "grid.color":        GRID,
    "text.color":        WHITE,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
})

fig = plt.figure(figsize=(18, 13), facecolor=BG)
fig.suptitle(
    "Archivist Memory System  vs  Context Stuffing (MD Files)",
    fontsize=20, fontweight="bold", color=WHITE, y=0.97,
)
fig.text(
    0.5, 0.935,
    "Real LLM evaluation · 32,768-token context budget · qwen3.5-122b · BAAI/bge-base-en-v1.5",
    ha="center", fontsize=11, color=MUTED,
)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38,
                       left=0.07, right=0.97, top=0.91, bottom=0.07)

x = np.arange(len(SCALES))
bar_w = 0.36


def add_overflow_shade(ax, xlim=None):
    """Shade the overflow region (large + stress)."""
    xmin, xmax = (xlim or ax.get_xlim())
    # shade right half (large + stress) where stuffing overflows
    ax.axvspan(1.5, xmax, color="#ef4444", alpha=0.06, zorder=0, label="_nolegend_")
    ax.axvline(1.5, color="#ef4444", alpha=0.4, lw=1.2, ls="--", zorder=1)


def style_ax(ax, title, xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    ax.set_title(title, fontsize=12, fontweight="bold", color=WHITE, pad=10)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    ax.grid(axis="x", visible=False)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)


# ════════════════════════════════════════════════════════════════════════════
# Panel 1 (top-left, wide): Recall@5 across scales
# ════════════════════════════════════════════════════════════════════════════
ax1 = fig.add_subplot(gs[0, :2])

recall_stuffing = [stuffing[s]["recall"] for s in SCALES]
recall_arch     = [archivist_full[s]["recall_at_5"] for s in SCALES]

bars_s = ax1.bar(x - bar_w/2, recall_stuffing, bar_w,
                 color=STUFFING, alpha=0.9, label="Context Stuffing (MD files)", zorder=3)
bars_a = ax1.bar(x + bar_w/2, recall_arch,     bar_w,
                 color=ARCH,     alpha=0.9, label="Archivist full_pipeline",    zorder=3)

# Annotate bars
for bar, val in zip(bars_s, recall_stuffing):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
             f"{val:.3f}", ha="center", va="bottom", fontsize=8.5, color=STUFFING, fontweight="bold")
for bar, val in zip(bars_a, recall_arch):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.004,
             f"{val:.3f}", ha="center", va="bottom", fontsize=8.5, color=ARCH, fontweight="bold")

# Overflow labels
for i in range(OVERFLOW_IDX, len(SCALES)):
    ax1.text(x[i], 0.73, "⚠ OVERFLOW", ha="center", fontsize=7.5, color="#ef4444",
             fontweight="bold", zorder=5,
             bbox=dict(boxstyle="round,pad=0.2", facecolor="#1c1010", edgecolor="#ef4444", linewidth=0.8))

add_overflow_shade(ax1, xlim=(-0.6, len(SCALES)-0.4))
ax1.set_xlim(-0.6, len(SCALES)-0.4)
ax1.set_ylim(0.70, 1.02)
ax1.set_xticks(x)
ax1.set_xticklabels(SCALE_LABELS, fontsize=9)
ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))

legend = ax1.legend(loc="lower right", framealpha=0.25, facecolor=PANEL,
                    edgecolor=GRID, fontsize=9)
for text in legend.get_texts():
    text.set_color(WHITE)

# Delta annotations
for i, (s, a) in enumerate(zip(recall_stuffing, recall_arch)):
    delta = a - s
    col = GREEN if delta >= 0 else "#f87171"
    sym = "▲" if delta >= 0 else "▼"
    ax1.annotate(f"{sym}{abs(delta):.3f}", xy=(x[i], max(s, a) + 0.018),
                 ha="center", fontsize=8, color=col, fontweight="bold")

style_ax(ax1, "Recall@5: Does the right memory surface?", ylabel="Recall@5")

overflow_patch = mpatches.Patch(color="#ef4444", alpha=0.3, label="Context overflow zone")
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles + [overflow_patch], labels + ["Context overflow zone"],
           loc="lower right", framealpha=0.25, facecolor=PANEL,
           edgecolor=GRID, fontsize=9, labelcolor=WHITE)


# ════════════════════════════════════════════════════════════════════════════
# Panel 2 (top-right): Token cost per query
# ════════════════════════════════════════════════════════════════════════════
ax2 = fig.add_subplot(gs[0, 2])

tok_stuffing = CORPUS_TOK  # stuffing sends the whole corpus each query
tok_arch     = [archivist_full[s]["avg_tokens_per_query"] for s in SCALES]

ax2.plot(x, tok_stuffing, "o-", color=STUFFING, lw=2.5, ms=7, zorder=3,
         label="Context Stuffing")
ax2.plot(x, tok_arch,     "o-", color=ARCH,     lw=2.5, ms=7, zorder=3,
         label="Archivist")

# Fill between
ax2.fill_between(x, tok_arch, tok_stuffing, alpha=0.12, color=ARCH, zorder=2)

# Annotate stress difference
ax2.annotate(
    f"20× cheaper\nat stress scale",
    xy=(3, tok_arch[-1]), xytext=(2.2, 50000),
    fontsize=8.5, color=ARCH, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=ARCH, lw=1.2),
)

add_overflow_shade(ax2, xlim=(-0.2, 3.3))
ax2.set_xlim(-0.2, 3.3)
ax2.set_xticks(x)
ax2.set_xticklabels(SCALE_LABELS, fontsize=8)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
ax2.set_ylim(0, 120_000)
legend = ax2.legend(loc="upper left", framealpha=0.25, facecolor=PANEL,
                    edgecolor=GRID, fontsize=8.5, labelcolor=WHITE)
style_ax(ax2, "Tokens Per Query", ylabel="Tokens (k)")


# ════════════════════════════════════════════════════════════════════════════
# Panel 3 (bottom-left): Latency p50 comparison
# ════════════════════════════════════════════════════════════════════════════
ax3 = fig.add_subplot(gs[1, 0])

lat_stuffing = [stuffing[s]["latency_p50_ms"] for s in SCALES]
lat_arch     = [archivist_full[s]["latency_p50_ms"] for s in SCALES]

ax3.bar(x - bar_w/2, lat_stuffing, bar_w, color=STUFFING, alpha=0.9,
        label="Context Stuffing", zorder=3)
ax3.bar(x + bar_w/2, lat_arch,     bar_w, color=ARCH,     alpha=0.9,
        label="Archivist",        zorder=3)

for bar, val in zip(ax3.patches[:4], lat_stuffing):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             f"{val/1000:.1f}s", ha="center", va="bottom", fontsize=7.5, color=STUFFING)
for bar, val in zip(ax3.patches[4:], lat_arch):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             f"{val:.0f}ms", ha="center", va="bottom", fontsize=7.5, color=ARCH)

# 44x annotation at stress
ratio = lat_stuffing[-1] / lat_arch[-1]
ax3.annotate(f"{ratio:.0f}× faster →", xy=(3 + bar_w/2, lat_arch[-1] + 400),
             ha="left", fontsize=8.5, color=GREEN, fontweight="bold")

add_overflow_shade(ax3, xlim=(-0.6, 3.6))
ax3.set_xlim(-0.6, 3.6)
ax3.set_ylim(0, 9000)
ax3.set_xticks(x)
ax3.set_xticklabels(SCALE_LABELS, fontsize=8)
ax3.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:.0f}s"))
legend = ax3.legend(loc="upper left", framealpha=0.25, facecolor=PANEL,
                    edgecolor=GRID, fontsize=8.5, labelcolor=WHITE)
style_ax(ax3, "Query Latency (p50)", ylabel="Latency")


# ════════════════════════════════════════════════════════════════════════════
# Panel 4 (bottom-middle): Per-query-type at stress scale (grouped bars)
# ════════════════════════════════════════════════════════════════════════════
ax4 = fig.add_subplot(gs[1, 1])

QTYPES = ["single_hop", "multi_hop", "broad", "agent_scoped", "temporal", "adversarial"]
QLABELS = ["Single\nHop", "Multi\nHop", "Broad", "Agent\nScoped", "Temporal", "Adversarial"]

s_vals = [stuffing["stress"]["by_query_type"].get(qt, {}).get("recall", 0) for qt in QTYPES]
a_vals = [archivist_full["stress"]["by_query_type"].get(qt, {}).get("recall", 0) for qt in QTYPES]

xq = np.arange(len(QTYPES))
bw = 0.36
bs = ax4.bar(xq - bw/2, s_vals, bw, color=STUFFING, alpha=0.9, label="Stuffing (overflow)")
ba = ax4.bar(xq + bw/2, a_vals, bw, color=ARCH,     alpha=0.9, label="Archivist")

# Delta labels above each group
for i, (sv, av) in enumerate(zip(s_vals, a_vals)):
    delta = av - sv
    col = GREEN if delta > 0 else ("#f87171" if delta < 0 else MUTED)
    sym = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
    ax4.text(xq[i], max(sv, av) + 0.015, f"{sym}{abs(delta):.2f}",
             ha="center", fontsize=7.5, color=col, fontweight="bold")

ax4.set_xlim(-0.5, len(QTYPES) - 0.5)
ax4.set_ylim(0, 1.18)
ax4.set_xticks(xq)
ax4.set_xticklabels(QLABELS, fontsize=8)
ax4.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
legend = ax4.legend(loc="lower right", framealpha=0.25, facecolor=PANEL,
                    edgecolor=GRID, fontsize=8.5, labelcolor=WHITE)
style_ax(ax4, "Recall by Query Type @ Stress", ylabel="Recall@5")


# ════════════════════════════════════════════════════════════════════════════
# Panel 5 (bottom-right): Archivist recall stays stable, stuffing degrades
# ════════════════════════════════════════════════════════════════════════════
ax5 = fig.add_subplot(gs[1, 2])

# Show multi_hop specifically — clearest degradation story
mh_stuff = [stuffing[s]["by_query_type"].get("multi_hop", {}).get("recall", None) for s in SCALES]
mh_arch  = [archivist_full[s]["by_query_type"].get("multi_hop", {}).get("recall", None) for s in SCALES]
sh_stuff = [stuffing[s]["by_query_type"].get("single_hop", {}).get("recall", None) for s in SCALES]
sh_arch  = [archivist_full[s]["by_query_type"].get("single_hop", {}).get("recall", None) for s in SCALES]

ax5.plot(x, mh_stuff, "s--", color=STUFFING, lw=2,   ms=7, alpha=0.8,  label="Stuffing: multi-hop")
ax5.plot(x, mh_arch,  "s-",  color=ARCH,     lw=2.5, ms=7,             label="Archivist: multi-hop")
ax5.plot(x, sh_stuff, "o--", color=YELLOW,   lw=1.5, ms=6, alpha=0.7,  label="Stuffing: single-hop")
ax5.plot(x, sh_arch,  "o-",  color="#86efac", lw=2,  ms=6,             label="Archivist: single-hop")

add_overflow_shade(ax5, xlim=(-0.2, 3.3))
ax5.set_xlim(-0.2, 3.3)
ax5.set_ylim(0.45, 1.10)
ax5.set_xticks(x)
ax5.set_xticklabels(SCALE_LABELS, fontsize=8)
ax5.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
ax5.axvline(1.5, color="#ef4444", alpha=0.4, lw=1.2, ls="--")
legend = ax5.legend(loc="lower left", framealpha=0.25, facecolor=PANEL,
                    edgecolor=GRID, fontsize=7.5, labelcolor=WHITE, ncol=1)
style_ax(ax5, "Recall Degradation by Query Type", ylabel="Recall@5")

# ── Overflow banner ──────────────────────────────────────────────────────────
fig.text(0.655, 0.905, "  ⚠ Context overflow zone (large → stress)  ",
         ha="center", fontsize=9, color="#ef4444",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#200808", edgecolor="#ef4444", linewidth=1))

# ── Save ─────────────────────────────────────────────────────────────────────
out = ROOT / ".benchmarks" / "benchmark_comparison.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved: {out}")
