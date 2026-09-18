"""
file: script.py
author: Chiagozie Okoye
description:
    Sweeps speckle_test (video generation) over TARGET_FPS x KERNEL_SIZE and
    speckle_analysis (multicore LSCI analysis) over NUM_CORES x TARGET_FPS x
    KERNEL_SIZE, capturing each run's stdout performance metrics into pandas
    DataFrames, persisting them as CSV, reporting pass/fail against the
    clinical targets speckle_analysis checks (frame rate, frame process time,
    latency jitter, per-core CPU utilization), and plotting the results.
"""
from pathlib import Path
import itertools
import re
import subprocess

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3D projection)

SHELL_DIR = Path(__file__).parent / "shell"
SPECKLE_TEST = SHELL_DIR / "speckle_test"
COMPILE_SPECKLE_TEST = SHELL_DIR / "compile.speckle.test.sh"
SPECKLE_ANALYSIS = SHELL_DIR / "speckle_analysis"
BUILD_SPECKLE_ANALYSIS = SHELL_DIR / "build.speckle.analysis.sh"

RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

# Constant Parameters
VIDEO_TYPE = 'random'
RESOLUTION = '640'
SOURCE_FPS = 80
EXPOSURE_MS = 3

# Variable (Experimental) Parameters
NUM_CORES = [1, 2, 3, 4, 8, 16, 32, 64]
TARGET_FPS = [40, 60, 80, 128, 256]
KERNEL_SIZE = [x for x in range(5, 16, 2)]  # speckle_test requires a positive odd kernel size

# Baselines used to slice the full-factorial analysis grid into one-factor-at-a-time plots
BASELINE_NUM_CORES = 4
BASELINE_TARGET_FPS = 80
BASELINE_KERNEL_SIZE = 5

# Clinical targets mirrored from speckle_analysis.cpp (Harris et al., 2026 EBPM). There is no
# shared config between the C++ and this script, so these must be kept in sync by hand if the
# thresholds in speckle_analysis.cpp ever change.
TARGET_PROCESSING_TIME_MS = 10.0   # Frame Process Time clinical limit ("<=")
CLINICAL_LATENCY_LIMIT_MS = 2.0    # Latency Jitter clinical limit ("<=")
CPU_UTIL_LIMIT_PCT = 70.0          # Per-Core CPU Util. clinical limit ("<")
# Frame Rate has no fixed constant -- its target is each row's own target_fps value.
# Power Consumption is intentionally excluded: speckle_analysis.cpp reports a hardcoded
# ACTUAL_POWER_W (4.2) that is never actually measured, and its "[ PASS ]" status is printed
# unconditionally, so it carries no real signal for summaries or plots.

# dataviz reference palette (light mode) - see references/palette.md
CHART_SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SERIES_BLUE = "#2a78d6"
SEQUENTIAL_BLUE_STEPS = [
    "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b",
]
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"

# Clinical metrics surfaced by speckle_analysis, shared across summary/OFAT/3D/facet plotting.
METRIC_SPECS = [
    {"slug": "fps", "col": "achieved_fps", "status_col": "fps_status",
     "label": "Achieved FPS", "threshold": None},
    {"slug": "delay", "col": "mean_delay_ms", "status_col": "delay_status",
     "label": "Frame Process Time (ms)", "threshold": TARGET_PROCESSING_TIME_MS},
    {"slug": "jitter", "col": "jitter_std_ms", "status_col": "jitter_status",
     "label": "Latency Jitter (ms)", "threshold": CLINICAL_LATENCY_LIMIT_MS},
    {"slug": "cpu", "col": "per_core_cpu_util", "status_col": "cpu_status",
     "label": "Per-Core CPU Util. (%)", "threshold": CPU_UTIL_LIMIT_PCT},
]

_FLOAT = r"[\d.]+"


def format_number(value) -> str:
    """Mirror speckle_test-v2.cpp's formatDouble(): integral values render without a decimal point."""
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.1f}"


def video_filename(video_type, resolution, source_fps, target_fps, kernel_size, exposure_ms) -> str:
    width = height = int(resolution)
    return (
        f"speckle_{video_type}_{width}x{height}"
        f"_s{format_number(source_fps)}_t{format_number(target_fps)}"
        f"_k{kernel_size}_e{format_number(exposure_ms)}.mp4"
    )


def run(cmd) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=SHELL_DIR)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(str(c) for c in cmd)}\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    return result.stdout


def parse_speckle_test_output(stdout: str) -> dict:
    def find(pattern):
        match = re.search(pattern, stdout)
        return float(match.group(1)) if match else None

    return {
        "frames_processed": find(rf"Frames Processed:\s+({_FLOAT})"),
        "total_execution_s": find(rf"Total Execution:\s+({_FLOAT})\s*s"),
        "avg_kernel_time_ms": find(rf"Avg Kernel Time:\s+({_FLOAT})\s*ms/frame"),
        "min_kernel_time_ms": find(rf"Min Kernel Time:\s+({_FLOAT})\s*ms"),
        "max_kernel_time_ms": find(rf"Max Kernel Time:\s+({_FLOAT})\s*ms"),
        "processing_speed_fps": find(rf"Processing Speed:\s+({_FLOAT})\s*FPS"),
        "data_throughput_gbps": find(rf"Data Throughput:\s+({_FLOAT})\s*GB/s"),
    }


def parse_speckle_analysis_output(stdout: str) -> dict:
    def find(pattern):
        match = re.search(pattern, stdout)
        return float(match.group(1)) if match else None

    def table_row(label, unit):
        pattern = (
            rf"{label}\s+[<>=]*\s*({_FLOAT})\s*{unit}\s+"
            rf"({_FLOAT})\s*{unit}\s+\[\s*(\w+)\s*\]"
        )
        match = re.search(pattern, stdout)
        if not match:
            return None, None
        return float(match.group(2)), match.group(3)

    achieved_fps, fps_status = table_row(r"Frame Rate \(FPS\)", "FPS")
    mean_delay_ms, delay_status = table_row("Frame Process Time", "ms")
    jitter_std_ms, jitter_status = table_row("Latency Jitter", "ms")
    power_w, power_status = table_row("Power Consumption", "W")
    per_core_cpu_util, cpu_status = table_row(r"Per-Core CPU Util\.", "%")
    cores_match = re.search(r"Cores Utilized\s+(\d+)\s*Cores\s+(\d+)\s*Cores\s+\[\s*(\w+)\s*\]", stdout)

    return {
        "avg_K": find(r"Average Speckle Contrast \(K\):\s+([\d.]+)"),
        "avg_tau_c_ms": find(r"Decorrelation Time \([^)]*\):\s+([\d.]+)\s*ms"),
        "achieved_fps": achieved_fps,
        "fps_status": fps_status,
        "mean_delay_ms": mean_delay_ms,
        "delay_status": delay_status,
        "jitter_std_ms": jitter_std_ms,
        "jitter_status": jitter_status,
        # ACTUAL_POWER_W is a hardcoded literal in speckle_analysis.cpp (never measured), and its
        # status is always "PASS" -- kept for CSV completeness but excluded from all summaries/plots.
        "power_w": power_w,
        "power_status": power_status,
        "per_core_cpu_util": per_core_cpu_util,
        "cpu_status": cpu_status,
        "cores_status": cores_match.group(3) if cores_match else None,
    }


def run_generation_sweep() -> list[dict]:
    rows = []
    for target_fps, kernel_size in itertools.product(TARGET_FPS, KERNEL_SIZE):
        filename = video_filename(VIDEO_TYPE, RESOLUTION, SOURCE_FPS, target_fps, kernel_size, EXPOSURE_MS)
        stdout = run([
            SPECKLE_TEST, VIDEO_TYPE, RESOLUTION,
            str(SOURCE_FPS), str(target_fps), str(kernel_size), str(EXPOSURE_MS),
        ])
        rows.append({
            "stage": "generation",
            "target_fps": target_fps,
            "kernel_size": kernel_size,
            "video_path": str(SHELL_DIR / filename),
            **parse_speckle_test_output(stdout),
        })
    return rows


def run_analysis_sweep(generation_rows: list[dict]) -> list[dict]:
    video_by_config = {
        (row["target_fps"], row["kernel_size"]): row["video_path"] for row in generation_rows
    }
    rows = []
    for num_cores, target_fps, kernel_size in itertools.product(NUM_CORES, TARGET_FPS, KERNEL_SIZE):
        video_path = video_by_config[(target_fps, kernel_size)]
        stdout = run([
            SPECKLE_ANALYSIS, video_path,
            str(num_cores), str(target_fps), str(kernel_size), str(EXPOSURE_MS),
        ])
        rows.append({
            "stage": "analysis",
            "num_cores": num_cores,
            "target_fps": target_fps,
            "kernel_size": kernel_size,
            **parse_speckle_analysis_output(stdout),
        })
    return rows


def summarize_clinical_targets(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Pass/fail tally per clinical metric (power consumption excluded, see METRIC_SPECS)."""
    rows = []
    for metric in METRIC_SPECS:
        status = analysis_df[metric["status_col"]]
        num_pass = int((status == "PASS").sum())
        num_fail = int((status == "FAIL").sum())
        total = num_pass + num_fail
        rows.append({
            "metric": metric["label"],
            "num_pass": num_pass,
            "num_fail": num_fail,
            "pass_rate": num_pass / total if total else float("nan"),
        })
    summary_df = pd.DataFrame(rows)

    print("\nClinical Target Summary (power consumption excluded -- not a real measurement):")
    print(summary_df.to_string(index=False))

    return summary_df


def find_clinical_failures(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Full rows where any real clinical metric (excluding power) reports FAIL."""
    status_cols = [metric["status_col"] for metric in METRIC_SPECS]
    failing_mask = (analysis_df[status_cols] == "FAIL").any(axis=1)
    columns = ["num_cores", "target_fps", "kernel_size"] + status_cols
    return analysis_df.loc[failing_mask, columns]


def _status_legend_handles(marker="o"):
    """Proxy legend entries explaining the green/red PASS/FAIL marker or border coloring."""
    return [
        Line2D([0], [0], marker=marker, linestyle="none", markerfacecolor=STATUS_GOOD,
               markeredgecolor=INK_PRIMARY, markersize=8, label="PASS"),
        Line2D([0], [0], marker=marker, linestyle="none", markerfacecolor=STATUS_CRITICAL,
               markeredgecolor=INK_PRIMARY, markersize=8, label="FAIL"),
    ]


def _style_axes(ax):
    ax.set_facecolor(CHART_SURFACE)
    ax.tick_params(colors=INK_MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRIDLINE)
    ax.xaxis.label.set_color(INK_PRIMARY)
    ax.yaxis.label.set_color(INK_PRIMARY)
    ax.title.set_color(INK_PRIMARY)


def plot_analysis_3d(analysis_df: pd.DataFrame, metric: dict, filename: str, title: str) -> None:
    fig = plt.figure(figsize=(9, 7), facecolor=CHART_SURFACE)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(CHART_SURFACE)
    ax.view_init(elev=25, azim=-60)

    cmap = LinearSegmentedColormap.from_list("sequential_blue", SEQUENTIAL_BLUE_STEPS)
    edge_colors = analysis_df[metric["status_col"]].map(
        {"PASS": STATUS_GOOD, "FAIL": STATUS_CRITICAL}
    ).fillna(INK_MUTED)

    scatter = ax.scatter(
        analysis_df["num_cores"], analysis_df["target_fps"], analysis_df["kernel_size"],
        c=analysis_df[metric["col"]], cmap=cmap, s=60, edgecolors=edge_colors, linewidths=1.0,
    )
    ax.set_xlabel("Num Cores")
    ax.set_ylabel("Target FPS")
    ax.set_zlabel("Kernel Size")
    ax.set_title(title, color=INK_PRIMARY)
    ax.xaxis.label.set_color(INK_PRIMARY)
    ax.yaxis.label.set_color(INK_PRIMARY)
    ax.zaxis.label.set_color(INK_PRIMARY)

    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.7, pad=0.1)
    colorbar.set_label(metric["label"], color=INK_PRIMARY)
    colorbar.ax.yaxis.set_tick_params(color=INK_MUTED)
    plt.setp(colorbar.ax.get_yticklabels(), color=INK_MUTED)

    legend = ax.legend(handles=_status_legend_handles(), title="Marker edge = clinical target",
                        loc="upper left", fontsize=8, framealpha=0.9)
    legend.get_title().set_fontsize(8)

    fig.savefig(PLOTS_DIR / filename, dpi=150, facecolor=CHART_SURFACE)
    plt.close(fig)


def plot_analysis_3d_family(analysis_df: pd.DataFrame) -> None:
    for metric in METRIC_SPECS:
        filename = "analysis_3d_full_factorial.png" if metric["slug"] == "fps" else f"analysis_3d_{metric['slug']}.png"
        title = f"speckle_analysis full-factorial sweep — {metric['label']}"
        plot_analysis_3d(analysis_df, metric, filename, title)


def plot_metric_facets(analysis_df: pd.DataFrame, metric: dict, filename: str) -> None:
    """Small-multiples 2D heatmap (target_fps x num_cores), one panel per kernel_size."""
    kernel_sizes = sorted(analysis_df["kernel_size"].unique())
    ncols = 3
    nrows = -(-len(kernel_sizes) // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), facecolor=CHART_SURFACE, squeeze=False,
    )
    cmap = LinearSegmentedColormap.from_list("sequential_blue", SEQUENTIAL_BLUE_STEPS)
    vmin, vmax = analysis_df[metric["col"]].min(), analysis_df[metric["col"]].max()

    mesh = None
    for idx, kernel_size in enumerate(kernel_sizes):
        ax = axes[idx // ncols][idx % ncols]
        panel = analysis_df[analysis_df["kernel_size"] == kernel_size]
        pivot = panel.pivot(index="target_fps", columns="num_cores", values=metric["col"])
        status_pivot = panel.pivot(index="target_fps", columns="num_cores", values=metric["status_col"])

        mesh = ax.imshow(pivot.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", origin="lower")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Num Cores")
        ax.set_ylabel("Target FPS")
        ax.set_title(f"kernel_size = {kernel_size}", color=INK_PRIMARY)

        for (row, col), value in np.ndenumerate(pivot.values):
            ax.text(col, row, f"{value:.1f}", ha="center", va="center", fontsize=7, color=INK_PRIMARY)
            status = status_pivot.values[row, col]
            border_color = STATUS_GOOD if status == "PASS" else STATUS_CRITICAL if status == "FAIL" else None
            if border_color:
                ax.add_patch(Rectangle(
                    (col - 0.5, row - 0.5), 1, 1, fill=False, edgecolor=border_color, linewidth=2,
                ))
        _style_axes(ax)

    for idx in range(len(kernel_sizes), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    if mesh is not None:
        colorbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.7, pad=0.02)
        colorbar.set_label(metric["label"], color=INK_PRIMARY)
        colorbar.ax.yaxis.set_tick_params(color=INK_MUTED)
        plt.setp(colorbar.ax.get_yticklabels(), color=INK_MUTED)

    fig.suptitle(f"{metric['label']} — faceted by kernel size", color=INK_PRIMARY)
    legend_handles = [
        Patch(facecolor="none", edgecolor=STATUS_GOOD, linewidth=2, label="PASS"),
        Patch(facecolor="none", edgecolor=STATUS_CRITICAL, linewidth=2, label="FAIL"),
    ]
    fig.legend(handles=legend_handles, title="Cell border = clinical target",
               loc="upper right", fontsize=8, framealpha=0.9)
    fig.savefig(PLOTS_DIR / filename, dpi=150, facecolor=CHART_SURFACE)
    plt.close(fig)


def plot_metric_facets_family(analysis_df: pd.DataFrame) -> None:
    for metric in METRIC_SPECS:
        plot_metric_facets(analysis_df, metric, f"heatmap_facets_{metric['slug']}.png")


# OFAT: one-factor-at-a-time
def plot_ofat_metric(df: pd.DataFrame, x_col: str, metric: dict, title: str, filename: str) -> None:
    data = df.sort_values(x_col)
    y_col, status_col, threshold = metric["col"], metric["status_col"], metric["threshold"]

    fig, ax = plt.subplots(figsize=(7, 5), facecolor=CHART_SURFACE)

    y_span = data[y_col].max() - data[y_col].min()
    y_margin = y_span * 0.15 if y_span else 1.0
    y_lo, y_hi = data[y_col].min() - y_margin, data[y_col].max() + y_margin
    ax.set_ylim(y_lo, y_hi)

    if y_col == "achieved_fps" and x_col == "target_fps":
        # The target itself is the swept factor here, so the pass zone is above the y = x diagonal.
        ax.fill_between(data[x_col], data[x_col], y_hi, color=STATUS_GOOD, alpha=0.08)
        ax.fill_between(data[x_col], y_lo, data[x_col], color=STATUS_CRITICAL, alpha=0.08)
        ax.plot(data[x_col], data[x_col], color=INK_MUTED, linestyle="--", linewidth=1,
                 label="Clinical Target (FPS >= Target)")
    else:
        target_value = threshold if threshold is not None else BASELINE_TARGET_FPS
        pass_below = y_col != "achieved_fps"  # lower-is-better metrics pass below their threshold
        pass_span, fail_span = (
            ((y_lo, target_value), (target_value, y_hi)) if pass_below
            else ((target_value, y_hi), (y_lo, target_value))
        )
        ax.axhspan(*pass_span, color=STATUS_GOOD, alpha=0.08)
        ax.axhspan(*fail_span, color=STATUS_CRITICAL, alpha=0.08)
        ax.axhline(target_value, color=INK_MUTED, linestyle="--", linewidth=1,
                    label=f"Clinical Target ({target_value:g})")

    ax.plot(data[x_col], data[y_col], color=SERIES_BLUE, linewidth=2)
    marker_colors = data[status_col].map({"PASS": STATUS_GOOD, "FAIL": STATUS_CRITICAL}).fillna(INK_MUTED)
    ax.scatter(data[x_col], data[y_col], c=marker_colors, s=50, zorder=3, edgecolors=INK_PRIMARY, linewidths=0.5)

    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(metric["label"])
    ax.set_title(title)
    ax.grid(True, color=GRIDLINE, linewidth=1)
    ax.set_axisbelow(True)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles + _status_legend_handles(), loc="best", fontsize=8, framealpha=0.9)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=150, facecolor=CHART_SURFACE)
    plt.close(fig)


def plot_ofat_family(analysis_df: pd.DataFrame) -> None:
    slices = {
        "num_cores": analysis_df[
            (analysis_df["target_fps"] == BASELINE_TARGET_FPS)
            & (analysis_df["kernel_size"] == BASELINE_KERNEL_SIZE)
        ],
        "target_fps": analysis_df[
            (analysis_df["num_cores"] == BASELINE_NUM_CORES)
            & (analysis_df["kernel_size"] == BASELINE_KERNEL_SIZE)
        ],
        "kernel_size": analysis_df[
            (analysis_df["num_cores"] == BASELINE_NUM_CORES)
            & (analysis_df["target_fps"] == BASELINE_TARGET_FPS)
        ],
    }
    baseline_labels = {
        "num_cores": f"target_fps={BASELINE_TARGET_FPS}, kernel_size={BASELINE_KERNEL_SIZE}",
        "target_fps": f"num_cores={BASELINE_NUM_CORES}, kernel_size={BASELINE_KERNEL_SIZE}",
        "kernel_size": f"num_cores={BASELINE_NUM_CORES}, target_fps={BASELINE_TARGET_FPS}",
    }

    for factor, data in slices.items():
        for metric in METRIC_SPECS:
            title = f"{metric['label']} vs. {factor.replace('_', ' ').title()} ({baseline_labels[factor]})"
            filename = f"ofat_{factor}_{metric['slug']}.png"
            plot_ofat_metric(data, factor, metric, title, filename)


def plot_results(analysis_df: pd.DataFrame) -> None:
    plot_analysis_3d_family(analysis_df)
    plot_metric_facets_family(analysis_df)
    plot_ofat_family(analysis_df)


def _run_build_step(script_path: Path, args: list, step_name: str) -> None:
    result = subprocess.run([script_path, *args], capture_output=True, text=True, cwd=SHELL_DIR)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to {step_name} ({result.returncode})\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )


if __name__ == '__main__':
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Speckle test will generate videos and run performance metrics
    if not SPECKLE_TEST.exists():
        if not COMPILE_SPECKLE_TEST.exists():
            raise FileNotFoundError("Could not compile Speckle Test")
        _run_build_step(COMPILE_SPECKLE_TEST, ["--opencv"], "compile speckle_test")

    if not SPECKLE_ANALYSIS.exists():
        if not BUILD_SPECKLE_ANALYSIS.exists():
            raise FileNotFoundError("Could not build Speckle Analysis")
        _run_build_step(BUILD_SPECKLE_ANALYSIS, [], "build speckle_analysis")

    # Capture performance boundaries of speckle_test and speckle_analysis with varied arguments
    generation_rows = run_generation_sweep()
    analysis_rows = run_analysis_sweep(generation_rows)

    generation_df = pd.DataFrame(generation_rows)
    analysis_df = pd.DataFrame(analysis_rows)
    generation_df.to_csv(RESULTS_DIR / "generation_results.csv", index=False)
    analysis_df.to_csv(RESULTS_DIR / "analysis_results.csv", index=False)

    clinical_summary_df = summarize_clinical_targets(analysis_df)
    clinical_summary_df.to_csv(RESULTS_DIR / "clinical_target_summary.csv", index=False)
    find_clinical_failures(analysis_df).to_csv(RESULTS_DIR / "clinical_target_failures.csv", index=False)

    plot_results(analysis_df)
