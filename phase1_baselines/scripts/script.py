"""
file: script.py
author: Chiagozie Okoye
description:
    Sweeps speckle_test (video generation) over TARGET_FPS x KERNEL_SIZE and
    speckle_analysis (multicore LSCI analysis) over NUM_CORES x TARGET_FPS x
    KERNEL_SIZE, capturing each run's stdout performance metrics into pandas
    DataFrames, persisting them as CSV, and plotting the results.
"""
from pathlib import Path
import itertools
import re
import subprocess

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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
KERNEL_SIZE = [x for x in range(5, 10 + 1)]

# Baselines used to slice the full-factorial analysis grid into one-factor-at-a-time plots
BASELINE_NUM_CORES = 4
BASELINE_TARGET_FPS = 80
BASELINE_KERNEL_SIZE = 5

# dataviz reference palette (light mode) - see references/palette.md
CHART_SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SERIES_BLUE = "#2a78d6"
SEQUENTIAL_BLUE_STEPS = [
    "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b",
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


def _style_axes(ax):
    ax.set_facecolor(CHART_SURFACE)
    ax.tick_params(colors=INK_MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRIDLINE)
    ax.xaxis.label.set_color(INK_PRIMARY)
    ax.yaxis.label.set_color(INK_PRIMARY)
    ax.title.set_color(INK_PRIMARY)


def plot_analysis_3d(analysis_df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(9, 7), facecolor=CHART_SURFACE)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(CHART_SURFACE)

    cmap = LinearSegmentedColormap.from_list("sequential_blue", SEQUENTIAL_BLUE_STEPS)
    scatter = ax.scatter(
        analysis_df["num_cores"], analysis_df["target_fps"], analysis_df["kernel_size"],
        c=analysis_df["achieved_fps"], cmap=cmap, s=40, edgecolors="none",
    )
    ax.set_xlabel("Num Cores")
    ax.set_ylabel("Target FPS")
    ax.set_zlabel("Kernel Size")
    ax.set_title("speckle_analysis full-factorial sweep", color=INK_PRIMARY)
    ax.xaxis.label.set_color(INK_PRIMARY)
    ax.yaxis.label.set_color(INK_PRIMARY)
    ax.zaxis.label.set_color(INK_PRIMARY)

    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.7, pad=0.1)
    colorbar.set_label("Achieved FPS", color=INK_PRIMARY)
    colorbar.ax.yaxis.set_tick_params(color=INK_MUTED)
    plt.setp(colorbar.ax.get_yticklabels(), color=INK_MUTED)

    fig.savefig(PLOTS_DIR / "analysis_3d_full_factorial.png", dpi=150, facecolor=CHART_SURFACE)
    plt.close(fig)


def plot_ofat_line(df: pd.DataFrame, x_col: str, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=CHART_SURFACE)
    data = df.sort_values(x_col)
    ax.plot(data[x_col], data["achieved_fps"], color=SERIES_BLUE, linewidth=2, marker="o", markersize=6)
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel("Achieved FPS")
    ax.set_title(title)
    ax.grid(True, color=GRIDLINE, linewidth=1)
    ax.set_axisbelow(True)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=150, facecolor=CHART_SURFACE)
    plt.close(fig)


def plot_results(analysis_df: pd.DataFrame) -> None:
    plot_analysis_3d(analysis_df)

    num_cores_slice = analysis_df[
        (analysis_df["target_fps"] == BASELINE_TARGET_FPS)
        & (analysis_df["kernel_size"] == BASELINE_KERNEL_SIZE)
    ]
    plot_ofat_line(
        num_cores_slice, "num_cores",
        f"Achieved FPS vs. Num Cores (target_fps={BASELINE_TARGET_FPS}, kernel_size={BASELINE_KERNEL_SIZE})",
        "ofat_num_cores.png",
    )

    target_fps_slice = analysis_df[
        (analysis_df["num_cores"] == BASELINE_NUM_CORES)
        & (analysis_df["kernel_size"] == BASELINE_KERNEL_SIZE)
    ]
    plot_ofat_line(
        target_fps_slice, "target_fps",
        f"Achieved FPS vs. Target FPS (num_cores={BASELINE_NUM_CORES}, kernel_size={BASELINE_KERNEL_SIZE})",
        "ofat_target_fps.png",
    )


if __name__ == '__main__':
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Speckle test will generate videos and run performance metrics
    if not SPECKLE_TEST.exists():
        if not COMPILE_SPECKLE_TEST.exists():
            raise FileNotFoundError("Could not compile Speckle Test")
        subprocess.run([COMPILE_SPECKLE_TEST, "--opencv"])

    if not SPECKLE_ANALYSIS.exists():
        if not BUILD_SPECKLE_ANALYSIS.exists():
            raise FileNotFoundError("Could not build Speckle Analysis")
        subprocess.run([BUILD_SPECKLE_ANALYSIS])

    # Capture performance boundaries of speckle_test and speckle_analysis with varied arguments
    generation_rows = run_generation_sweep()
    analysis_rows = run_analysis_sweep(generation_rows)

    generation_df = pd.DataFrame(generation_rows)
    analysis_df = pd.DataFrame(analysis_rows)
    generation_df.to_csv(RESULTS_DIR / "generation_results.csv", index=False)
    analysis_df.to_csv(RESULTS_DIR / "analysis_results.csv", index=False)

    plot_results(analysis_df)
