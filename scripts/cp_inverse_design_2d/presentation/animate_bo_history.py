#!/usr/bin/env python3
"""Create the three-panel BO history animation from 28 real FEM evaluations."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_SCRIPT_DIR = SCRIPT_DIR.parent
for directory in (SCRIPT_DIR, PARENT_SCRIPT_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from presentation_utils import (  # noqa: E402
    COLOR_BEST,
    COLOR_BO,
    COLOR_CURRENT,
    COLOR_DOE,
    COLOR_TARGET,
    GP_OBJECTIVE_FLOOR,
    MESH_FILES,
    PRESENTATION,
    audit_history,
    common_inputs,
    configure_style,
    curve_arrays,
    draw_texture,
    file_sha256,
    load_real_curve,
    manifest_record,
    mesh_faces,
    objective_gp_surface,
    source_path,
    verification_entry_sources,
)


def _frame_duration_ms(call: int, total: int) -> int:
    if call == total:
        return 3000
    return 550 if call <= 9 else 800


def _render_frame(
    path: Path,
    record: dict[str, object],
    records: list[dict[str, object]],
    target_curve: list[dict[str, float]],
    mesh: tuple[np.ndarray, np.ndarray, np.ndarray],
    surface: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    log_limits: tuple[float, float],
    stress_upper: float,
) -> None:
    call = int(record["call"])
    completed = records[:call]
    best = records[int(record["best_call"]) - 1]
    target_x, target_y = curve_arrays(target_curve)
    current_x, current_y = curve_arrays(record["curve"])
    best_x, best_y = curve_arrays(best["curve"])

    fig = plt.figure(figsize=(16.0, 9.0), dpi=120)
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=(0.91, 1.22, 1.08),
        left=0.045,
        right=0.965,
        bottom=0.11,
        top=0.84,
        wspace=0.25,
    )
    texture_axis = fig.add_subplot(grid[0, 0])
    response_axis = fig.add_subplot(grid[0, 1])
    design_axis = fig.add_subplot(grid[0, 2])

    stage_label = "Initial DOE" if record["stage"] == "initial" else "Sequential BO"
    draw_texture(
        texture_axis,
        float(record["eta"]),
        float(record["theta_deg"]),
        "Current candidate texture\n"
        rf"$\eta$={float(record['eta']):.4f}, $\theta$={float(record['theta_deg']):+.2f}°",
        mesh,
    )
    texture_axis.grid(False)
    texture_axis.text(
        0.5,
        -0.14,
        "Fixed Neper geometry · fixed cubic IPF-X key",
        transform=texture_axis.transAxes,
        ha="center",
        fontsize=9,
    )

    response_axis.plot(
        target_x,
        target_y,
        color=COLOR_TARGET,
        linewidth=2.6,
        label="Target — real FEM",
        zorder=2,
    )
    response_axis.plot(
        current_x,
        current_y,
        color=COLOR_CURRENT,
        linewidth=2.3,
        label="Current — real FEM",
        zorder=3,
    )
    response_axis.plot(
        best_x,
        best_y,
        color=COLOR_BEST,
        linewidth=2.2,
        linestyle="--",
        label=f"Best so far — call {int(best['call'])}",
        zorder=4,
    )
    response_axis.set(
        xlim=(-0.0003, 0.0206),
        ylim=(-0.02 * stress_upper, stress_upper),
        xlabel=r"Macroscopic nominal strain $\bar F_{11}-1$",
        ylabel=r"Macroscopic PK1 $\bar P_{11}$ (MPa)",
        title="Real stress–strain responses",
    )
    response_axis.legend(loc="lower right", fontsize=8.8)
    response_axis.text(
        0.035,
        0.965,
        f"Current proof error: {100*float(record['relative_error_proof']):6.3f}%\n"
        f"Current flow error:  {100*float(record['relative_error_flow']):6.3f}%\n"
        rf"Current $J$: {float(record['objective_J']):.3e}"
        "\n"
        rf"Best $J$:    {float(record['best_objective_J']):.3e}",
        transform=response_axis.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "0.78",
            "alpha": 0.94,
        },
    )

    if surface is not None:
        eta_mesh, theta_mesh, objective = surface
        log_objective = np.log10(np.maximum(objective, GP_OBJECTIVE_FLOOR))
        levels = np.linspace(log_limits[0], log_limits[1], 22)
        image = design_axis.contourf(
            eta_mesh,
            theta_mesh,
            log_objective,
            levels=levels,
            cmap="magma_r",
            extend="both",
        )
        colorbar = fig.colorbar(image, ax=design_axis, fraction=0.046, pad=0.03)
        colorbar.set_label(r"Current GP $\log_{10}(J)$", fontsize=9)
        colorbar.ax.tick_params(labelsize=8)
        posterior_label = f"GP posterior trained on calls 1–{call}"
    else:
        design_axis.set_facecolor("0.94")
        posterior_label = "Neutral design plane — fewer than 3 observations"

    eta_completed = [float(item["eta"]) for item in completed]
    theta_completed = [float(item["theta_deg"]) for item in completed]
    design_axis.plot(
        eta_completed,
        theta_completed,
        color="0.50",
        linewidth=1.0,
        alpha=0.75,
        zorder=3,
    )
    for completed_record in completed:
        design_axis.scatter(
            completed_record["eta"],
            completed_record["theta_deg"],
            marker="s" if completed_record["stage"] == "initial" else "o",
            s=42,
            facecolor=COLOR_DOE if completed_record["stage"] == "initial" else COLOR_BO,
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
    design_axis.scatter(
        record["eta"],
        record["theta_deg"],
        marker="*",
        s=230,
        facecolor=COLOR_CURRENT,
        edgecolor="black",
        linewidth=0.9,
        zorder=6,
    )
    design_axis.scatter(
        best["eta"],
        best["theta_deg"],
        marker="*",
        s=150,
        facecolor=COLOR_BEST,
        edgecolor="black",
        linewidth=0.8,
        zorder=7,
    )
    design_axis.set(
        xlim=(0.1, 0.9),
        ylim=(-45.0, 45.0),
        xlabel=r"Texture concentration $\eta$",
        ylabel=r"Preferred rotation $\theta$ (deg)",
        title="Design space\n" + posterior_label,
    )
    design_axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor=COLOR_DOE,
                markeredgecolor="white",
                label="DOE",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=COLOR_BO,
                markeredgecolor="white",
                label="Sequential BO",
            ),
            Line2D(
                [0],
                [0],
                marker="*",
                color="none",
                markerfacecolor=COLOR_CURRENT,
                markeredgecolor="black",
                markersize=11,
                label="Current",
            ),
            Line2D(
                [0],
                [0],
                marker="*",
                color="none",
                markerfacecolor=COLOR_BEST,
                markeredgecolor="black",
                markersize=10,
                label="Best so far",
            ),
        ],
        loc="lower left",
        fontsize=8,
        ncol=2,
        frameon=True,
        facecolor="white",
        framealpha=0.90,
    )

    fig.suptitle(
        f"Bayesian inverse design · Evaluation {call}/28 · {stage_label}",
        fontsize=18,
        y=0.955,
    )
    fig.text(
        0.5,
        0.90,
        "Current texture, current real FEM response, best-so-far real FEM response, and posterior using only completed calls",
        ha="center",
        fontsize=11,
    )
    if call == len(records):
        fig.text(
            0.5,
            0.025,
            rf"TARGET MET · best $\eta$={float(best['eta']):.6f}, $\theta$={float(best['theta_deg']):+.4f}° · "
            f"proof error={100*float(best['relative_error_proof']):.3f}% · "
            f"flow error={100*float(best['relative_error_flow']):.3f}%",
            ha="center",
            fontsize=13,
            color="#007A3D",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "#E8F5E9",
                "edgecolor": "#007A3D",
            },
        )
    else:
        fig.text(
            0.5,
            0.030,
            "Each frame is one executed design; the connecting path indicates evaluation order, not physical evolution.",
            ha="center",
            fontsize=9.5,
        )
    fig.savefig(path, dpi=120, facecolor="white")
    plt.close(fig)


def _encode_mp4(frame_paths: list[Path], durations_ms: list[int], output: Path) -> None:
    concat_path = frame_paths[0].parent / "frames.ffconcat"
    variable_rate_path = frame_paths[0].parent / "variable_rate.mp4"
    lines = ["ffconcat version 1.0"]
    for frame, duration in zip(frame_paths, durations_ms):
        lines.extend([f"file '{frame}'", f"duration {duration / 1000.0:.6f}"])
    lines.append(f"file '{frame_paths[-1]}'")
    concat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-vsync",
            "vfr",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(variable_rate_path),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(variable_rate_path),
            "-vf",
            "fps=25",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )


def _encode_gif(frame_paths: list[Path], durations_ms: list[int], output: Path) -> None:
    images: list[Image.Image] = []
    for frame in frame_paths:
        with Image.open(frame) as source:
            resized = source.convert("RGB").resize(
                (1280, 720), Image.Resampling.LANCZOS
            )
            images.append(
                resized.quantize(
                    colors=160,
                    method=Image.Quantize.MEDIANCUT,
                    dither=Image.Dither.FLOYDSTEINBERG,
                )
            )
    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=durations_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )


def _probe_video(path: Path) -> dict[str, object]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,r_frame_rate",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(completed.stdout)


def build_bo_animation(audit: dict[str, object] | None = None) -> dict[str, object]:
    configure_style()
    audit = audit or audit_history()
    records = audit["records"]
    inputs = common_inputs()
    target_sources = verification_entry_sources(inputs["verification"]["coarse_target"])
    target_curve = load_real_curve(target_sources["csv"])
    mesh = mesh_faces()

    surfaces: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray] | None] = {}
    for record in records:
        call = int(record["call"])
        surfaces[call] = objective_gp_surface(
            records, through_call=call, eta_count=81, theta_count=91
        )
    objective_values = np.asarray([float(record["objective_J"]) for record in records])
    log_limits = (
        float(np.log10(objective_values.min()) - 0.75),
        float(np.log10(objective_values.max()) + 0.30),
    )
    all_stress = [curve_arrays(target_curve)[1]] + [
        curve_arrays(record["curve"])[1] for record in records
    ]
    stress_upper = 1.08 * max(float(values.max()) for values in all_stress)
    durations_ms = [
        _frame_duration_ms(int(record["call"]), len(records)) for record in records
    ]

    PRESENTATION.mkdir(parents=True, exist_ok=True)
    mp4_path = PRESENTATION / "bo_inverse_design_history.mp4"
    gif_path = PRESENTATION / "bo_inverse_design_history.gif"
    with tempfile.TemporaryDirectory(prefix="spp2489_bo_frames_") as temporary:
        frame_directory = Path(temporary)
        frame_paths: list[Path] = []
        for record in records:
            frame_path = frame_directory / f"frame_{int(record['call']):03d}.png"
            _render_frame(
                frame_path,
                record,
                records,
                target_curve,
                mesh,
                surfaces[int(record["call"])],
                log_limits,
                stress_upper,
            )
            frame_paths.append(frame_path)
        _encode_mp4(frame_paths, durations_ms, mp4_path)
        _encode_gif(frame_paths, durations_ms, gif_path)

    frames: list[dict[str, object]] = []
    geometry_sha256 = file_sha256(MESH_FILES["coarse"])
    for record, duration in zip(records, durations_ms):
        item = manifest_record(record)
        item.update(
            {
                "duration_ms": duration,
                "current_candidate_call": record["call"],
                "current_curve_source": source_path(record["paths"]["csv"]),
                "best_curve_call": record["best_call"],
                "best_curve_source": source_path(
                    records[int(record["best_call"]) - 1]["paths"]["csv"]
                ),
                "posterior_background": (
                    "neutral parameter plane"
                    if int(record["call"]) < 3
                    else f"SingleTaskGP objective posterior trained only on calls 1..{record['call']}"
                ),
                "texture_source": "deterministic SO(3) parameterization at the same eta/theta as the current curve",
                "fixed_geometry_sha256": geometry_sha256,
            }
        )
        frames.append(item)

    mp4_probe = _probe_video(mp4_path)
    return {
        "name": "BO inverse-design history animation",
        "outputs": {
            "gif": {
                "path": source_path(gif_path),
                "size_bytes": gif_path.stat().st_size,
                "sha256": file_sha256(gif_path),
                "width": 1280,
                "height": 720,
                "frame_count": len(records),
            },
            "mp4": {
                "path": source_path(mp4_path),
                "size_bytes": mp4_path.stat().st_size,
                "sha256": file_sha256(mp4_path),
                "probe": mp4_probe,
                "frame_count": len(records),
            },
        },
        "target_curve": {
            "classification": "real MOOSE FCC CP FEM curve",
            "solution_csv": source_path(target_sources["csv"]),
            "run_identifier": target_sources["run_directory"].name,
            "curve_points": len(target_curve),
        },
        "posterior_policy": "Calls 1-2 use a neutral plane; frame k>=3 refits the objective GP using only calls 1..k.",
        "parameter_interpolation": False,
        "fixed_geometry": {
            "mesh": source_path(MESH_FILES["coarse"]),
            "sha256": geometry_sha256,
        },
        "total_duration_ms": sum(durations_ms),
        "final_hold_ms": durations_ms[-1],
        "frames": frames,
    }


def main() -> int:
    metadata = build_bo_animation()
    print(
        json.dumps(
            {key: value for key, value in metadata.items() if key != "frames"}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
