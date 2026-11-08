"""
Gráficas de las repeticiones de reps.csv, en plots/:
  (a) <video>_reps.png: rodilla, cadera y torso con las repeticiones numeradas.
  (b) squat_correct_vs_incorrect.png: un punto por repetición válida, por métrica y sujeto.
Cada panel usa la validez de su propia métrica (valid_knee, valid_hip, valid_torso).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # sin ventanas: guarda las figuras directamente
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from reps import get_fps

# Configuración
METADATA = Path("dataset/metadata.csv")
DATASET = Path("dataset")
ANGLES_DIR = Path("angles")
REPS_CSV = Path("reps.csv")
OUT_DIR = Path("plots")
EXERCISE = "squats"

# (columna, nombre, límites y) de los 3 paneles
PANELS = [
    ("knee", "Rodilla", (40, 205)),
    ("hip", "Cadera", (40, 205)),
    ("torso", "Torso", (0, 75)),
]
# (columna de reps.csv, métrica, título)
COMPARE = [
    ("knee_min", "knee", "Rodilla mínima (°)"),
    ("hip_min", "hip", "Cadera mínima (°)"),
    ("torso_max", "torso", "Torso máximo (°)"),
]
LABELS = {"correct": "Correcto", "incorrect": "Incorrecto"}

# Colores: paleta validada para daltonismo en columnas vecinas; cada sujeto lleva además su marcador
SUBJECT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SUBJECT_MARKERS = ["o", "s", "^", "D"]
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
LINE, CRITICAL = "#2a78d6", "#d03b3b"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
})


def is_valid(rep, metric):
    return getattr(rep, f"valid_{metric}")


def rep_label(rep, metric):
    """Número de la repetición: × = no válida en esta métrica, * = cortada por el vídeo."""
    return f"{rep.rep}{'×' if not is_valid(rep, metric) else ''}{'*' if rep.partial else ''}"


def extreme_frame(rep, metric, angles):
    """Frame donde se alcanza el valor que recoge reps.csv para esa métrica."""
    if metric == "knee":
        return rep.frame_min
    window = angles[metric].iloc[rep.frame_inicio:rep.frame_fin + 1]
    return window.idxmax() if metric == "torso" else window.idxmin()


def plot_video(stem, subject, label, angles, reps, fps):
    """(a) Tres paneles con un eje de frames común y las repeticiones numeradas."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 9.5), sharex=True)

    for i, (ax, (metric, name, ylim)) in enumerate(zip(axes, PANELS)):
        for rep in reps.itertuples():
            if not is_valid(rep, metric):
                ax.axvspan(rep.frame_inicio, rep.frame_fin, color=CRITICAL, alpha=0.10, lw=0)
            ax.axvline(rep.frame_inicio, color=MUTED, lw=0.7)
            ax.axvline(rep.frame_fin, color=MUTED, lw=0.7)

        ax.plot(angles["frame"], angles[metric], color=LINE, lw=1.4, solid_joinstyle="round")

        for rep in reps.itertuples():
            valid = is_valid(rep, metric)
            frame = extreme_frame(rep, metric, angles)
            ax.scatter(frame, angles[metric].iloc[frame], zorder=4,
                       s=46 if valid else 54, marker="o" if valid else "X",
                       color=LINE if valid else CRITICAL, edgecolor=SURFACE, linewidth=0.8)
            # Número de repetición arriba del panel, sobre el mínimo de rodilla
            ax.text(rep.frame_min, 0.965, rep_label(rep, metric), ha="center", va="top", fontsize=11,
                    fontweight="bold", transform=ax.get_xaxis_transform())

        # Marcas pequeñas abajo: frames con algún keypoint interpolado de esta métrica
        interp = angles.loc[angles[f"interp_{metric}"], "frame"]
        ax.plot(interp, np.full(len(interp), 0.02), "|", color=INK2, ms=6, mew=0.9,
                transform=ax.get_xaxis_transform())
        ax.set_ylabel(f"{name} (°)")
        ax.set_ylim(ylim)
        n_valid = int(reps[f"valid_{metric}"].sum())
        # El primer panel deja sitio arriba al eje de segundos
        ax.set_title(f"{name}: n = {n_valid} válidas de {len(reps)}", loc="left", fontsize=12,
                     fontweight="bold", pad=44 if i == 0 else 6)

    axes[-1].set_xlabel("Frame")
    axes[-1].set_xlim(0, len(angles) - 1)
    top = axes[0].secondary_xaxis("top", functions=(lambda f: f / fps, lambda s: s * fps))
    top.set_xlabel("Tiempo en el vídeo (s)")

    fig.suptitle(f"Sujeto {subject} · {LABELS[label].lower()} · {len(reps)} repeticiones detectadas",
                 x=0.01, ha="left", fontsize=14, fontweight="bold")
    handles = [
        Line2D([], [], color=LINE, lw=1.6, label="Ángulo"),
        Line2D([], [], color=MUTED, lw=0.8, label="Inicio / fin de repetición"),
        Line2D([], [], marker="o", ls="", color=LINE, label="Extremo válido"),
        Line2D([], [], marker="X", ls="", color=CRITICAL, label="Extremo no válido"),
        Patch(color=CRITICAL, alpha=0.15, label="No válida en esta métrica (N×)"),
        Line2D([], [], marker="|", ls="", color=INK2, label="Frame interpolado"),
        Line2D([], [], ls="", label="N* = cortada por el vídeo"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, fontsize=9.5)
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    fig.savefig(OUT_DIR / f"{stem}_reps.png", dpi=150)
    plt.close(fig)


def plot_comparison(reps):
    """(b) Por métrica: columnas por sujeto dentro de correcto e incorrecto, solo reps válidas en esa métrica."""
    subjects = sorted(reps["subject"].unique())
    rng = np.random.default_rng(0)   # desplazamiento lateral reproducible, solo para no solapar puntos
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    for ax, (column, metric, title) in zip(axes, COMPARE):
        valid = reps[reps[f"valid_{metric}"]]
        ticks, ticklabels = [], []
        for g, (label, name) in enumerate(LABELS.items()):
            offset = g * (len(subjects) + 1)
            for s, subject in enumerate(subjects):
                values = valid.loc[(valid["label"] == label) & (valid["subject"] == subject), column]
                x = offset + s + rng.uniform(-0.2, 0.2, len(values))
                ax.scatter(x, values, s=52, color=SUBJECT_COLORS[s], marker=SUBJECT_MARKERS[s],
                           edgecolor=SURFACE, linewidth=0.8, zorder=3)
                ticks.append(offset + s)
                ticklabels.append(f"{subject}\nn={len(values)}")
            ax.text(offset + (len(subjects) - 1) / 2, -0.17, name, ha="center", va="top", fontsize=12,
                    fontweight="bold", transform=ax.get_xaxis_transform())
        ax.axvline(len(subjects), color=GRID, lw=1)
        ax.set_xticks(ticks, ticklabels, fontsize=9)
        ax.grid(False, axis="x")
        ax.set_title(f"{title} · n = {len(valid)} de {len(reps)}", loc="left", fontsize=12)

    fig.suptitle(f"Sentadilla: correcto frente a incorrecto · {len(reps)} repeticiones detectadas, "
                 "solo las válidas en cada métrica", x=0.01, ha="left", fontsize=14, fontweight="bold")
    handles = [Line2D([], [], marker=m, ls="", color=c, ms=8, label=f"Sujeto {s}")
               for s, c, m in zip(subjects, SUBJECT_COLORS, SUBJECT_MARKERS)]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    fig.savefig(OUT_DIR / "squat_correct_vs_incorrect.png", dpi=150)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    # dtype=str en subject: "004" no debe convertirse en el número 4
    reps = pd.read_csv(REPS_CSV, dtype={"subject": str})
    meta = pd.read_csv(METADATA, dtype=str, keep_default_na=False)
    todo = meta[(meta["subject_ids"] != "") & (meta["exercise"] == EXERCISE)]

    for row in todo.itertuples():
        stem = Path(row.file).stem
        video_reps = reps[(reps["subject"] == row.subject) & (reps["label"] == row.label)]
        angles = pd.read_csv(ANGLES_DIR / f"{stem}.csv")
        fps = get_fps(DATASET / row.exercise / row.label / row.file)
        plot_video(stem, row.subject, row.label, angles, video_reps, fps)

    plot_comparison(reps)
    print(f"{len(todo) + 1} figuras en {OUT_DIR}/")


if __name__ == "__main__":
    main()
