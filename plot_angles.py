"""
Gráfica ángulo vs frame: vídeo correcto (azul) vs incorrecto (rojo) del mismo sujeto,
con líneas horizontales en los límites del rango del paper.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Configuración
CSV_GOOD = Path("angles/subject_022_squat_good_side.csv")
CSV_BAD = Path("angles/subject_022_squat_bad_side.csv")
ANGLE = "knee_left"            # pierna cercana a la cámara
PAPER_RANGE = (220, 280)       # rango correcto del paper (squat)
MIRROR = True                  # el sujeto mira a la izquierda -> nuestro ángulo = 360 - θ del paper
OUT_DIR = Path("plots")


def main():
    good = pd.read_csv(CSV_GOOD)
    bad = pd.read_csv(CSV_BAD)

    low, high = sorted((PAPER_RANGE[0], PAPER_RANGE[1])) if not MIRROR else sorted((360 - PAPER_RANGE[0], 360 - PAPER_RANGE[1]))

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(good["frame"], good[ANGLE], color="blue", label="Correcto")
    ax.plot(bad["frame"], bad[ANGLE], color="red", label="Incorrecto")

    # Rango del paper; la etiqueta indica si está en espejo
    range_label = f"Rango paper: {low}-{high}°"
    if MIRROR:
        range_label += f" (espejo de {PAPER_RANGE[0]}-{PAPER_RANGE[1]}°)"
    ax.axhline(low, color="gray", linestyle="--", alpha=0.7, label=range_label)
    ax.axhline(high, color="gray", linestyle="--", alpha=0.7)

    # Sujeto y ejercicio a partir del nombre: subject_022_squat_good_side -> "022", "squat"
    parts = CSV_GOOD.stem.split("_")
    subject, exercise = parts[1], "_".join(parts[2:-2])
    ax.set_title(f"Sujeto {subject} · {exercise} · {ANGLE}")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Ángulo (grados)")
    # Leyenda fuera del gráfico para no tapar los valles ni los picos
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(True)
    fig.tight_layout()

    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUT_DIR / f"subject_{subject}_{exercise}_{ANGLE}.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
