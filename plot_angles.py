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
ANGLE = "knee"                 # pierna cercana a la cámara (near_side)
RANGE = (80, 140)              # rango correcto en ángulo interior (= 220-280° del paper, 360 - θ)
OUT_DIR = Path("plots")


def main():
    good = pd.read_csv(CSV_GOOD)
    bad = pd.read_csv(CSV_BAD)

    low, high = RANGE

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(good["frame"], good[ANGLE], color="blue", label="Correcto")
    ax.plot(bad["frame"], bad[ANGLE], color="red", label="Incorrecto")

    range_label = f"Rango correcto: {low}-{high}° (= {360 - high}-{360 - low}° del paper)"
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
